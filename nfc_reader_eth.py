import nfc
import nfc.tag.tt4 # For Type4ATag specific definitions if needed
import time
import binascii # To print byte arrays nicely
import ndef # From 'pip install ndeflib'
import urllib.parse
from eth_utils import keccak, to_checksum_address
import threading
import traceback # For detailed error logging

class NFCReaderETH:
    def __init__(self, process_eth_address_callback, reader_path='usb', max_read_chunk=120, debug=False):
        """
        Initializes the NFC Reader.

        Args:
            process_eth_address_callback: A function to be called when an Ethereum address
                                          is successfully derived from an NFC tag's URI.
                                          The callback will receive:
                                          - eth_addr (str): The Ethereum address.
                                          - public_key (str, optional): The public key used.
                                          - uri (str, optional): The URI from which data was extracted.
            reader_path (str): The path to the NFC reader (e.g., 'usb', 'tty:S0:pn532').
            max_read_chunk (int): Maximum number of bytes to read in a single NDEF data APDU.
                                  Default is 120. Adjust based on tag/reader capabilities.
            debug (bool): If True, enables detailed printing of NFC operations. Default is False.
        """
        if not callable(process_eth_address_callback):
            raise ValueError("process_eth_address_callback must be a callable function.")
        
        self.process_eth_address_callback = process_eth_address_callback
        self.reader_path = reader_path
        self.max_read_chunk = max_read_chunk
        self.debug = debug
        
        self.clf = None # Contactless Frontend, managed by the listening thread
        self._running = False # Flag to control the listening loop
        self.nfc_thread = None

        if self.debug:
            print(f"[NFCReaderETH DEBUG] Initialized with: reader_path='{reader_path}', "
                  f"max_read_chunk={max_read_chunk}, debug={debug}")

    @staticmethod
    def _public_key_to_eth_address(pubkey_hex: str) -> str:
        """
        Convert an uncompressed public key (hex string starting with 04) to Ethereum address.
        """
        if not pubkey_hex.startswith('04') or len(pubkey_hex) != 130:
            raise ValueError("Invalid uncompressed public key format (must be 04... and 130 chars).")
        
        pubkey_bytes = bytes.fromhex(pubkey_hex[2:])  # Skip '04' prefix
        keccak_digest = keccak(pubkey_bytes)
        address_bytes = keccak_digest[-20:]  # Last 20 bytes
        return to_checksum_address('0x' + address_bytes.hex())

    def _process_uri(self, uri: str):
        """
        Parses a URI, extracts a public key, converts it to an Ethereum address,
        and then calls the user-provided callback.
        """
        unquoted_uri = urllib.parse.unquote(uri)
        if self.debug:
            print(f"    [*] Processing URI: {unquoted_uri}")
        eth_addr = None
        pk1_val = None
        try:
            parsed_url = urllib.parse.urlparse(unquoted_uri)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            pk1_val = query_params.get('pk1', [None])[0]

            if pk1_val:
                if self.debug:
                    print(f"        [*] Extracted pk1: {pk1_val[:10]}...{pk1_val[-10:] if len(pk1_val) > 20 else pk1_val}")
                eth_addr = self._public_key_to_eth_address(pk1_val)
                if self.debug:
                    print(f"        [*] Derived Ethereum Address: {eth_addr}")
                
                if self.process_eth_address_callback:
                    self.process_eth_address_callback(eth_addr, pk1_val, unquoted_uri)
            else:
                if self.debug:
                    print("        [!] 'pk1' not found in URI query parameters.")

        except ValueError as e: 
            print(f"        [!] Error converting public key: {e}") # Always print conversion errors
        except Exception as e:
            print(f"        [!] Error parsing URI or processing pk1: {e}") # Always print parsing errors
            if self.debug:
                print(f"            URI was: {unquoted_uri}")
                # traceback.print_exc() 

    def _on_connect_apdu(self, tag):
        """
        Handles NFC tag connection and APDU interactions to read NDEF data.
        """
        if self.debug:
            print(f"[-] Card connected: {tag}")
        
        if not tag.is_present:
            if self.debug:
                print("[-] Tag disappeared immediately after connect.")
            return False

        if not isinstance(tag, nfc.tag.tt4.Type4ATag):
            if self.debug:
                print("[-] Tag is not a Type4ATag. APDU interaction might differ or fail.")
            return False 

        try:
            # --- 1. Select NDEF Tag Application ---
            aid_ndef_v1 = bytes.fromhex("D2760000850101")
            aid_ndef_v0 = bytes.fromhex("D2760000850100")
            apdu_select_ndef_app = bytes.fromhex("00A40400") + bytes([len(aid_ndef_v1)]) + aid_ndef_v1 + bytes.fromhex("00")
            
            if self.debug:
                print(f"[*] Sending SELECT NDEF APP (v1.0): {binascii.hexlify(apdu_select_ndef_app).decode()}")
            response = tag.transceive(apdu_select_ndef_app)
            if self.debug:
                print(f"[*] Response: {binascii.hexlify(response).decode()}")

            if not response or not response.endswith(b'\x90\x00'):
                if self.debug:
                    print("[-] Failed to select NDEF Application (v1.0). Trying older AID (v0.x)...")
                apdu_select_ndef_app_v0 = bytes.fromhex("00A40400") + bytes([len(aid_ndef_v0)]) + aid_ndef_v0 + bytes.fromhex("00")
                if self.debug:
                    print(f"[*] Sending SELECT NDEF APP (v0.x): {binascii.hexlify(apdu_select_ndef_app_v0).decode()}")
                response = tag.transceive(apdu_select_ndef_app_v0)
                if self.debug:
                    print(f"[*] Response: {binascii.hexlify(response).decode()}")
                if not response or not response.endswith(b'\x90\x00'):
                    if self.debug:
                        print("[-] Failed to select NDEF Application (v0.x) as well.")
                    return False

            # --- 2. Select Capability Container (CC) File ---
            cc_file_id_bytes = bytes.fromhex("E103")
            apdu_select_cc = bytes.fromhex("00A4000C") + bytes([len(cc_file_id_bytes)]) + cc_file_id_bytes
            
            if self.debug:
                print(f"[*] Sending SELECT CC FILE: {binascii.hexlify(apdu_select_cc).decode()}")
            response = tag.transceive(apdu_select_cc)
            if self.debug:
                print(f"[*] Response: {binascii.hexlify(response).decode()}")
            if not response or not response.endswith(b'\x90\x00'):
                if self.debug:
                    print("[-] Failed to select CC File.")
                return False

            # --- 3. Read CC File ---
            apdu_read_cc = bytes.fromhex("00B000000F") 
            if self.debug:
                print(f"[*] Sending READ CC FILE: {binascii.hexlify(apdu_read_cc).decode()}")
            cc_data_response = tag.transceive(apdu_read_cc)

            if not cc_data_response or not cc_data_response.endswith(b'\x90\x00'):
                if self.debug:
                    print("[-] Failed to read CC File data.")
                return False 
            
            cc_data = cc_data_response[:-2]
            if self.debug:
                print(f"[*] CC Data (first 15 bytes): {binascii.hexlify(cc_data).decode()}")

            # --- 4. Parse CC File (Simplified for NDEF File ID) ---
            ndef_file_id = None
            try:
                if len(cc_data) >= 8: 
                    offset_for_tlv_tag = 7 
                    if cc_data[offset_for_tlv_tag] == 0x04: 
                        if self.debug:
                            print("[*] Found NDEF File Control TLV (0x04) in CC.")
                        if len(cc_data) >= offset_for_tlv_tag + 4:
                            ndef_file_id_bytes_from_cc = cc_data[offset_for_tlv_tag+2 : offset_for_tlv_tag+4]
                            ndef_file_id = int.from_bytes(ndef_file_id_bytes_from_cc, 'big')
                            if self.debug:
                                print(f"    [*] NDEF File ID from CC: {ndef_file_id:04X}")
                        elif self.debug:
                            print("[-] CC data too short for NDEF File ID within TLV.")
                    elif self.debug:
                        print("[-] NDEF File Control TLV (0x04) not found at typical offset in CC.")
            except IndexError:
                if self.debug:
                    print("[-] Error parsing CC data (IndexError). CC structure might be non-standard.")

            if ndef_file_id is None:
                if self.debug:
                    print("[!] NDEF File ID not found/parsed in CC. Assuming common NDEF File ID E104 as fallback.")
                ndef_file_id = 0xE104 

            # --- 5. Select NDEF File ---
            apdu_select_ndef_file = bytes.fromhex("00A4000C02") + ndef_file_id.to_bytes(2, 'big')
            if self.debug:
                print(f"[*] Sending SELECT NDEF FILE ({ndef_file_id:04X}): {binascii.hexlify(apdu_select_ndef_file).decode()}")
            response = tag.transceive(apdu_select_ndef_file)
            if self.debug:
                print(f"[*] Response: {binascii.hexlify(response).decode()}")
            if not response or not response.endswith(b'\x90\x00'):
                if self.debug:
                    print("[-] Failed to select NDEF File.")
                return False 

            # --- 6. Read NDEF Message ---
            apdu_read_nlen = bytes.fromhex("00B0000002")
            if self.debug:
                print(f"[*] Sending READ NLEN (NDEF message length): {binascii.hexlify(apdu_read_nlen).decode()}")
            nlen_response = tag.transceive(apdu_read_nlen)

            if not nlen_response or not nlen_response.endswith(b'\x90\x00') or len(nlen_response) < 4: 
                if self.debug:
                    print("[-] Failed to read NLEN or response too short.")
                return False 
            
            nlen_bytes = nlen_response[0:2]
            ndef_message_length = int.from_bytes(nlen_bytes, 'big')
            if self.debug:
                print(f"[*] NDEF Message Length (NLEN): {ndef_message_length} bytes")

            if ndef_message_length == 0:
                if self.debug:
                    print("[-] NDEF message length is 0. No NDEF data in file.")
                return False 
            
            all_ndef_data = bytearray()
            bytes_to_read = ndef_message_length
            current_offset = 2 
            # Use the instance variable for max_read_chunk
            # max_read_chunk_local = self.max_read_chunk # Already defined as self.max_read_chunk

            while bytes_to_read > 0:
                if not tag.is_present:
                    if self.debug:
                        print("[-] Tag removed during NDEF data read.")
                    return False

                read_len = min(bytes_to_read, self.max_read_chunk)
                p1 = (current_offset >> 8) & 0xFF
                p2 = current_offset & 0xFF
                
                apdu_read_ndef_data = bytes([0x00, 0xB0, p1, p2, read_len])
                # No debug print for each chunk read command to avoid excessive logging unless needed
                ndef_data_response = tag.transceive(apdu_read_ndef_data)

                if not ndef_data_response or not ndef_data_response.endswith(b'\x90\x00'):
                    print(f"[-] Failed to read NDEF data chunk. Response: {binascii.hexlify(ndef_data_response).decode()}") # Error always printed
                    return False 
                
                chunk = ndef_data_response[:-2]
                all_ndef_data.extend(chunk)
                bytes_to_read -= len(chunk)
                current_offset += len(chunk)

                if len(chunk) < read_len and bytes_to_read > 0:
                    if self.debug:
                        print(f"[!] Warning: Read less data ({len(chunk)}) than requested ({read_len}), "
                              f"but {bytes_to_read} bytes still expected. Ending read.")
                    break
            
            if self.debug:
                print(f"[*] Total Raw NDEF Message Read ({len(all_ndef_data)} bytes): {binascii.hexlify(all_ndef_data).decode()[:100]}...")

            # --- 7. Parse the Raw NDEF Message ---
            if all_ndef_data:
                ndef_message_parsed = False
                try:
                    for record in ndef.message_decoder(all_ndef_data):
                        if isinstance(record, ndef.uri.UriRecord):
                            if self.debug:
                                print(f"        [*] Found NDEF URI Record: {record.uri}")
                            self._process_uri(record.uri) 
                            ndef_message_parsed = True 
                    if not ndef_message_parsed and self.debug:
                        print("    [-] No relevant NDEF URI records found for processing.")
                except ndef.DecodeError as e:
                    print(f"[-] NDEF Decode Error: {e}. Data might not be valid NDEF.") # Error always printed
                except Exception as e:
                    print(f"[-] Error parsing NDEF message with ndeflib: {e}") # Error always printed
            elif self.debug:
                print("[-] No NDEF data was successfully read to parse.")
            
            return False

        except nfc.tag.TagCommandError as e:
            print(f"[-] Tag Command Error during APDU exchange: {e}") # Always print
        except nfc.tag.NotSupportedError as e:
            print(f"[-] Tag operation not supported: {e}") # Always print
        except Exception as e:
            print(f"[-] An unexpected error occurred in _on_connect_apdu: {e}") # Always print
            if self.debug:
                traceback.print_exc() 
        
        return False 

    def _nfc_listen_loop(self):
        """The core loop for the NFC listening thread."""
        print(f"[*] NFC Listener thread started. Waiting for NFC card on '{self.reader_path}'...")
        
        while self._running: 
            try:
                # Re-initialize clf inside the loop for resilience
                with nfc.ContactlessFrontend(self.reader_path) as clf_local:
                    self.clf = clf_local 
                    if not clf_local: # Should not happen if context manager succeeds
                        print(f"[-] Error: Could not initialize NFC reader at '{self.reader_path}'. Retrying in 5s...")
                        for _ in range(50): 
                            if not self._running: break
                            time.sleep(0.1)
                        continue 

                    print(f"[*] Opened NFC reader: {clf_local}. Polling for tags...")
                    
                    rdwr_options = {
                        'on-connect': self._on_connect_apdu,
                        'targets': ('106A',), 
                        'interval': 0.5, 
                        'beep-on-connect': True, # Good feedback, keep it
                    }
                    if self._running: # Check running flag before blocking call
                        # This connect call is blocking until a tag is (briefly) processed or an error.
                        # _on_connect_apdu returns False to allow polling to continue.
                        clf_local.connect(rdwr=rdwr_options) 
                        if self._running and self.debug: # Only print if still running and debug is on
                            print("[_] clf.connect() returned (tag processed or timeout/error). Resuming poll loop.")
                        if self._running: # Brief pause before next polling cycle by connect()
                           time.sleep(0.1) 
            
            except nfc.NFCError as e: 
                if not self._running: 
                    break 
                print(f"[-] NFCError in listen loop for '{self.reader_path}': {e}") # Important error, always show
                err_str = str(e).lower()
                if "no nfc device found" in err_str or \
                   "aucun périphérique nfc" in err_str or \
                   "could not claim interface" in err_str or \
                   "libnfc_edevnotfound" in err_str or \
                   "libnfc_erio" in err_str : 
                    print("[-] NFC device likely disconnected or critical error. Stopping listener.")
                    self._running = False # Stop the loop
                    break 
                else:
                    if self.debug:
                        print("[-] Retrying NFC setup after error in 5 seconds...")
                    for _ in range(50): 
                        if not self._running: break
                        time.sleep(0.1)
            except Exception as e: 
                if not self._running:
                    break
                print(f"[-] Unexpected critical error in listen loop: {e}") # Always show critical errors
                if self.debug:
                    traceback.print_exc()
                if self.debug:
                    print("[-] Retrying NFC setup after critical error in 5 seconds...")
                for _ in range(50): 
                    if not self._running: break
                    time.sleep(0.1)
            finally:
                self.clf = None # Clear self.clf if it was set
                if self._running and self.debug:
                    print("[_] NFC reader session ended or failed. Will attempt to restart if still running.")
        
        self._running = False 
        print(f"[*] NFC Listener thread for '{self.reader_path}' has stopped.")

    def start_listening(self):
        """Starts the NFC listening thread."""
        if self.nfc_thread is not None and self.nfc_thread.is_alive():
            print("[!] NFC Listener thread is already running.")
            return

        self._running = True
        self.nfc_thread = threading.Thread(target=self._nfc_listen_loop, daemon=True)
        self.nfc_thread.start()
        print(f"[*] NFC listening thread initiated for reader: {self.reader_path}")

    def stop_listening(self):
        """Stops the NFC listening thread gracefully."""
        print(f"[*] Attempting to stop NFC listener thread for {self.reader_path}...")
        self._running = False 

        if self.clf: 
            if self.debug:
                print("[*] Actively closing ContactlessFrontend to interrupt any blocking NFC calls...")
            try:
                self.clf.close() 
            except Exception as e:
                if self.debug:
                    print(f"[!] Exception while trying to close NFC reader: {e}")
        
        if self.nfc_thread and self.nfc_thread.is_alive():
            if self.debug:
                print("[*] Waiting for NFC listener thread to join...")
            self.nfc_thread.join(timeout=5.0) 
            if self.nfc_thread.is_alive():
                print("[!] NFC thread did not stop in the allocated time.")
            elif self.debug:
                print("[*] NFC listening thread successfully joined and stopped.")
        elif self.debug:
            print("[*] NFC listener thread was not running or already stopped.")
        
        self.nfc_thread = None
        self.clf = None

# --- Example Usage ---
if __name__ == "__main__":
    def my_custom_eth_processor(eth_address, public_key, uri):
        """
        This is the callback function that gets executed when an Ethereum address is found.
        """
        print("\n" + "="*40)
        print("🎉 Ethereum Address Processed! 🎉")
        print(f"   ETH Address: {eth_address}")
        if public_key:
            print(f"   Public Key : {public_key[:15]}...{public_key[-15:]}")
        if uri:
            print(f"   Source URI : {uri[:50]}{'...' if len(uri)>50 else ''}")
        print("="*40 + "\n")

    # Initialize the reader with your callback and new options
    nfc_eth_reader = NFCReaderETH(
        process_eth_address_callback=my_custom_eth_processor,
        reader_path='usb',      # Or your specific reader path
        max_read_chunk=100,     # Example: Custom max read chunk
        debug=True              # Enable debug printing
    )
    
    # To run without debug messages:
    # nfc_eth_reader = NFCReaderETH(
    #     process_eth_address_callback=my_custom_eth_processor,
    #     debug=False
    # )
    
    try:
        nfc_eth_reader.start_listening()
        print("NFC Reader is now listening for tags.")
        print("Place an NFC tag (Type4A with NDEF URI containing 'pk1') near the reader.")
        print("Press Ctrl+C to stop the program.")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Shutting down...")
    except Exception as e:
        print(f"An unexpected error occurred in the main program: {e}")
        if nfc_eth_reader.debug: # Show traceback if debug is on
            traceback.print_exc()
    finally:
        if nfc_eth_reader:
            nfc_eth_reader.stop_listening()
        print("Program terminated.")