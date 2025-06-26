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
    def __init__(self, process_eth_address_callback, reader_path='usb'):
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
        """
        if not callable(process_eth_address_callback):
            raise ValueError("process_eth_address_callback must be a callable function.")
        
        self.process_eth_address_callback = process_eth_address_callback
        self.reader_path = reader_path
        
        self.clf = None # Contactless Frontend, managed by the listening thread
        self._running = False # Flag to control the listening loop
        self.nfc_thread = None

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
        print(f"    [*] Processing URI: {unquoted_uri}")
        eth_addr = None
        pk1_val = None
        try:
            parsed_url = urllib.parse.urlparse(unquoted_uri)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            pk1_val = query_params.get('pk1', [None])[0]

            if pk1_val:
                print(f"        [*] Extracted pk1: {pk1_val[:10]}...{pk1_val[-10:] if len(pk1_val) > 20 else pk1_val}")
                eth_addr = self._public_key_to_eth_address(pk1_val)
                print(f"        [*] Derived Ethereum Address: {eth_addr}")
                
                # Call the callback with the processed information
                if self.process_eth_address_callback:
                    self.process_eth_address_callback(eth_addr, pk1_val, unquoted_uri)
            else:
                print("        [!] 'pk1' not found in URI query parameters.")

        except ValueError as e: # Specifically for errors from _public_key_to_eth_address
            print(f"        [!] Error converting public key: {e}")
        except Exception as e:
            print(f"        [!] Error parsing URI or processing pk1: {e}")
            print(f"            URI was: {unquoted_uri}")
            # traceback.print_exc() # Uncomment for more detailed debugging if needed

    def _on_connect_apdu(self, tag):
        """
        Handles NFC tag connection and APDU interactions to read NDEF data.
        This method is called by nfcpy when a tag is connected.
        It should return False to allow clf.connect to continue polling.
        """
        print(f"[-] Card connected: {tag}")
        if not tag.is_present: # Check if tag disappeared
            print("[-] Tag disappeared immediately after connect.")
            return False # Continue polling

        if not isinstance(tag, nfc.tag.tt4.Type4ATag):
            print("[-] Tag is not a Type4ATag. APDU interaction might differ or fail.")
            return False # Continue polling

        try:
            # --- 1. Select NDEF Tag Application ---
            aid_ndef_v1 = bytes.fromhex("D2760000850101") # NDEF Application AID v1.0
            aid_ndef_v0 = bytes.fromhex("D2760000850100") # NDEF Application AID v0.x (older)
            
            apdu_select_ndef_app = bytes.fromhex("00A40400") + bytes([len(aid_ndef_v1)]) + aid_ndef_v1 + bytes.fromhex("00") # Le=00
            
            print(f"[*] Sending SELECT NDEF APP (v1.0): {binascii.hexlify(apdu_select_ndef_app).decode()}")
            response = tag.transceive(apdu_select_ndef_app)
            print(f"[*] Response: {binascii.hexlify(response).decode()}")

            if not response or not response.endswith(b'\x90\x00'):
                print("[-] Failed to select NDEF Application (v1.0). Trying older AID (v0.x)...")
                apdu_select_ndef_app_v0 = bytes.fromhex("00A40400") + bytes([len(aid_ndef_v0)]) + aid_ndef_v0 + bytes.fromhex("00")
                print(f"[*] Sending SELECT NDEF APP (v0.x): {binascii.hexlify(apdu_select_ndef_app_v0).decode()}")
                response = tag.transceive(apdu_select_ndef_app_v0)
                print(f"[*] Response: {binascii.hexlify(response).decode()}")
                if not response or not response.endswith(b'\x90\x00'):
                    print("[-] Failed to select NDEF Application (v0.x) as well.")
                    return False # Continue polling

            # --- 2. Select Capability Container (CC) File ---
            cc_file_id_bytes = bytes.fromhex("E103") # Standard CC File ID
            apdu_select_cc = bytes.fromhex("00A4000C") + bytes([len(cc_file_id_bytes)]) + cc_file_id_bytes
            
            print(f"[*] Sending SELECT CC FILE: {binascii.hexlify(apdu_select_cc).decode()}")
            response = tag.transceive(apdu_select_cc)
            print(f"[*] Response: {binascii.hexlify(response).decode()}")
            if not response or not response.endswith(b'\x90\x00'):
                print("[-] Failed to select CC File.")
                return False # Continue polling

            # --- 3. Read CC File ---
            apdu_read_cc = bytes.fromhex("00B000000F") # Read first 15 bytes of CC
            print(f"[*] Sending READ CC FILE: {binascii.hexlify(apdu_read_cc).decode()}")
            cc_data_response = tag.transceive(apdu_read_cc)

            if not cc_data_response or not cc_data_response.endswith(b'\x90\x00'):
                print("[-] Failed to read CC File data.")
                return False 
            
            cc_data = cc_data_response[:-2] # Strip SW1SW2 (status bytes)
            print(f"[*] CC Data (first 15 bytes): {binascii.hexlify(cc_data).decode()}")

            # --- 4. Parse CC File (Simplified for NDEF File ID) ---
            ndef_file_id = None
            try:
                if len(cc_data) >= 8: 
                    offset_for_tlv_tag = 7 
                    if cc_data[offset_for_tlv_tag] == 0x04: # NDEF File Control TLV Tag
                        print("[*] Found NDEF File Control TLV (0x04) in CC.")
                        if len(cc_data) >= offset_for_tlv_tag + 4: # Tag(1)+Len(1)+FileID(2)
                            ndef_file_id_bytes_from_cc = cc_data[offset_for_tlv_tag+2 : offset_for_tlv_tag+4]
                            ndef_file_id = int.from_bytes(ndef_file_id_bytes_from_cc, 'big')
                            print(f"    [*] NDEF File ID from CC: {ndef_file_id:04X}")
                        else:
                            print("[-] CC data too short for NDEF File ID within TLV.")
                    else:
                        print("[-] NDEF File Control TLV (0x04) not found at typical offset in CC.")
            except IndexError:
                print("[-] Error parsing CC data (IndexError). CC structure might be non-standard.")

            if ndef_file_id is None:
                print("[!] NDEF File ID not found/parsed in CC. Assuming common NDEF File ID E104 as fallback.")
                ndef_file_id = 0xE104 

            # --- 5. Select NDEF File ---
            apdu_select_ndef_file = bytes.fromhex("00A4000C02") + ndef_file_id.to_bytes(2, 'big')
            print(f"[*] Sending SELECT NDEF FILE ({ndef_file_id:04X}): {binascii.hexlify(apdu_select_ndef_file).decode()}")
            response = tag.transceive(apdu_select_ndef_file)
            print(f"[*] Response: {binascii.hexlify(response).decode()}")
            if not response or not response.endswith(b'\x90\x00'):
                print("[-] Failed to select NDEF File.")
                return False 

            # --- 6. Read NDEF Message ---
            apdu_read_nlen = bytes.fromhex("00B0000002") # Read NLEN (2 bytes) from offset 0
            print(f"[*] Sending READ NLEN (NDEF message length): {binascii.hexlify(apdu_read_nlen).decode()}")
            nlen_response = tag.transceive(apdu_read_nlen)

            if not nlen_response or not nlen_response.endswith(b'\x90\x00') or len(nlen_response) < 4: 
                print("[-] Failed to read NLEN or response too short.")
                return False 
            
            nlen_bytes = nlen_response[0:2]
            ndef_message_length = int.from_bytes(nlen_bytes, 'big')
            print(f"[*] NDEF Message Length (NLEN): {ndef_message_length} bytes")

            if ndef_message_length == 0:
                print("[-] NDEF message length is 0. No NDEF data in file.")
                return False 
            
            all_ndef_data = bytearray()
            bytes_to_read = ndef_message_length
            current_offset = 2 # Data starts after NLEN field
            max_read_chunk = 120 # Conservative max read size per APDU (MLe can be up to 255)

            while bytes_to_read > 0:
                if not tag.is_present:
                    print("[-] Tag removed during NDEF data read.")
                    return False

                read_len = min(bytes_to_read, max_read_chunk)
                p1 = (current_offset >> 8) & 0xFF
                p2 = current_offset & 0xFF
                
                apdu_read_ndef_data = bytes([0x00, 0xB0, p1, p2, read_len])
                ndef_data_response = tag.transceive(apdu_read_ndef_data)

                if not ndef_data_response or not ndef_data_response.endswith(b'\x90\x00'):
                    print(f"[-] Failed to read NDEF data chunk. Response: {binascii.hexlify(ndef_data_response).decode()}")
                    return False 
                
                chunk = ndef_data_response[:-2]
                all_ndef_data.extend(chunk)
                bytes_to_read -= len(chunk)
                current_offset += len(chunk)

                if len(chunk) < read_len and bytes_to_read > 0:
                    print(f"[!] Warning: Read less data ({len(chunk)}) than requested ({read_len}), "
                          f"but {bytes_to_read} bytes still expected. Ending read.")
                    break
            
            print(f"[*] Total Raw NDEF Message Read ({len(all_ndef_data)} bytes): {binascii.hexlify(all_ndef_data).decode()[:100]}...")

            # --- 7. Parse the Raw NDEF Message ---
            if all_ndef_data:
                ndef_message_parsed = False
                try:
                    for record in ndef.message_decoder(all_ndef_data):
                        if isinstance(record, ndef.uri.UriRecord):
                            print(f"        [*] Found NDEF URI Record: {record.uri}")
                            self._process_uri(record.uri) 
                            ndef_message_parsed = True 
                    if not ndef_message_parsed:
                        print("    [-] No relevant NDEF URI records found for processing.")
                except ndef.DecodeError as e:
                    print(f"[-] NDEF Decode Error: {e}. Data might not be valid NDEF.")
                except Exception as e:
                    print(f"[-] Error parsing NDEF message with ndeflib: {e}")
            else:
                print("[-] No NDEF data was successfully read to parse.")
            
            return False # IMPORTANT: Continue polling for other tags

        except nfc.tag.TagCommandError as e:
            print(f"[-] Tag Command Error during APDU exchange: {e}")
        except nfc.tag.NotSupportedError as e: # Catch if tag does not support an operation
            print(f"[-] Tag operation not supported: {e}")
        except Exception as e:
            print(f"[-] An unexpected error occurred in _on_connect_apdu: {e}")
            # traceback.print_exc() 
        
        return False # Ensure polling continues if any error occurs

    def _nfc_listen_loop(self):
        """The core loop for the NFC listening thread."""
        print(f"[*] NFC Listener thread started. Waiting for NFC card on '{self.reader_path}'...")
        
        while self._running: 
            try:
                with nfc.ContactlessFrontend(self.reader_path) as clf_local:
                    self.clf = clf_local 
                    if not clf_local:
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
                        'beep-on-connect': True,
                    }
                    if self._running:
                        clf_local.connect(rdwr=rdwr_options) 
                        if self._running:
                            print("[!] clf.connect() exited. Will retry if still running.")
                            time.sleep(1.0)
            except nfc.NFCError as e: 
                if not self._running: 
                    break 
                print(f"[-] NFCError in listen loop for '{self.reader_path}': {e}")
                err_str = str(e).lower()
                if "no nfc device found" in err_str or \
                   "aucun périphérique nfc" in err_str or \
                   "could not claim interface" in err_str or \
                   "libnfc_edevnotfound" in err_str or \
                   "libnfc_erio" in err_str : 
                    print("[-] NFC device likely disconnected or critical error. Stopping listener.")
                    self._running = False 
                    break 
                else:
                    print("[-] Retrying NFC setup after error in 5 seconds...")
                    for _ in range(50): 
                        if not self._running: break
                        time.sleep(0.1)
            except Exception as e: 
                if not self._running:
                    break
                print(f"[-] Unexpected critical error in listen loop: {e}")
                traceback.print_exc()
                print("[-] Retrying NFC setup after critical error in 5 seconds...")
                for _ in range(50): 
                    if not self._running: break
                    time.sleep(0.1)
            finally:
                self.clf = None 
                if self._running:
                    print("[_] NFC reader session ended. Will attempt to restart if still running.")
        
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
            print("[*] Actively closing ContactlessFrontend to interrupt any blocking NFC calls...")
            try:
                self.clf.close() 
            except Exception as e:
                print(f"[!] Exception while trying to close NFC reader: {e}")
        
        if self.nfc_thread and self.nfc_thread.is_alive():
            print("[*] Waiting for NFC listener thread to join...")
            self.nfc_thread.join(timeout=5.0) 
            if self.nfc_thread.is_alive():
                print("[!] NFC thread did not stop in the allocated time.")
            else:
                print("[*] NFC listening thread successfully joined and stopped.")
        else:
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
        # You can add more logic here, e.g., send to a server, update UI, etc.

    # Initialize the reader with your callback
    # You can specify the reader path if not 'usb', e.g., 'tty:AMA0' for Raspberry Pi GPIO
    # or a specific USB device like 'usb:072f:2200'
    nfc_eth_reader = NFCReaderETH(
        process_eth_address_callback=my_custom_eth_processor,
        # reader_path='usb:072f:2200' # Example: replace with your reader if not default 'usb'
    )
    
    try:
        nfc_eth_reader.start_listening()
        print("NFC Reader is now listening for tags.")
        print("Place an NFC tag (Type4A with NDEF URI containing 'pk1') near the reader.")
        print("Press Ctrl+C to stop the program.")
        
        # Keep the main thread alive, otherwise the program will exit as the NFC thread is a daemon.
        while True:
            time.sleep(1) # Or use input("Press Enter to stop.\n") for manual stop.
            
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Shutting down...")
    except Exception as e:
        print(f"An unexpected error occurred in the main program: {e}")
        traceback.print_exc()
    finally:
        if nfc_eth_reader:
            nfc_eth_reader.stop_listening()
        print("Program terminated.")