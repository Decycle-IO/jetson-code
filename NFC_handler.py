import nfc
import nfc.tag.tt4 # For Type4ATag specific definitions if needed
import time
import binascii # To print byte arrays nicely

# --- If you want to use ndeflib to parse the raw NDEF bytes later ---
import ndef # From 'pip install ndeflib'

def on_connect_apdu(tag):
    print(f"[-] Card connected: {tag}")
    if not isinstance(tag, nfc.tag.tt4.Type4ATag):
        print("[-] Tag is not a Type4ATag. APDU interaction might differ.")
        return False

    try:
        # --- 1. Select NDEF Tag Application ---
        # AID for NDEF Application (Version 1.0)
        # You might also try D2760000850100 for older tags
        aid_ndef_v1 = bytes.fromhex("D2760000850101")
        apdu_select_ndef_app = bytes.fromhex("00A40400") + bytes([len(aid_ndef_v1)]) + aid_ndef_v1 + bytes.fromhex("00")
        
        print(f"[*] Sending SELECT NDEF APP: {binascii.hexlify(apdu_select_ndef_app).decode()}")
        response = tag.transceive(apdu_select_ndef_app)
        print(f"[*] Response: {binascii.hexlify(response).decode()}")
        # Expected response: 9000 (OK)

        if not response or not response.endswith(b'\x90\x00'):
            print("[-] Failed to select NDEF Application or unexpected response.")
            # Try the older AID as a fallback
            aid_ndef_v0 = bytes.fromhex("D2760000850100")
            apdu_select_ndef_app_v0 = bytes.fromhex("00A40400") + bytes([len(aid_ndef_v0)]) + aid_ndef_v0 + bytes.fromhex("00")
            print(f"[*] Sending SELECT NDEF APP (v0): {binascii.hexlify(apdu_select_ndef_app_v0).decode()}")
            response = tag.transceive(apdu_select_ndef_app_v0)
            print(f"[*] Response: {binascii.hexlify(response).decode()}")
            if not response or not response.endswith(b'\x90\x00'):
                print("[-] Failed to select NDEF Application (v0) as well.")
                return False

        # --- 2. Select Capability Container (CC) File ---
        # File ID for CC file is typically E103
        cc_file_id = bytes.fromhex("E103")
        apdu_select_cc = bytes.fromhex("00A4000C") + bytes([len(cc_file_id)]) + cc_file_id
        # No Le here, some tags prefer it this way for SELECT by ID.
        # Or you can add Le=00 : bytes.fromhex("00A4000C") + bytes([len(cc_file_id)]) + cc_file_id + bytes([0x00])
        
        print(f"[*] Sending SELECT CC FILE: {binascii.hexlify(apdu_select_cc).decode()}")
        response = tag.transceive(apdu_select_cc)
        print(f"[*] Response: {binascii.hexlify(response).decode()}")
        if not response or not response.endswith(b'\x90\x00'):
            print("[-] Failed to select CC File.")
            return False

        # --- 3. Read CC File ---
        # CC file is usually 15 bytes long (0Fh)
        apdu_read_cc = bytes.fromhex("00B000000F") # Read 15 bytes from offset 0
        print(f"[*] Sending READ CC FILE: {binascii.hexlify(apdu_read_cc).decode()}")
        cc_data_response = tag.transceive(apdu_read_cc)
        print(f"[*] CC Raw Response: {binascii.hexlify(cc_data_response).decode()}")

        if not cc_data_response or not cc_data_response.endswith(b'\x90\x00'):
            print("[-] Failed to read CC File.")
            return False
        
        cc_data = cc_data_response[:-2] # Strip SW1SW2 (9000)
        print(f"[*] CC Data: {binascii.hexlify(cc_data).decode()}")

        # --- 4. Parse CC File (Simplified Example) ---
        # A proper CC parse is more involved, checking CCLEN, version, TLVs
        # This is a very basic parse assuming Type 04h NDEF File Control TLV
        # CCLEN (2 bytes), Mapping Version (1 byte), MLe (2 bytes), MLc (2 bytes)
        # Then NDEF File Control TLV (04h)
        # Example: 000F 20 MLeR MLeS NDEFTLV_T(04) NDEFTLV_L(06) NDEF_FileID(2) MaxNDEFSize(2) ReadAccess(1) WriteAccess(1)
        
        ndef_file_id = None
        max_ndef_size = 0
        
        # Look for NDEF File Control TLV (Tag 0x04)
        # This is a very simplified search, real parsing should iterate TLVs
        try:
            # Typical offset for NDEF File Control TLV tag after CCLEN, MappingVer, MLe, MLc (2+1+2+2 = 7 bytes)
            offset = 7 
            if cc_data[offset] == 0x04: # NDEF File Control TLV Tag
                print("[*] Found NDEF File Control TLV (0x04)")
                ndef_tlv_len = cc_data[offset+1]
                ndef_file_id_bytes = cc_data[offset+2 : offset+4]
                ndef_file_id = int.from_bytes(ndef_file_id_bytes, 'big')
                print(f"    [*] NDEF File ID from CC: {ndef_file_id:04X}")
                
                max_ndef_size_bytes = cc_data[offset+4 : offset+6]
                max_ndef_size = int.from_bytes(max_ndef_size_bytes, 'big')
                print(f"    [*] Max NDEF Size from CC: {max_ndef_size}")

                read_access = cc_data[offset+6]
                print(f"    [*] Read Access: {read_access:02X}")
                # write_access = cc_data[offset+7] # if ndef_tlv_len is long enough
                # print(f"    [*] Write Access: {write_access:02X}")
            else:
                print("[-] NDEF File Control TLV (0x04) not found at expected offset. CC structure might differ.")
                # You might need to implement a full TLV parser for the CC file.
                # For now, let's try a common NDEF File ID if not found.
                print("[!] Assuming common NDEF File ID E104 as fallback.")
                ndef_file_id = 0xE104 # Common default if parsing fails.
                
        except IndexError:
            print("[-] Error parsing CC data (IndexError). CC data might be shorter than expected.")
            print("[!] Assuming common NDEF File ID E104 as fallback.")
            ndef_file_id = 0xE104 # Common default if parsing fails.

        if ndef_file_id is None:
            print("[-] Could not determine NDEF File ID from CC.")
            return False

        # --- 5. Select NDEF File ---
        apdu_select_ndef_file = bytes.fromhex("00A4000C02") + ndef_file_id.to_bytes(2, 'big')
        print(f"[*] Sending SELECT NDEF FILE ({ndef_file_id:04X}): {binascii.hexlify(apdu_select_ndef_file).decode()}")
        response = tag.transceive(apdu_select_ndef_file)
        print(f"[*] Response: {binascii.hexlify(response).decode()}")
        if not response or not response.endswith(b'\x90\x00'):
            print("[-] Failed to select NDEF File.")
            return False

        # --- 6. Read NDEF Message ---
        # First, read the NLEN (2 bytes, length of the NDEF message)
        apdu_read_nlen = bytes.fromhex("00B0000002") # Read 2 bytes from offset 0 of NDEF file
        print(f"[*] Sending READ NLEN: {binascii.hexlify(apdu_read_nlen).decode()}")
        nlen_response = tag.transceive(apdu_read_nlen)
        print(f"[*] NLEN Raw Response: {binascii.hexlify(nlen_response).decode()}")

        if not nlen_response or not nlen_response.endswith(b'\x90\x00') or len(nlen_response) < 4: # 2 data bytes + 9000
            print("[-] Failed to read NLEN or response too short.")
            return False
        
        nlen_bytes = nlen_response[0:2]
        ndef_message_length = int.from_bytes(nlen_bytes, 'big')
        print(f"[*] NDEF Message Length (NLEN): {ndef_message_length} bytes")

        if ndef_message_length == 0:
            print("[-] NDEF message length is 0. No NDEF data in file.")
            return False
        if max_ndef_size > 0 and ndef_message_length > max_ndef_size: # max_ndef_size from CC
             print(f"[!] Warning: NLEN ({ndef_message_length}) > Max NDEF Size from CC ({max_ndef_size})")


        # Read the actual NDEF message
        # Need to handle reading in chunks if ndef_message_length > MLe (from CC, max read size)
        # For simplicity, this example assumes it can be read in one go or a few chunks.
        # A robust implementation would check MLe from CC. Let's assume MLe is at least 250 for now.
        # Data starts at offset 2 (after NLEN)
        
        all_ndef_data = bytearray()
        bytes_to_read = ndef_message_length
        current_offset = 2 # Start reading after NLEN field
        max_read_chunk = 120 # Adjust based on MLe if known, or conservative value

        while bytes_to_read > 0:
            read_len = min(bytes_to_read, max_read_chunk)
            # P1P2 = offset (big endian)
            p1 = (current_offset >> 8) & 0xFF
            p2 = current_offset & 0xFF
            
            apdu_read_ndef_data = bytes([0x00, 0xB0, p1, p2, read_len])
            print(f"[*] Sending READ NDEF DATA (offset {current_offset}, len {read_len}): {binascii.hexlify(apdu_read_ndef_data).decode()}")
            ndef_data_response = tag.transceive(apdu_read_ndef_data)
            # print(f"[*] NDEF Data Raw Response: {binascii.hexlify(ndef_data_response).decode()}")

            if not ndef_data_response or not ndef_data_response.endswith(b'\x90\x00'):
                print(f"[-] Failed to read NDEF data chunk. Response: {binascii.hexlify(ndef_data_response).decode()}")
                return False
            
            chunk = ndef_data_response[:-2] # Strip SW1SW2
            all_ndef_data.extend(chunk)
            
            bytes_to_read -= len(chunk)
            current_offset += len(chunk)

            if len(chunk) < read_len: # Read less than requested, probably end of file
                if bytes_to_read > 0:
                    print(f"[!] Warning: Read less data than expected for chunk but still {bytes_to_read} bytes_to_read.")
                break


        print(f"[*] Raw NDEF Message ({len(all_ndef_data)} bytes): {binascii.hexlify(all_ndef_data).decode()}")

        # --- 7. Parse the Raw NDEF Message (using ndeflib) ---
        if all_ndef_data:
            try:
                # The ndef.message.decode() function takes a bytestring
                for record in ndef.message_decoder(all_ndef_data):
                    print(f"    [+] NDEF Record Type: {record.type}, Name: {record.name}")
                    if isinstance(record, ndef.uri.UriRecord):
                        print(f"        [*] NDEF URI: {record.uri}")
                        process_uri(record.uri)
                        # Found what we wanted
                    elif isinstance(record, ndef.text.TextRecord):
                        print(f"        [*] NDEF Text: {record.text}")
                    # Add other record types as needed (ndef.smartposter.SmartposterRecord etc.)
            except ndef.DecodeError as e:
                print(f"[-] NDEF Decode Error: {e}")
                print("    The data read might not be a valid NDEF message or ndeflib cannot parse it.")
            except Exception as e:
                print(f"[-] Error parsing NDEF with ndeflib: {e}")
        else:
            print("[-] No NDEF data was read to parse.")

        return True # Indicate success or completion for this tag

    except nfc.tag.TagCommandError as e:
        print(f"[-] Tag Command Error: {e}")
    except Exception as e:
        print(f"[-] An unexpected error occurred: {e}")
    return False


def main_apdu_reader():
    reader_path = 'usb'
    try:
        with nfc.ContactlessFrontend(reader_path) as clf:
            if not clf:
                print(f"[-] Error: Could not connect to NFC reader at '{reader_path}'.")
                return

            print(f"[*] Opened NFC reader: {clf}")
            print("[*] Waiting for an NFC card (Type4A for APDU demo)...")
            print("[*] Press Ctrl+C to exit.")

            while True:
                # The target_type ensures we only try to connect to Type 4A tags
                # for this specific APDU demo. Remove if you want to try others.
                # target = clf.sense(nfc.clf.RemoteTarget('106A')) # Type A
                # if target:
                #    tag = nfc.tag.activate(clf, target)
                #    if tag:
                #        on_connect_apdu(tag)
                # else:
                #    time.sleep(0.1)

                # Simpler connect loop that calls our APDU handler
                # Set a short timeout for connect to make Ctrl+C more responsive
                options = {
                    'on-connect': on_connect_apdu,
                    'targets': ('106A',), # Only look for Type A targets
                    'interval': 0.5, # Polling interval in seconds
                    'beep-on-connect': True
                }
                # clf.connect will block until a tag is found or timeout occurs (if timeout is set in rdwr)
                # Here, interval in options makes it non-blocking in a sense that it retries.
                # The on-connect will be called. If it returns False, connect() will continue polling.
                # If it returns True, connect() in this usage might release and poll again.
                # A custom loop with sense() might be more flexible for some scenarios.
                
                # Using a simple blocking connect and letting on_connect_apdu handle everything
                # after activation.
                try:
                    # Note: on-connect returning True might cause connect to release the tag.
                    # If you want to keep it connected for more interactions in the main loop,
                    # on-connect should return False or the tag object itself.
                    # For this example, we do all work in on_connect_apdu.
                    activated_tag = clf.connect(rdwr={'on-connect': on_connect_apdu})
                    if activated_tag:
                         print("[-] Tag processed by on_connect_apdu. Waiting for next tag...")
                    # No explicit sleep needed here if connect blocks.
                    # If connect has a timeout, it will unblock and loop.
                    time.sleep(0.5) # Brief pause before next poll attempt
                
                except nfc.tag.TIMEOUT_ERROR: # This might not be hit if connect() is fully blocking
                    pass 
                except nfc.tag.TAG_NOT_FOUND_ERROR:
                    pass


    except nfc.NFCError as e:
        print(f"[-] NFCError: {e}")
    except IOError as e:
        print(f"[-] IOError: {e}")
    except KeyboardInterrupt:
        print("\n[*] Exiting...")
    finally:
        print("[*] Script finished.")

def process_uri(uri):
    print(uri)

if __name__ == "__main__":
    main_apdu_reader()