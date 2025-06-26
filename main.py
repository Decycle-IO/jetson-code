import time
import traceback

from NFCReaderETH import NFCReaderETH

def main():
    def my_custom_eth_processor(eth_address, public_key, uri):
        """
        This is the callback function that gets executed when an Ethereum address is found.
        """
        print(eth_address)
        # You can add more logic here, e.g., send to a server, update UI, etc.

    # Initialize the reader with your callback
    # You can specify the reader path if not 'usb', e.g., 'tty:AMA0' for Raspberry Pi GPIO
    # or a specific USB device like 'usb:072f:2200'
    nfc_eth_reader = NFCReaderETH(
        process_eth_address_callback=my_custom_eth_processor
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

if __name__ == "__main__":
    main()