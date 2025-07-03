import time
import traceback
import serial
import cv2
import requests

from nfc_reader_eth import NFCReaderETH

def send_arduino_command(cmd,ser):
    try:
        ser.write(f"{cmd}\n".encode())
        print(f"Sent command {cmd} to Arduino.")
    except Exception as e:
        print(f"Failed to send serial command: {e}")

def capture_and_detect(api_url="http://localhost:8000/processImage", camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to capture frame.")
        return []
    _, buffer = cv2.imencode('.jpg', frame)
    files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
    try:
        response = requests.post(api_url, files=files, timeout=5)
        result = response.json()
        detected_classes = result.get("detected_classes", [])
        print("Detected:", detected_classes)
        return detected_classes
    except Exception as e:
        print(f"Error sending frame to API: {e}")
        return []

def my_custom_eth_processor(eth_address, public_key, uri):
    print(eth_address)
    # 1. Open the magnetic lock (door)
    #send_arduino_command(3)
    # 2. Capture and detect object
    detected = capture_and_detect()
    # 3. Send bucket command based on detection
    if "plastic" in detected:
        send_arduino_command(1)
    elif "metal" in detected:
        send_arduino_command(2)
    else:
        print("No known object detected.")

def main(ser):
    nfc_eth_reader = NFCReaderETH(
        process_eth_address_callback=my_custom_eth_processor
    )
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
        traceback.print_exc()
    finally:
        if nfc_eth_reader:
            nfc_eth_reader.stop_listening()
        print("Program terminated.")

if __name__ == "__main__":
    try:    
        with serial.Serial('/dev/ttyACM0', 115200, timeout=2) as ser:
            main(ser)
    except Exception as e:
        print(f"An unexpected error occurred in the main program: {e}")
        traceback.print_exc()