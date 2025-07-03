# main_app.py

import time
import traceback
import cv2
import requests
from pathlib import Path
import threading
import base64

# Assuming this custom module exists and is in the same directory or Python path
from nfc_reader_eth import NFCReaderETH

# --- Configuration ---
SERIAL_API_URL = "http://localhost:8001/send_command"
SAMPLE_DATA_DIR = Path("./sample_data")
CUSTOM_MODEL_PATH = SAMPLE_DATA_DIR / "custom_model.pt"
CUSTOM_CLASSES_PATH = SAMPLE_DATA_DIR / "classes.json"
VISION_API_URL = "http://localhost:8000/processImage"

webcam_thread = None

class WebcamThread(threading.Thread):
    """
    A thread that continuously reads frames from a webcam.
    """
    def __init__(self, src=0, name="WebcamThread"):
        """
        Initializes the thread.
        :param src: The source of the webcam (default is 0).
        :param name: The name of the thread.
        """
        super(WebcamThread, self).__init__(name=name)
        self.src = src
        self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open webcam at source {self.src}")

        # A flag to signal the thread to stop
        self._stop_event = threading.Event()
        
        # A lock to ensure thread-safe access to the frame
        self.frame_lock = threading.Lock()
        
        # The latest frame read from the webcam
        self.latest_frame = None

    def run(self):
        """
        The main loop of the thread. Reads frames from the webcam
        until the stop event is set.
        """
        print(f"{self.name}: Starting.")
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow("Frame", frame)
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                # If reading fails, it might be the end of a video file
                # or a camera disconnection.
                print(f"{self.name}: Failed to read frame. Stopping.")
                self._stop_event.set()
        
        # Release the camera resource when the loop is finished
        self.cap.release()
        print(f"{self.name}: Stopped and camera released.")

    def read(self):
        """
        Returns the latest frame read by the thread.
        This is a thread-safe method.
        """
        with self.frame_lock:
            return self.latest_frame

    def stop(self):
        """
        Signals the thread to stop.
        """
        print(f"{self.name}: Stop signal received.")
        self._stop_event.set()

def send_arduino_command(cmd_char: str):
    """
    Sends a command character to the serial API server via HTTP.
    """
    try:
        payload = {"command": cmd_char}
        response = requests.post(SERIAL_API_URL, json=payload, timeout=5)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()  
        print(f"Sent command '{cmd_char}' to Arduino via API. Response: {response.json()}")
        time.sleep(10)
    except requests.exceptions.RequestException as e:
        # This will catch connection errors, timeouts, and bad status codes
        print(f"Failed to send command '{cmd_char}' via API: {e}")

def capture_and_detect():
    return input()
    """
    Captures an image from the webcam, sends it to a vision API for detection,
    displays the result, and returns the detected material type ('p' or 'm').
    Falls back to manual input on failure.
    """
    global webcam_thread

    # 1. Get a frame from the webcam thread
    if webcam_thread is None or not webcam_thread.is_alive():
        print("Error: Webcam thread is not running.")
        return input("Fallback: Enter material manually (p/m): ")

    image_np = webcam_thread.read()
    if image_np is None:
        print("Error: Failed to read frame from webcam.")
        return input("Fallback: Enter material manually (p/m): ")

    # 2. Encode the image to JPEG format in memory
    is_success, buffer = cv2.imencode(".jpg", image_np)
    if not is_success:
        print("Error: Failed to encode image to JPEG format.")
        return input("Fallback: Enter material manually (p/m): ")
    image_bytes = buffer.tobytes()

    # 3. Prepare the multipart-form data for the POST request
    files = {"file": ("webcam_capture.jpg", image_bytes, "image/jpeg")}

    detected_classes = []
    # 4. Make the API call to the vision server
    try:
        print("Sending image to vision API for processing...")
        response = requests.post(VISION_API_URL, files=files, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        detected_classes = response_data.get("detected_classes", [])
        print(f"API Response: Detected classes -> {detected_classes}")

        # (Optional but recommended) Decode and display the annotated image from the API
        image_b64_string = response_data.get("image_b64")
        if image_b64_string:
            decoded_image_bytes = base64.b64decode(image_b64_string)
            image_np_array = np.frombuffer(decoded_image_bytes, np.uint8)
            annotated_frame = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
            if annotated_frame is not None:
                cv2.imshow("Detection Result (from API)", annotated_frame)
                cv2.waitKey(1)  # Allow the window to update

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the vision API: {e}")
        return input("API Error. Enter material manually (p/m): ")
    except (KeyError, ValueError) as e:
        print(f"Error parsing API response: {e}")
        return input("API Response Error. Enter material manually (p/m): ")

    # 5. Return the material code based on detection results
    if "plastic" in detected_classes:
        print("Material identified as: Plastic")
        return input("Enter p or press Enter to skip: ")
    elif "metal" in detected_classes:
        print("Material identified as: Metal")
        return input("Enter m or press Enter to skip: ")
    else:
        print("No known material (plastic/metal) detected by the API.")
        return input("Enter material manually (p/m) or press Enter to skip: ")


def my_custom_eth_processor(eth_address, public_key, uri):
    print(eth_address)
    with open("addresses.txt", "a") as f:
        f.write(f"{eth_address}\n")
    
    if eth_address == "0x6314F3D251f0f944540f189Bc4d9A71f1B6eF28e":
        send_arduino_command('3')
        return
    # 1. Open the magnetic lock (door)
    send_arduino_command('0')
    # 2. Capture and detect object
    material = capture_and_detect() # input()
    # 3. Send bucket command based on detection
    if material == "p": #"plastic" in detected:
        send_arduino_command('1')
    elif material == "m": # "metal" in detected:
        send_arduino_command('2')
    else:
        print("No known object detected.")

def main():
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
    # webcam_thread = WebcamThread(src=0)
    # webcam_thread.start()

    try:    
        # No longer needs to manage the serial port connection
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()