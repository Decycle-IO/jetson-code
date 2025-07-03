# main_app.py

import time
import traceback
import cv2
import requests
import numpy as np
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

# --- Globals ---
# We use globals to facilitate communication between the main thread (GUI)
# and the NFC callback thread (background processing).
webcam_thread = None
annotated_frame_for_display = None
annotated_frame_lock = threading.Lock()


class WebcamThread(threading.Thread):
    """
    A thread that continuously reads frames from a webcam.
    This thread's only job is to capture frames. It does NOT display them.
    """
    def __init__(self, src=0, name="WebcamThread"):
        super(WebcamThread, self).__init__(name=name)
        self.src = src
        self.cap = cv2.VideoCapture(self.src)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open webcam at source {self.src}")

        self._stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.running = False

    def run(self):
        """
        The main loop of the thread. Reads frames from the webcam
        until the stop event is set.
        """
        print(f"{self.name}: Starting.")
        self.running = True
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                print(f"{self.name}: Failed to read frame. Stopping.")
                self._stop_event.set()
        
        self.running = False
        self.cap.release()
        print(f"{self.name}: Stopped and camera released.")

    def read(self):
        """Returns the latest frame in a thread-safe manner."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        """Signals the thread to stop."""
        print(f"{self.name}: Stop signal received.")
        self._stop_event.set()

def send_arduino_command(cmd_char: str):
    """Sends a command character to the serial API server via HTTP."""
    try:
        payload = {"command": cmd_char}
        response = requests.post(SERIAL_API_URL, json=payload, timeout=5)
        response.raise_for_status()
        print(f"Sent command '{cmd_char}' to Arduino via API. Response: {response.json()}")
        # Reduced sleep time for better responsiveness
        time.sleep(5)
    except requests.exceptions.RequestException as e:
        print(f"Failed to send command '{cmd_char}' via API: {e}")

def capture_and_detect():
    """
    Captures an image, sends it to a vision API, and returns the material type.
    This function is called from a callback, so it does NOT perform GUI operations.
    """
    # Use global variables to communicate back to the main thread
    global webcam_thread, annotated_frame_for_display, annotated_frame_lock

    if webcam_thread is None or not webcam_thread.running:
        print("Error: Webcam thread is not running.")
        return input("Fallback: Enter material manually (p/m): ")

    image_np = webcam_thread.read()
    if image_np is None:
        print("Error: Failed to read frame from webcam.")
        return input("Fallback: Enter material manually (p/m): ")

    is_success, buffer = cv2.imencode(".jpg", image_np)
    if not is_success:
        print("Error: Failed to encode image to JPEG format.")
        return input("Fallback: Enter material manually (p/m): ")
    image_bytes = buffer.tobytes()

    files = {"file": ("webcam_capture.jpg", image_bytes, "image/jpeg")}
    detected_classes = []
    try:
        print("Sending image to vision API for processing...")
        response = requests.post(VISION_API_URL, files=files, timeout=10)
        response.raise_for_status()

        response_data = response.json()
        detected_classes = response_data.get("detected_classes", [])
        print(f"API Response: Detected classes -> {detected_classes}")

        image_b64_string = response_data.get("image_b64")
        if image_b64_string:
            decoded_image_bytes = base64.b64decode(image_b64_string)
            image_np_array = np.frombuffer(decoded_image_bytes, np.uint8)
            annotated_frame = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)

            # --- KEY CHANGE: DO NOT SHOW IMAGE HERE ---
            # Instead, acquire the lock and place the annotated frame in the
            # shared global variable for the main thread to pick up and display.
            # cv2.imshow("Object detection", annotated_frame) # <-- This was the error
            with annotated_frame_lock:
                annotated_frame_for_display = annotated_frame
                print("Annotated frame stored for display by the main thread.")


    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the vision API: {e}")
        return input("API Error. Enter material manually (p/m): ")
    except (KeyError, ValueError) as e:
        print(f"Error parsing API response: {e}")
        return input("API Response Error. Enter material manually (p/m): ")

    if "plastic" in detected_classes:
        print("Material identified as: Plastic")
        return 'p'
    elif "metal" in detected_classes:
        print("Material identified as: Metal")
        return 'm'
    else:
        print("No known material (plastic/metal) detected by the API.")
        return input("Enter material manually (p/m) or press Enter to skip: ")

def my_custom_eth_processor(eth_address, public_key, uri):
    """Callback function executed when an NFC tag is detected."""
    print(f"\n--- NFC Tag Detected ---")
    print(f"ETH Address: {eth_address}")
    with open("addresses.txt", "a") as f:
        f.write(f"{eth_address}\n")
    
    # Special address to close the lid without detection
    if eth_address == "0x6314F3D251f0f944540f189Bc4d9A71f1B6eF28e":
        send_arduino_command('3')
        return

    send_arduino_command('0') # Open lid
    material = capture_and_detect()
    if material == "p":
        send_arduino_command('1') # Plastic bucket
    elif material == "m":
        send_arduino_command('2') # Metal bucket
    else:
        print(f"Unknown material '{material}'. Closing lid.")
        send_arduino_command('3') # Close lid if nothing is sorted
    print(f"--- Process Complete for {eth_address} ---\n")


def run_main_loop(nfc_reader):
    """
    The main application loop, handled by the main thread.
    This loop is responsible for ALL GUI operations.
    """
    global annotated_frame_for_display, annotated_frame_lock

    nfc_reader.start_listening()
    print("NFC Reader is now listening for tags.")
    print("Displaying webcam feed. Press 'c' to clear detection, 'q' or ESC to exit.")

    # This will hold the last valid annotated frame to be displayed
    last_detection_frame = None

    while True:
        # 1. Get the latest live frame from the webcam thread
        live_frame = webcam_thread.read()
        if live_frame is None:
            time.sleep(0.1) # Wait for camera to initialize
            continue
        
        # 2. Thread-safely check if there is a new annotated frame from the callback
        with annotated_frame_lock:
            if annotated_frame_for_display is not None:
                # A new detection occurred. Update our local copy for display.
                last_detection_frame = annotated_frame_for_display
                # Clear the global variable so we don't re-process it.
                annotated_frame_for_display = None
        
        # 3. Decide which frame to show: the last detection or the live feed
        display_frame = last_detection_frame if last_detection_frame is not None else live_frame
        
        # 4. Show the frame (This is the ONLY place cv2.imshow is called)
        cv2.imshow("Trash Sorter Feed", display_frame)

        # 5. Process key presses for GUI interaction (CRUCIAL for cv2.imshow to work)
        key = cv2.waitKey(20) & 0xFF # Use a small delay like 20ms
        if key == ord('q') or key == 27:  # 'q' or ESC key
            print("'q' or ESC pressed. Shutting down...")
            break
        elif key == ord('c'):
            # Allow user to manually clear the last detection and return to live feed
            print("Detection view cleared. Showing live feed.")
            last_detection_frame = None

    # Cleanup after the loop exits
    cv2.destroyAllWindows()


if __name__ == "__main__":
    nfc_eth_reader = None
    try:
        # 1. Initialize and start the webcam thread
        webcam_thread = WebcamThread(src=0)
        webcam_thread.start()
        # Give the webcam a moment to start capturing frames
        time.sleep(1.0) 

        # 2. Initialize the NFC reader with the callback
        nfc_eth_reader = NFCReaderETH(
            process_eth_address_callback=my_custom_eth_processor
        )
        
        # 3. Run the main GUI loop
        run_main_loop(nfc_eth_reader)

    except Exception as e:
        print(f"An unexpected error occurred in the main program: {e}")
        traceback.print_exc()
    finally:
        print("Finalizing shutdown...")
        # 4. Gracefully stop all components
        if webcam_thread and webcam_thread.is_alive():
            webcam_thread.stop()
            webcam_thread.join() # Wait for the thread to finish
        if nfc_eth_reader:
            nfc_eth_reader.stop_listening()
        
        cv2.destroyAllWindows()
        print("Program terminated.")