# client.py

import requests
import cv2
import numpy as np
import base64
import os

# --- Configuration ---
# The URL where your FastAPI server is running
API_URL = "http://127.0.0.1:8000/processImage"
# The path to the image you want to send for processing
IMAGE_PATH = "./sample_data/124.png"

def process_and_display_image(image_path: str, api_url: str):
    """
    Sends an image to the API, receives the processed image, and displays it.
    """
    # 1. Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    # 2. Prepare the image file to be sent in the POST request
    # We open the file in binary mode ('rb')
    with open(image_path, "rb") as image_file:
        files = {"file": (os.path.basename(image_path), image_file, "image/png")}
        
        print(f"Sending '{image_path}' to the API at '{api_url}'...")
        
        try:
            # 3. Make the POST request to the API
            response = requests.post(api_url, files=files)
            
            # Raise an exception if the request returned an error status code
            response.raise_for_status()
            
            # 4. Parse the JSON response from the server
            response_data = response.json()
            
            # 5. Extract the Base64 encoded image string and detected classes
            image_b64_string = response_data.get("image_b64")
            detected_classes = response_data.get("detected_classes", [])
            
            if not image_b64_string:
                print("Error: 'image_b64' not found in the API response.")
                return

            print(f"API Response successful. Detected classes: {detected_classes}")

            # 6. Decode the Base64 string back into bytes
            decoded_image_bytes = base64.b64decode(image_b64_string)
            
            # 7. Convert the bytes into a NumPy array
            image_np_array = np.frombuffer(decoded_image_bytes, np.uint8)
            
            # 8. Decode the NumPy array into an OpenCV image
            final_image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
            
            # 9. Display the final image with bounding boxes
            cv2.imshow("Processed Image from API", final_image)
            print("\nDisplaying processed image. Press any key to close the window.")
            cv2.waitKey(0)  # Wait indefinitely for a key press
            cv2.destroyAllWindows() # Clean up the window

        except requests.exceptions.RequestException as e:
            print(f"An error occurred while communicating with the API: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    process_and_display_image(image_path=IMAGE_PATH, api_url=API_URL)