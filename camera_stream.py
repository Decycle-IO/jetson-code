import cv2
import requests
import time
import base64
import numpy as np

# Replace with your FastAPI endpoint
FASTAPI_URL = "http://192.168.87.210:8000/processImage"  # update endpoint

# Optional: delay between frames (in seconds)
FRAME_DELAY = 0.1

def stream_camera():
    cap = cv2.VideoCapture(0)  # Change to 1 or video file path if needed

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame.")
                continue

            # Prepare the image as bytes
            files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}

            try:
                # Send POST request to FastAPI server
                response = requests.post(FASTAPI_URL, files=files, timeout=5)
                result = response.json()
                print("Detection:", result["detected_classes"])
                # Decode the annotated image from base64
                img_bytes = base64.b64decode(result["image_b64"])
                nparr = np.frombuffer(img_bytes, np.uint8)
                annotated_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                print("Error:", e)
                annotated_frame = frame  # fallback to original

            cv2.imshow("Camera Stream", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(FRAME_DELAY)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")

if __name__ == "__main__":
    stream_camera()