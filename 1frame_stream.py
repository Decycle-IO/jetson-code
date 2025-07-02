import cv2
import requests
import base64
import numpy as np

FASTAPI_URL = "http://192.168.87.210:8000/processImage"
IMAGE_PATH = "124.png"

# Read the image
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print("Failed to read image.")
    exit(1)

# Encode the frame as JPEG
ret, buffer = cv2.imencode('.jpg', frame)
if not ret:
    print("Failed to encode image.")
    exit(1)

files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}

try:
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

cv2.imshow("Detection Result", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()