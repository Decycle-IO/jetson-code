# main.py

import io
import cv2
import base64
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# Import your ImageProcessor class
from image_processor import ImageProcessor

# --- Configuration ---
# Adjust these paths to your actual model and class files
# Ensure these files exist where you run the application
SAMPLE_DATA_DIR = Path("./sample_data")
CUSTOM_MODEL_PATH = SAMPLE_DATA_DIR / "custom_model.pt"
CUSTOM_CLASSES_PATH = SAMPLE_DATA_DIR / "classes.json"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Waste Detection API",
    description="An API that uses YOLO and a custom model to detect special types of trash in an image.",
    version="1.0.0"
)

# --- Response Model ---
# This defines the structure of the JSON response for good practice and documentation
class ProcessResponse(BaseModel):
    image_b64: str
    detected_classes: List[str]

# --- Global ImageProcessor Instance ---
# Load models only once when the server starts up
print("Initializing Image Processor...")
try:
    processor = ImageProcessor(
        custom_model_path=CUSTOM_MODEL_PATH,
        custom_classes_path=CUSTOM_CLASSES_PATH,
        target_classes=['bottle', 'cup'],
        yolo_confidence_threshold=0.25,
        custom_model_confidence_threshold=0.5
    )
    print("Image Processor initialized successfully.")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not initialize Image Processor. {e}")
    # In a real production app, you might exit or handle this more gracefully
    processor = None

# --- API Endpoint ---
@app.post("/processImage", response_model=ProcessResponse)
async def process_image(file: UploadFile = File(...)):
    """
    Processes an uploaded image to detect specific types of waste.

    - **Receives**: An image file (`.png`, `.jpg`, etc.).
    - **Returns**: A JSON object with the annotated image in Base64 format
      and a list of detected waste classes.
    """
    if not processor:
        raise HTTPException(status_code=500, detail="Image Processor is not available due to a startup error.")
        
    # 1. Read image data from the uploaded file
    # Using await is important for async performance
    image_data = await file.read()

    # 2. Convert the image data to an OpenCV format
    # We use numpy to create an array from the bytes, then cv2.imdecode
    np_arr = np.frombuffer(image_data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_np is None:
        raise HTTPException(status_code=400, detail="Invalid image file. Could not decode image.")

    # 3. Process the image using your class
    # We use the new `process_single_image` method
    results = processor.process_single_image(image_np)
    annotated_frame = results["annotated_frame"]
    detected_classes = results["detected_classes"]
    
    # 4. Encode the annotated image to return in the response
    # Encode the image to JPEG format in memory
    _, buffer = cv2.imencode('.png', annotated_frame)
    
    # Convert the buffer to a Base64 string
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # 5. Return the final JSON response
    return {
        "image_b64": image_b64,
        "detected_classes": detected_classes
    }

# --- Health Check Endpoint (Good Practice) ---
@app.get("/health")
def health_check():
    return {"status": "ok" if processor else "error"}

# --- Run the server ---
# This allows you to run the app directly with `python main.py`
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)