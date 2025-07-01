# image_processor.py

import cv2
import torch
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms
from typing import List, Callable, Dict, Any, Tuple

# A type hint for a frame, which is a NumPy array
Frame = np.ndarray

class ImageProcessor:
    """
    A class to process images for object detection using YOLO, classify specific
    objects with a custom model, and trigger a callback based on persistent
    detection.
    """

    def __init__(
        self,
        process_waste_callback: Callable[[Dict[str, Any]], None],
        custom_model_path: Path,
        custom_classes_path: Path,
        yolo_model_name: str = "yolov8m.pt",
        target_classes: List[str] = None,
        yolo_confidence_threshold: float = 0.05,
        custom_model_confidence_threshold: float = 0.6,
        trigger_frame_count: int = 5,
        custom_model_input_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initializes the ImageProcessor.

        Args:
            process_waste_callback (Callable): Function to call when a trigger condition is met.
                It receives a dictionary with detection details.
            custom_model_path (Path): Path to the custom .pt model file.
            custom_classes_path (Path): Path to the JSON file with custom class names.
            yolo_model_name (str): The name of the YOLOv8 model to use.
            target_classes (List[str]): YOLO classes that should be further processed by the custom model.
                                        Defaults to ['bottle', 'cup'].
            yolo_confidence_threshold (float): Minimum confidence for YOLO detections.
            custom_model_confidence_threshold (float): Minimum confidence for custom model classifications.
            trigger_frame_count (int): Number of consecutive frames an object must be detected to trigger the callback.
            custom_model_input_size (Tuple[int, int]): The (height, width) to resize crops to for the custom model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # --- Models ---
        self.yolo_model = YOLO(yolo_model_name)
        self._load_custom_model(custom_model_path)
        
        # --- Configuration ---
        self.yolo_names = self.yolo_model.names
        self.custom_classes = self._load_json_classes(custom_classes_path)
        self.target_classes = target_classes if target_classes is not None else ['bottle', 'cup']
        self.yolo_conf_thresh = yolo_confidence_threshold
        self.custom_conf_thresh = custom_model_confidence_threshold
        
        # --- Callback and Trigger Logic ---
        self.process_waste_callback = process_waste_callback
        self.trigger_frame_count = trigger_frame_count
        self.trigger_consecutive_frames = 0
        self.last_triggered_details = None

        # --- Preprocessing for Custom Model ---
        self.custom_model_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(custom_model_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_custom_model(self, model_path: Path):
        """Loads the custom PyTorch model."""
        if not model_path.exists():
            raise FileNotFoundError(f"Custom model not found at {model_path}")
        # When loading a model for inference, it's common to load the state_dict.
        # If your .pt file is the entire model, use torch.load(model_path).
        # We assume it's a state_dict for this example.
        # For a full runnable example, we will create a dummy model and save its state_dict.
        # In a real scenario, you might need a model definition class.
        # For this example, we'll handle this in the `if __name__ == '__main__'` block.
        self.custom_model = torch.load(model_path, map_location=self.device)
        self.custom_model.eval()
        print(f"Custom model loaded from {model_path} and set to evaluation mode.")

    def _load_json_classes(self, path: Path) -> List[str]:
        """Loads class names from a JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Classes JSON not found at {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def process_frame(self, frame: Frame) -> Frame:
        """
        Processes a single frame for object detection and classification.

        Args:
            frame (Frame): The input image/frame as a NumPy array.

        Returns:
            Frame: The frame with bounding boxes and labels drawn on it.
        """
        annotated_frame = frame.copy()
        yolo_results = self.yolo_model(frame, verbose=True)
        
        found_target_this_frame = False
        best_target_details = None

        for result in yolo_results:
            for box in result.boxes:
                # --- Primary YOLO Detection ---
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.yolo_names[cls_id]

                # --- Secondary Custom Classification ---
                if cls_name in self.target_classes and conf >= self.yolo_conf_thresh:
                    # Crop the object from the original frame
                    crop = frame[y1:y2, x1:x2]
                    
                    # Ensure crop is not empty
                    if crop.size == 0:
                        continue

                    # Preprocess and classify with the custom model
                    input_tensor = self.custom_model_transform(crop).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.custom_model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        custom_conf, predicted_idx = torch.max(probabilities, 1)
                    
                    custom_conf = float(custom_conf[0])
                    custom_cls_name = self.custom_classes[predicted_idx[0]]

                    # Update label to show custom classification
                    label = f"{custom_cls_name}: {custom_conf:.2f}"
                    
                    # --- Hysteresis / Trigger Logic ---
                    if custom_conf >= self.custom_conf_thresh:
                        found_target_this_frame = True
                        current_detection = {
                            "waste_type": custom_cls_name,
                            "confidence": custom_conf,
                            "bounding_box": (x1, y1, x2, y2)
                        }
                        # Keep track of the best detection in this frame
                        if best_target_details is None or custom_conf > best_target_details['confidence']:
                             best_target_details = current_detection
                             
                    # --- Annotation ---
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # else:
                #     label = f"{cls_name}: {conf:.2f}"

        # Update trigger counter after processing all objects in the frame
        if found_target_this_frame:
            self.trigger_consecutive_frames += 1
            self.last_triggered_details = best_target_details
        else:
            self.trigger_consecutive_frames = 0 # Reset if no target found
        
        # Check if the trigger condition is met
        if self.trigger_consecutive_frames >= self.trigger_frame_count:
            print(f"TRIGGER MET! Detected for {self.trigger_consecutive_frames} consecutive frames.")
            if self.process_waste_callback and self.last_triggered_details:
                self.process_waste_callback(self.last_triggered_details)
            
            # Reset after triggering to avoid continuous calls
            self.trigger_consecutive_frames = 0
            self.last_triggered_details = None

        # Save the final annotated image
        # cv2.imshow("det", annotated_frame)
        cv2.imwrite("/tmp/detection.png", annotated_frame)
        
        return annotated_frame

# ==============================================================================
# ======================== SAMPLE USAGE EXAMPLE ================================
# ==============================================================================

if __name__ == '__main__':
    
    # --- 1. Define the Callback Function ---
    # This is the function that will be called when the trigger conditions are met.
    def my_waste_processing_function(details: Dict[str, Any]):
        """A sample callback to process detected waste."""
        print("\n--- ACTION TRIGGERED ---")
        print(f"Processing waste of type: {details['waste_type']}")
        print(f"Confidence: {details['confidence']:.2f}")
        print(f"Location (bbox): {details['bounding_box']}")
        print("------------------------\n")

    # --- 2. Create Dummy Files for a Self-Contained Example ---
    # In your real project, you would point these paths to your actual files.
    
    # Create a directory for our sample data
    sample_data_dir = Path("./sample_data")
    sample_data_dir.mkdir(exist_ok=True)
    
    # a. Dummy Custom Model (`custom_model.pt`)
    # A simple model that can classify an image into one of three classes.
    custom_model_path = sample_data_dir / "custom_model.pt"
    # # This is a simple Convolutional Neural Network
    # dummy_model = torch.nn.Sequential(
    #     torch.nn.Conv2d(3, 16, 3, padding=1),
    #     torch.nn.ReLU(),
    #     torch.nn.AdaptiveAvgPool2d((1, 1)),
    #     torch.nn.Flatten(),
    #     torch.nn.Linear(16, 3) # 3 output classes
    # )
    # torch.save(dummy_model, custom_model_path)
    # print(f"Created dummy model at: {custom_model_path}")

    # b. Classes JSON file (`classes.json`)
    classes_path = sample_data_dir / "classes.json"
    # with open(classes_path, 'w') as f:
    #     json.dump(["metal", "other", "plastic"], f)
    # print(f"Created classes file at: {classes_path}")

    # c. Dummy Test Image (`test_image.jpg`)
    # An image with a green rectangle to simulate a "bottle".
    # We use a solid color so YOLO can easily detect it (as a 'bench' or similar).
    # We'll configure our processor to look for 'bench' instead of 'bottle' for this test.
    test_image_path = sample_data_dir / "124.png"
    # dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # cv2.rectangle(dummy_image, (200, 150), (300, 400), (0, 255, 0), -1) # A green "object"
    # cv2.imwrite(str(test_image_path), dummy_image)
    # print(f"Created dummy test image at: {test_image_path}")

    # --- 3. Instantiate the ImageProcessor ---
    print("\nInitializing ImageProcessor...")
    # NOTE: Since our dummy image has a green box, YOLO might detect it as a 'bench' or 'chair'.
    # We will set `target_classes` to `['bench']` to make this example work out-of-the-box.
    # In a real scenario with a bottle, you'd use `['bottle']`.
    processor = ImageProcessor(
        process_waste_callback=my_waste_processing_function,
        custom_model_path=custom_model_path,
        custom_classes_path=classes_path,
        target_classes=['bottle', 'cup'], # Adjusted for our dummy image
        trigger_frame_count=3, # Trigger after 3 consecutive frames
        yolo_confidence_threshold=0.25, # Lowered for a general object
        custom_model_confidence_threshold=0.5
    )

    # --- 4. Simulate a Video Stream ---
    print("\nSimulating video stream...")
    frame_to_process = cv2.imread(str(test_image_path))

    # Simulate 5 frames. The trigger should fire on frame 3.
    for i in range(1, 6):
        print(f"--- Processing frame {i} ---")
        # In a real application, you would get a new frame from a camera here.
        annotated_frame = processor.process_frame(frame_to_process)
        print(f"Consecutive detections: {processor.trigger_consecutive_frames}")

    print(f"\nProcessing finished. Annotated image saved to /tmp/detection.png")