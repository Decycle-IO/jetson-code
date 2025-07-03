import cv2
import requests
import base64
import numpy as np
import threading
import queue
import time

def send_frame_worker(frame_queue, result_queue, api_url):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
        try:
            response = requests.post(api_url, files=files, timeout=5)
            result = response.json()
            img_bytes = base64.b64decode(result["image_b64"])
            nparr = np.frombuffer(img_bytes, np.uint8)
            annotated_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            result_queue.put((annotated_frame, result["detected_classes"]))
        except Exception:
            result_queue.put((None, []))

def camera_stream_loop(
    api_url="http://localhost:8000/processImage",
    frame_delay=0.03,
    queue_size=5,
    camera_index=0,
    window_name="Camera Stream"
):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    frame_queue = queue.Queue(maxsize=queue_size)
    result_queue = queue.Queue()
    worker = threading.Thread(target=send_frame_worker, args=(frame_queue, result_queue, api_url), daemon=True)
    worker.start()

    latest_annotated = None
    latest_classes = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not frame_queue.full():
                frame_queue.put(frame.copy())

            while not result_queue.empty():
                latest_annotated, latest_classes = result_queue.get()

            display_frame = latest_annotated if latest_annotated is not None else frame
            if latest_classes:
                cv2.putText(display_frame, str(latest_classes), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(frame_delay)
    finally:
        frame_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")

def start_camera_stream(
    api_url="http://localhost:8000/processImage",
    frame_delay=0.03,
    queue_size=5,
    camera_index=0,
    window_name="Camera Stream"
):
    camera_stream_loop(
        api_url=api_url,
        frame_delay=frame_delay,
        queue_size=queue_size,
        camera_index=camera_index,
        window_name=window_name
    )

if __name__ == "__main__":
    start_camera_stream(api_url="http://192.168.87.210:8000/processImage") 