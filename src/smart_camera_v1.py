import cv2
import time
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "models/yolov8n.pt"
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

prev_time = time.time()
img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break

    results = model(frame, conf=0.5)
    annotated = results[0].plot()

    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    cv2.putText(annotated, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Smart Camera V1", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        img_count += 1
        filename = RESULT_DIR / f"result_{img_count:03d}.jpg"
        cv2.imwrite(str(filename), annotated)
        print(f"Saved {filename}")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()