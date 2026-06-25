import cv2
import time
from pathlib import Path
from collections import Counter
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

    results = model(frame, conf=0.5, verbose=False)
    r = results[0]

    annotated = r.plot()

    class_names = []

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = r.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        class_names.append(name)

        print(
            f"Object: {name}, "
            f"Confidence: {conf:.2f}, "
            f"Box: ({x1}, {y1}) -> ({x2}, {y2})"
        )

    counts = Counter(class_names)

    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    cv2.putText(
        annotated,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    y = 80
    for name, count in counts.items():
        cv2.putText(
            annotated,
            f"{name}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        y += 35

    cv2.imshow("Smart Camera V2", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        img_count += 1
        filename = RESULT_DIR / f"v2_result_{img_count:03d}.jpg"
        cv2.imwrite(str(filename), annotated)
        print(f"Saved {filename}")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()