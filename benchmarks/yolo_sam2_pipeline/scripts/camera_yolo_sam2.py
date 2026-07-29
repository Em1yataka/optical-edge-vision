import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ==============================
# Config
# ==============================
CAMERA_ID = 0

YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.4

SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_tiny.pt"
SAM2_CONFIG = "sam2_hiera_t.yaml"

DEVICE = "cuda"

# Run SAM2 every N frames
SAM2_INTERVAL = 60


# ==============================
# Helper functions
# ==============================
def get_best_person_box(yolo_result):
    best_box = None
    best_conf = 0.0

    for box in yolo_result.boxes:
        cls_id = int(box.cls[0])
        name = yolo_result.names[cls_id]
        conf = float(box.conf[0])

        if name == "person" and conf > best_conf:
            x1, y1, x2, y2 = box.xyxy[0]
            best_box = np.array(
                [float(x1), float(y1), float(x2), float(y2)],
                dtype=np.float32,
            )
            best_conf = conf

    return best_box, best_conf


def overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.35):
    overlay = frame.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_mask_contour(frame, mask):
    mask_uint8 = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    return frame


def draw_box(frame, box, label):
    x1, y1, x2, y2 = box.astype(int)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2,
    )

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )


# ==============================
# Main
# ==============================
def main():
    print("========================================")
    print("YOLO + SAM2 Live Camera Demo")
    print("========================================")

    print("CUDA available:", torch.cuda.is_available())
    print("Device:", DEVICE)

    print("Loading YOLO...")
    yolo = YOLO(YOLO_MODEL)

    print("Loading SAM2...")
    sam2_model = build_sam2(
        SAM2_CONFIG,
        SAM2_CHECKPOINT,
        device=DEVICE,
    )
    predictor = SAM2ImagePredictor(sam2_model)

    print("Models loaded successfully")

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Camera opened")
    print("Press ESC to quit")

    frame_count = 0
    prev_time = time.time()

    last_mask = None
    last_sam2_score = None
    last_sam2_time = None
    last_box = None
    last_yolo_conf = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frame")
            break

        frame_count += 1

        display = frame.copy()

        # ------------------------------
        # YOLO detection every frame
        # ------------------------------
        yolo_results = yolo.predict(
            frame,
            device=0,
            conf=YOLO_CONF,
            verbose=False,
        )

        result = yolo_results[0]
        person_box, person_conf = get_best_person_box(result)

        # ------------------------------
        # SAM2 segmentation every N frames
        # ------------------------------
        if person_box is not None:
            last_box = person_box
            last_yolo_conf = person_conf

            if frame_count % SAM2_INTERVAL == 0 or last_mask is None:
                print("Running SAM2 with YOLO box prompt...")

                start_sam2 = time.time()

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                predictor.set_image(image_rgb)

                masks, scores, _ = predictor.predict(
                    box=person_box,
                    multimask_output=True,
                )

                best_idx = int(np.argmax(scores))
                last_mask = masks[best_idx].astype(bool)
                last_sam2_score = float(scores[best_idx])
                last_sam2_time = time.time() - start_sam2

                print(f"YOLO conf: {last_yolo_conf:.3f}")
                print(f"SAM2 score: {last_sam2_score:.3f}")
                print(f"SAM2 time: {last_sam2_time:.3f} s")

        else:
            # If no person detected, keep previous mask for demo stability.
            pass

        # ------------------------------
        # Draw SAM2 mask
        # ------------------------------
        if last_mask is not None:
            display = overlay_mask(display, last_mask)
            display = draw_mask_contour(display, last_mask)

        # ------------------------------
        # Draw YOLO box
        # ------------------------------
        if last_box is not None:
            label = "person"
            if last_yolo_conf is not None:
                label += f" {last_yolo_conf:.2f}"

            draw_box(display, last_box, label)

        # ------------------------------
        # FPS calculation
        # ------------------------------
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # ------------------------------
        # Draw text
        # ------------------------------
        cv2.putText(
            display,
            f"FPS: {fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display,
            "YOLO + SAM2 Live Demo | ESC: quit",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        if last_yolo_conf is not None:
            cv2.putText(
                display,
                f"YOLO Conf: {last_yolo_conf:.3f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        if last_sam2_score is not None:
            cv2.putText(
                display,
                f"SAM2 Score: {last_sam2_score:.3f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if last_sam2_time is not None:
            cv2.putText(
                display,
                f"SAM2 Time: {last_sam2_time:.3f} s",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            display,
            f"SAM2 Interval: {SAM2_INTERVAL} frames",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO + SAM2 Live Camera Demo", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()