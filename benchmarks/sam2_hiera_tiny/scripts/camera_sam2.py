import time
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


CHECKPOINT = "./checkpoints/sam2_hiera_tiny.pt"
CONFIG = "sam2_hiera_t.yaml"
DEVICE = "cuda"

clicked_point = None
last_mask = None
last_score = None
last_sam2_time = None


def mouse_callback(event, x, y, flags, param):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked point: {clicked_point}")


def overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.35):
    overlay = frame.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def main():
    global clicked_point, last_mask, last_score, last_sam2_time

    print("Loading SAM2 model...")
    print("CUDA available:", torch.cuda.is_available())

    model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(model)

    print("SAM2 loaded successfully")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    cv2.namedWindow("SAM2 Live Camera Demo")
    cv2.setMouseCallback("SAM2 Live Camera Demo", mouse_callback)

    print("Camera opened")
    print("Left click on the target object to run SAM2 segmentation")
    print("Press ESC to quit")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frame")
            break

        display = frame.copy()

        if clicked_point is not None:
            x, y = clicked_point

            print("Running SAM2 segmentation...")

            start_time = time.time()

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)

            input_point = np.array([[x, y]])
            input_label = np.array([1])

            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            best_idx = int(np.argmax(scores))
            last_mask = masks[best_idx].astype(bool)
            last_score = float(scores[best_idx])
            last_sam2_time = time.time() - start_time

            print(f"SAM2 score: {last_score:.3f}")
            print(f"SAM2 time: {last_sam2_time:.3f} s")

            clicked_point = None

        if last_mask is not None:
            display = overlay_mask(display, last_mask)

        if clicked_point is not None:
            cv2.circle(display, clicked_point, 6, (0, 0, 255), -1)

        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

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
            "Left click: SAM2 segment | ESC: quit",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        if last_score is not None:
            cv2.putText(
                display,
                f"SAM2 Score: {last_score:.3f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if last_sam2_time is not None:
            cv2.putText(
                display,
                f"SAM2 Time: {last_sam2_time:.3f} s",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("SAM2 Live Camera Demo", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()