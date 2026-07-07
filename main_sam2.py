import cv2
import time

from config import MODEL_PATH, CAMERA_ID, CONFIDENCE
from camera.camera import Camera
from detection.yolo_detector import YOLODetector
from analysis.counter import count_objects
from analysis.image_quality import compute_image_quality
from analysis.logger import CSVLogger
from utils.visualization import draw_fps, draw_counts, draw_quality
from segmentation.sam2_segmenter import SAM2Segmenter


def get_first_person_box(result):
    for box in result.boxes:
        cls_id = int(box.cls[0])
        name = result.names[cls_id]

        if name == "person":
            x1, y1, x2, y2 = box.xyxy[0]
            return int(x1), int(y1), int(x2), int(y2)

    return None


def overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.35):
    overlay = frame.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def main():
    camera = Camera(CAMERA_ID)
    detector = YOLODetector(MODEL_PATH, CONFIDENCE)
    segmenter = SAM2Segmenter(device="cpu")
    logger = CSVLogger()

    prev_time = time.time()

    frame_count = 0
    sam2_interval = 100

    last_mask = None
    last_sam2_score = None
    last_mask_area = None
    last_sam2_time_ms = None

    while True:
        frame = camera.read()

        if frame is None:
            print("Cannot receive frame")
            break

        frame_count += 1

        result = detector.detect(frame)
        quality = compute_image_quality(frame)
        counts = count_objects(result)

        annotated = result.plot()

        person_box = get_first_person_box(result)

        if person_box is not None and frame_count % sam2_interval == 0:
            print("Running SAM2 segmentation with box prompt...")

            start_sam2 = time.time()

            last_mask, last_sam2_score = segmenter.segment_from_box(
                frame,
                person_box,
            )

            last_sam2_time_ms = (time.time() - start_sam2) * 1000
            last_mask_area = int(last_mask.sum())

            print(f"SAM2 score: {last_sam2_score:.3f}")
            print(f"Mask area: {last_mask_area}")
            print(f"SAM2 time: {last_sam2_time_ms:.2f} ms")

        if last_mask is not None:
            annotated = overlay_mask(annotated, last_mask)

            cv2.putText(
                annotated,
                f"SAM2: {last_sam2_score:.3f}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            cv2.putText(
                annotated,
                f"Mask Area: {last_mask_area}",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            cv2.putText(
                annotated,
                f"SAM2 Time: {last_sam2_time_ms:.1f} ms",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        draw_fps(annotated, fps)
        draw_counts(annotated, counts)
        draw_quality(annotated, quality)

        logger.log(
            fps,
            quality,
            counts,
            sam2_score=last_sam2_score,
            mask_area=last_mask_area,
            sam2_time_ms=last_sam2_time_ms,
        )

        cv2.imshow("Smart Camera with SAM2", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()