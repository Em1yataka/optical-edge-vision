import cv2
import time
from analysis.image_quality import compute_image_quality
from utils.visualization import draw_fps, draw_counts, draw_quality

from config import MODEL_PATH, CAMERA_ID, CONFIDENCE
from camera.camera import Camera
from detection.yolo_detector import YOLODetector
from analysis.counter import count_objects
from utils.visualization import draw_fps, draw_counts


def main():
    camera = Camera(CAMERA_ID)
    detector = YOLODetector(MODEL_PATH, CONFIDENCE)

    prev_time = time.time()

    while True:
        frame = camera.read()

        if frame is None:
            print("Cannot receive frame")
            break

        result = detector.detect(frame)
        quality = compute_image_quality(frame)
        annotated = result.plot()

        counts = count_objects(result)

        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        draw_fps(annotated, fps)
        draw_counts(annotated, counts)
        draw_quality(annotated, quality)

        cv2.imshow("Smart Camera V3", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()