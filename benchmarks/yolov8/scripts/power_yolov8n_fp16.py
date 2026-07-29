import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


MODEL_PATH = Path("yolov8n.pt")
IMAGE_PATH = Path("../test_image.jpg")
OUTPUT_DIR = Path("benchmark_results")

DURATION_S = 60.0
WARMUP_RUNS = 20
IMAGE_SIZE = 640
CONFIDENCE = 0.4
USE_FP16 = True


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read {IMAGE_PATH}")

    model = YOLO(str(MODEL_PATH))

    print("Loading and warming up YOLOv8n...")

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model.predict(
                image,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device=0,
                half=USE_FP16,
                verbose=False,
            )
            torch.cuda.synchronize()

    print()
    print("YOLO is ready.")
    print("Start tegrastats in the second terminal.")
    input("Press Enter here only after tegrastats is running: ")

    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    start_time = datetime.now()
    inference_count = 0

    with torch.inference_mode():
        while True:
            model.predict(
                image,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device=0,
                half=USE_FP16,
                verbose=False,
            )
            torch.cuda.synchronize()
            inference_count += 1

            if time.perf_counter() - start_wall >= DURATION_S:
                break

    torch.cuda.synchronize()
    end_time = datetime.now()
    elapsed_s = time.perf_counter() - start_wall

    result = {
        "model": "YOLOv8n",
        "precision": "FP16",
        "power_mode": "15W",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_s": elapsed_s,
        "inference_count": inference_count,
        "mean_time_s": elapsed_s / inference_count,
        "mean_time_ms": elapsed_s * 1000 / inference_count,
        "throughput_fps": inference_count / elapsed_s,
    }

    output = OUTPUT_DIR / "yolov8n_fp16_power_run.json"
    output.write_text(json.dumps(result, indent=2))

    print()
    print("POWER RUN COMPLETE")
    print("Start:", result["start_time"])
    print("End:", result["end_time"])
    print("Duration:", f"{elapsed_s:.6f} s")
    print("Inferences:", inference_count)
    print("Mean latency:", f"{result['mean_time_ms']:.3f} ms")
    print("Throughput:", f"{result['throughput_fps']:.3f} FPS")
    print("Saved:", output)


if __name__ == "__main__":
    main()
