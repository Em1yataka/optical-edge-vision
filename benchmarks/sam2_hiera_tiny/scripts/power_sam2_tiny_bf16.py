import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


CHECKPOINT = Path("checkpoints/sam2_hiera_tiny.pt")
CONFIG = "sam2_hiera_t.yaml"
IMAGE_PATH = Path("../test_image.jpg")
OUTPUT_DIR = Path("benchmark_results")

DURATION_S = 60.0
WARMUP_RUNS = 5
SOURCE_SIZE = 512


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read {IMAGE_PATH}")

    image = cv2.resize(
        image,
        (SOURCE_SIZE, SOURCE_SIZE),
    )
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    h, w = image.shape[:2]
    point = np.array(
        [[w // 2, h // 2]],
        dtype=np.float32,
    )
    label = np.array([1], dtype=np.int32)

    print("Loading SAM2 Hiera Tiny...")

    model = build_sam2(
        CONFIG,
        str(CHECKPOINT),
        device="cuda",
    )
    model.eval()
    predictor = SAM2ImagePredictor(model)

    def run_once():
        predictor.set_image(image_rgb)
        return predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=True,
        )

    print("Warming up...")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(WARMUP_RUNS):
            run_once()
            torch.cuda.synchronize()

    print()
    print("SAM2 is ready.")
    print("Start tegrastats in the second terminal.")
    input("Press Enter only after tegrastats is running: ")

    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    start_time = datetime.now()
    inference_count = 0

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while True:
            run_once()
            torch.cuda.synchronize()
            inference_count += 1

            if time.perf_counter() - start_wall >= DURATION_S:
                break

    torch.cuda.synchronize()
    end_time = datetime.now()
    elapsed_s = time.perf_counter() - start_wall

    result = {
        "model": "SAM2 Hiera Tiny",
        "generation": "SAM2 original",
        "precision": "BF16 autocast with FP32 weights",
        "power_mode": "15W",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_s": elapsed_s,
        "inference_count": inference_count,
        "mean_time_s":
            elapsed_s / inference_count,
        "mean_time_ms":
            elapsed_s * 1000 / inference_count,
        "throughput_fps":
            inference_count / elapsed_s,
        "workload": {
            "image_encoding_included": True,
            "prompt_and_decoder_included": True,
            "prompt": "one positive center point",
            "multimask_output": True,
            "camera_capture_included": False,
            "visualization_included": False,
        },
    }

    output = (
        OUTPUT_DIR /
        "sam2_hiera_tiny_bf16_power_run.json"
    )
    output.write_text(json.dumps(result, indent=2))

    print()
    print("POWER RUN COMPLETE")
    print("Start:", result["start_time"])
    print("End:", result["end_time"])
    print("Duration:", f"{elapsed_s:.6f} s")
    print("Inferences:", inference_count)
    print(
        "Mean latency:",
        f"{result['mean_time_ms']:.3f} ms",
    )
    print(
        "Throughput:",
        f"{result['throughput_fps']:.3f} FPS",
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
