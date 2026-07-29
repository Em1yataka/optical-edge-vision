import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from segment_anything import sam_model_registry, SamPredictor


CHECKPOINT = Path("checkpoints/sam_vit_b_01ec64.pth")
IMAGE_PATH = Path(
    "/home/xia/jetson_projects/sam2_test/test_image.jpg"
)
OUTPUT_DIR = Path("benchmark_results")

DURATION_S = 60.0
WARMUP_RUNS = 5
SOURCE_SIZE = 256


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
    input_point = np.array(
        [[w // 2, h // 2]],
        dtype=np.float32,
    )
    input_label = np.array(
        [1],
        dtype=np.int32,
    )

    print("Loading SAM ViT-B...")

    sam = sam_model_registry["vit_b"](
        checkpoint=str(CHECKPOINT)
    )
    sam.to(device="cuda")

    sam.half()
    sam.prompt_encoder.float()
    sam.mask_decoder.float()
    sam.eval()

    predictor = SamPredictor(sam)

    def run_once():
        predictor.set_image(image_rgb)
        predictor.features = predictor.features.float()

        return predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

    print("Warming up...")

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            run_once()
            torch.cuda.synchronize()

    print()
    print("SAM is ready.")
    print("Start tegrastats in the second terminal.")
    input("Press Enter only after tegrastats is running: ")

    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    start_time = datetime.now()
    inference_count = 0

    with torch.inference_mode():
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
        "model": "SAM ViT-B",
        "precision": {
            "image_encoder": "FP16",
            "embedding": "FP32",
            "prompt_encoder": "FP32",
            "mask_decoder": "FP32",
        },
        "power_mode": "15W",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_s": elapsed_s,
        "inference_count": inference_count,
        "mean_time_s": elapsed_s / inference_count,
        "mean_time_ms":
            elapsed_s * 1000 / inference_count,
        "throughput_fps":
            inference_count / elapsed_s,
        "workload": {
            "includes_image_encoding": True,
            "includes_prompt_and_decoder": True,
            "camera_capture_included": False,
            "visualization_included": False,
            "prompt": "one positive center point",
            "multimask_output": True,
        },
    }

    output = (
        OUTPUT_DIR /
        "sam_vit_b_mixed_power_run.json"
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
