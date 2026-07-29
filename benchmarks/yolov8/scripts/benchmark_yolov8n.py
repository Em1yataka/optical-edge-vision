import json
import os
import platform
import resource
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


MODEL_PATH = Path("yolov8n.pt")
IMAGE_PATH = Path("../test_image.jpg")
OUTPUT_DIR = Path("benchmark_results")

DEVICE = 0
IMAGE_SIZE = 640
CONFIDENCE = 0.4
WARMUP_RUNS = 10
MEASURED_RUNS = 100
USE_FP16 = False


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    original_height, original_width = image.shape[:2]

    print("========================================")
    print("YOLOv8n Jetson Benchmark")
    print("========================================")
    print("Model:", MODEL_PATH)
    print("Image:", IMAGE_PATH)
    print("Original image:", original_width, "x", original_height)
    print("YOLO imgsz:", IMAGE_SIZE)
    print("Warm-up runs:", WARMUP_RUNS)
    print("Measured runs:", MEASURED_RUNS)
    print("FP16:", USE_FP16)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    model = YOLO(str(MODEL_PATH))

    # Run once so that the PyTorch model is moved to CUDA and initialized.
    first_result = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        half=USE_FP16,
        verbose=False,
    )
    torch.cuda.synchronize()
    model_initialization_s = time.perf_counter() - load_start

    pytorch_model = model.model
    total_parameters = sum(p.numel() for p in pytorch_model.parameters())
    trainable_parameters = sum(
        p.numel() for p in pytorch_model.parameters() if p.requires_grad
    )
    parameter_dtypes = sorted(
        {str(p.dtype) for p in pytorch_model.parameters()}
    )

    # Ultralytics model.info reports layers, parameters, gradients and GFLOPs.
    info_result = model.info(
        detailed=False,
        verbose=False,
        imgsz=IMAGE_SIZE,
    )

    print("Model info:", info_result)
    print("Total parameters:", f"{total_parameters:,}")
    print("Parameter dtypes:", parameter_dtypes)

    # Warm-up is excluded from the reported result.
    print("Warming up...")
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model.predict(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device=DEVICE,
                half=USE_FP16,
                verbose=False,
            )
            torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    wall_times_ms = []
    preprocess_times_ms = []
    inference_times_ms = []
    postprocess_times_ms = []
    detection_counts = []

    print("Running measured iterations...")

    with torch.inference_mode():
        for run_index in range(MEASURED_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()

            results = model.predict(
                source=image,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device=DEVICE,
                half=USE_FP16,
                verbose=False,
            )

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            result = results[0]
            speed = result.speed

            wall_times_ms.append(elapsed_ms)
            preprocess_times_ms.append(float(speed.get("preprocess", 0.0)))
            inference_times_ms.append(float(speed.get("inference", 0.0)))
            postprocess_times_ms.append(float(speed.get("postprocess", 0.0)))
            detection_counts.append(
                0 if result.boxes is None else len(result.boxes)
            )

            if (run_index + 1) % 10 == 0:
                print(f"Completed {run_index + 1}/{MEASURED_RUNS}")

    torch.cuda.synchronize()

    peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
    max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    wall_summary = summarize(wall_times_ms)
    preprocess_summary = summarize(preprocess_times_ms)
    inference_summary = summarize(inference_times_ms)
    postprocess_summary = summarize(postprocess_times_ms)

    mean_fps = 1000.0 / wall_summary["mean"]

    result_data = {
        "benchmark_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "name": "YOLOv8n",
            "task": "object detection",
            "checkpoint": str(MODEL_PATH.resolve()),
            "checkpoint_bytes": MODEL_PATH.stat().st_size,
            "parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "parameter_dtypes": parameter_dtypes,
            "quantization": {
                "is_integer_quantized": False,
                "weight_bit_width": 32 if not USE_FP16 else 16,
                "requested_inference_precision": (
                    "FP16" if USE_FP16 else "FP32"
                ),
            },
            "ultralytics_info": list(info_result)
            if isinstance(info_result, tuple)
            else str(info_result),
        },
        "input": {
            "source_image": str(IMAGE_PATH.resolve()),
            "original_width": original_width,
            "original_height": original_height,
            "model_imgsz": IMAGE_SIZE,
            "batch_size": 1,
            "confidence_threshold": CONFIDENCE,
        },
        "protocol": {
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
            "cuda_synchronization": True,
            "includes_preprocessing": True,
            "includes_postprocessing": True,
            "camera_capture_included": False,
            "model_loading_included_in_latency": False,
        },
        "timing_ms": {
            "wall_clock_end_to_end": wall_summary,
            "ultralytics_preprocess": preprocess_summary,
            "ultralytics_inference": inference_summary,
            "ultralytics_postprocess": postprocess_summary,
            "mean_end_to_end_fps": mean_fps,
            "model_initialization_s": model_initialization_s,
        },
        "memory": {
            "cuda_peak_allocated_bytes": peak_allocated_bytes,
            "cuda_peak_allocated_mb": peak_allocated_bytes / (1024**2),
            "cuda_peak_reserved_bytes": peak_reserved_bytes,
            "cuda_peak_reserved_mb": peak_reserved_bytes / (1024**2),
            "process_max_rss_kb": max_rss_kb,
            "process_max_rss_mb": max_rss_kb / 1024,
        },
        "detections": {
            "mean_per_image": float(np.mean(detection_counts)),
            "counts": detection_counts,
        },
        "environment": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "ultralytics": ultralytics.__version__,
            "platform": platform.platform(),
            "power_mode": "15W",
        },
    }

    output_path = OUTPUT_DIR / "yolov8n_fp32_benchmark.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result_data, file, indent=2)

    print()
    print("========================================")
    print("FINAL RESULT")
    print("========================================")
    print(f"Parameters: {total_parameters:,}")
    print("Model info:", info_result)
    print(
        "End-to-end latency:",
        f"{wall_summary['mean']:.3f} ± {wall_summary['std']:.3f} ms",
    )
    print(f"Median latency: {wall_summary['median']:.3f} ms")
    print(f"P95 latency: {wall_summary['p95']:.3f} ms")
    print(f"End-to-end FPS: {mean_fps:.3f}")
    print(
        "Ultralytics inference:",
        f"{inference_summary['mean']:.3f} ms",
    )
    print(
        "Peak CUDA allocated:",
        f"{peak_allocated_bytes / (1024**2):.2f} MB",
    )
    print(
        "Peak CUDA reserved:",
        f"{peak_reserved_bytes / (1024**2):.2f} MB",
    )
    print(f"Process maximum RSS: {max_rss_kb / 1024:.2f} MB")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
