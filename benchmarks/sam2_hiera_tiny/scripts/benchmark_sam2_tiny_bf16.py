import json
import platform
import resource
import sys
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

SOURCE_SIZE = 512
ENCODER_WARMUP = 3
ENCODER_RUNS = 20
PROMPT_WARMUP = 10
PROMPT_RUNS = 100
END_TO_END_WARMUP = 3
END_TO_END_RUNS = 30


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1))
        if values.size > 1 else 0.0,
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def encode_image(predictor, image_rgb):
    predictor.set_image(image_rgb)


def predict_mask(predictor, point, label):
    return predictor.predict(
        point_coords=point,
        point_labels=label,
        multimask_output=True,
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Cannot read {IMAGE_PATH}")

    original_h, original_w = image.shape[:2]

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

    print("========================================")
    print("SAM2 Hiera Tiny BF16 Autocast Benchmark")
    print("========================================")
    print("Checkpoint:", CHECKPOINT)
    print("Config:", CONFIG)
    print("Original image:", original_w, "x", original_h)
    print("Source passed to predictor:", w, "x", h)
    print("Expected internal input: 1024 x 1024")
    print("Precision: BF16 autocast with FP32 weights")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    load_start = time.perf_counter()

    model = build_sam2(
        CONFIG,
        str(CHECKPOINT),
        device="cuda",
    )
    model.eval()

    predictor = SAM2ImagePredictor(model)

    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_start

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )
    trainable_parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    parameter_dtypes = sorted(
        {str(p.dtype) for p in model.parameters()}
    )

    print("Model load:", f"{model_load_s:.3f} s")
    print("Parameters:", f"{total_parameters:,}")
    print("Dtypes:", parameter_dtypes)

    # Image encoder benchmark
    print("Encoder warm-up...")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(ENCODER_WARMUP):
            encode_image(predictor, image_rgb)
            torch.cuda.synchronize()

    encoder_times_ms = []

    print("Measuring image encoding...")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(ENCODER_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()

            encode_image(predictor, image_rgb)

            torch.cuda.synchronize()
            encoder_times_ms.append(
                (time.perf_counter() - start) * 1000
            )

            print(
                f"Encoder {index + 1}/{ENCODER_RUNS}",
                f"{encoder_times_ms[-1]:.3f} ms",
            )

    # Keep one embedding for prompt-only measurement.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        encode_image(predictor, image_rgb)
        torch.cuda.synchronize()

    # Prompt + decoder benchmark
    print("Prompt/decoder warm-up...")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(PROMPT_WARMUP):
            predict_mask(
                predictor,
                input_point,
                input_label,
            )
            torch.cuda.synchronize()

    prompt_times_ms = []

    print("Measuring prompt + decoder...")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(PROMPT_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()

            masks, scores, logits = predict_mask(
                predictor,
                input_point,
                input_label,
            )

            torch.cuda.synchronize()
            prompt_times_ms.append(
                (time.perf_counter() - start) * 1000
            )

            if (index + 1) % 10 == 0:
                print(
                    f"Prompt {index + 1}/{PROMPT_RUNS}"
                )

    # End-to-end benchmark
    print("End-to-end warm-up...")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(END_TO_END_WARMUP):
            encode_image(predictor, image_rgb)
            predict_mask(
                predictor,
                input_point,
                input_label,
            )
            torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    end_to_end_times_ms = []

    print("Measuring end-to-end SAM2...")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(END_TO_END_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()

            encode_image(predictor, image_rgb)
            masks, scores, logits = predict_mask(
                predictor,
                input_point,
                input_label,
            )

            torch.cuda.synchronize()
            end_to_end_times_ms.append(
                (time.perf_counter() - start) * 1000
            )

            print(
                f"End-to-end {index + 1}/"
                f"{END_TO_END_RUNS}",
                f"{end_to_end_times_ms[-1]:.3f} ms",
            )

    torch.cuda.synchronize()

    peak_allocated = int(
        torch.cuda.max_memory_allocated()
    )
    peak_reserved = int(
        torch.cuda.max_memory_reserved()
    )
    max_rss_kb = int(
        resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
    )

    encoder_summary = summarize(encoder_times_ms)
    prompt_summary = summarize(prompt_times_ms)
    end_to_end_summary = summarize(
        end_to_end_times_ms
    )

    best_index = int(np.argmax(scores))

    result = {
        "benchmark_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "name": "SAM2 Hiera Tiny",
            "generation": "SAM2 original",
            "checkpoint":
                str(CHECKPOINT.resolve()),
            "checkpoint_bytes":
                CHECKPOINT.stat().st_size,
            "config": CONFIG,
            "parameters": total_parameters,
            "trainable_parameters":
                trainable_parameters,
            "parameter_dtypes":
                parameter_dtypes,
            "precision": "BF16 autocast with FP32 weights",
            "integer_quantized": False,
        },
        "input": {
            "image_path":
                str(IMAGE_PATH.resolve()),
            "original_size":
                [original_w, original_h],
            "source_size_after_manual_resize":
                [w, h],
            "expected_internal_input":
                [1024, 1024],
            "batch_size": 1,
            "prompt":
                "one positive center point",
            "multimask_output": True,
        },
        "protocol": {
            "encoder_warmup":
                ENCODER_WARMUP,
            "encoder_runs":
                ENCODER_RUNS,
            "prompt_warmup":
                PROMPT_WARMUP,
            "prompt_runs":
                PROMPT_RUNS,
            "end_to_end_warmup":
                END_TO_END_WARMUP,
            "end_to_end_runs":
                END_TO_END_RUNS,
            "cuda_synchronization": True,
            "model_eval_mode": True,
            "model_loading_excluded": True,
            "camera_capture_included": False,
            "visualization_included": False,
        },
        "timing_ms": {
            "image_encoder": encoder_summary,
            "prompt_and_mask_decoder":
                prompt_summary,
            "end_to_end":
                end_to_end_summary,
            "end_to_end_fps":
                1000.0 /
                end_to_end_summary["mean"],
            "model_load_s": model_load_s,
        },
        "memory": {
            "cuda_peak_allocated_bytes":
                peak_allocated,
            "cuda_peak_allocated_mb":
                peak_allocated / (1024**2),
            "cuda_peak_reserved_bytes":
                peak_reserved,
            "cuda_peak_reserved_mb":
                peak_reserved / (1024**2),
            "process_max_rss_kb":
                max_rss_kb,
            "process_max_rss_mb":
                max_rss_kb / 1024,
        },
        "output": {
            "best_score_last_run":
                float(scores[best_index]),
            "mask_count": len(masks),
        },
        "environment": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu":
                torch.cuda.get_device_name(0),
            "platform":
                platform.platform(),
            "power_mode": "15W",
        },
    }

    output_path = (
        OUTPUT_DIR /
        "sam2_hiera_tiny_bf16_benchmark.json"
    )
    output_path.write_text(
        json.dumps(result, indent=2)
    )

    print()
    print("========================================")
    print("FINAL RESULT")
    print("========================================")
    print(
        "Parameters:",
        f"{total_parameters:,}",
    )
    print(
        "Image encoding:",
        f"{encoder_summary['mean']:.3f} ± "
        f"{encoder_summary['std']:.3f} ms",
    )
    print(
        "Prompt + decoder:",
        f"{prompt_summary['mean']:.3f} ± "
        f"{prompt_summary['std']:.3f} ms",
    )
    print(
        "End-to-end:",
        f"{end_to_end_summary['mean']:.3f} ± "
        f"{end_to_end_summary['std']:.3f} ms",
    )
    print(
        "End-to-end P95:",
        f"{end_to_end_summary['p95']:.3f} ms",
    )
    print(
        "End-to-end FPS:",
        f"{1000 / end_to_end_summary['mean']:.3f}",
    )
    print(
        "Peak CUDA allocated:",
        f"{peak_allocated / (1024**2):.2f} MB",
    )
    print(
        "Peak CUDA reserved:",
        f"{peak_reserved / (1024**2):.2f} MB",
    )
    print(
        "Process maximum RSS:",
        f"{max_rss_kb / 1024:.2f} MB",
    )
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
