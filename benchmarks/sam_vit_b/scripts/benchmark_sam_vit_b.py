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

from segment_anything import sam_model_registry, SamPredictor


CHECKPOINT = Path("checkpoints/sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
IMAGE_PATH = Path(
    "/home/xia/jetson_projects/sam2_test/test_image.jpg"
)
OUTPUT_DIR = Path("benchmark_results")

SOURCE_SIZE = 256
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

    # Required by the mixed-precision implementation currently used
    # on this Jetson: encoder FP16 output is converted to FP32 before
    # the FP32 prompt encoder and mask decoder.
    predictor.features = predictor.features.float()


def predict_mask(predictor, input_point, input_label):
    return predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
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
    image = cv2.resize(image, (SOURCE_SIZE, SOURCE_SIZE))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]
    input_point = np.array(
        [[w // 2, h // 2]],
        dtype=np.float32,
    )
    input_label = np.array([1], dtype=np.int32)

    print("========================================")
    print("SAM ViT-B Mixed-Precision Benchmark")
    print("========================================")
    print("Checkpoint:", CHECKPOINT)
    print("Original image:", original_w, "x", original_h)
    print("Source image passed to predictor:", w, "x", h)
    print("Model internal image size: 1024 x 1024")
    print("Encoder precision: FP16")
    print("Prompt encoder precision: FP32")
    print("Mask decoder precision: FP32")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    load_start = time.perf_counter()

    sam = sam_model_registry[MODEL_TYPE](
        checkpoint=str(CHECKPOINT)
    )
    sam.to(device="cuda")

    # Reproduce the deployment precision that already worked.
    sam.half()
    sam.prompt_encoder.float()
    sam.mask_decoder.float()
    sam.eval()

    predictor = SamPredictor(sam)

    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - load_start

    total_parameters = sum(p.numel() for p in sam.parameters())
    trainable_parameters = sum(
        p.numel() for p in sam.parameters()
        if p.requires_grad
    )

    encoder_parameters = sum(
        p.numel() for p in sam.image_encoder.parameters()
    )
    prompt_parameters = sum(
        p.numel() for p in sam.prompt_encoder.parameters()
    )
    decoder_parameters = sum(
        p.numel() for p in sam.mask_decoder.parameters()
    )

    encoder_dtypes = sorted(
        {str(p.dtype) for p in sam.image_encoder.parameters()}
    )
    prompt_dtypes = sorted(
        {str(p.dtype) for p in sam.prompt_encoder.parameters()}
    )
    decoder_dtypes = sorted(
        {str(p.dtype) for p in sam.mask_decoder.parameters()}
    )

    print("Model load:", f"{model_load_s:.3f} s")
    print("Total parameters:", f"{total_parameters:,}")
    print("Encoder parameters:", f"{encoder_parameters:,}")
    print("Prompt encoder parameters:", f"{prompt_parameters:,}")
    print("Mask decoder parameters:", f"{decoder_parameters:,}")
    print("Encoder dtypes:", encoder_dtypes)
    print("Prompt dtypes:", prompt_dtypes)
    print("Decoder dtypes:", decoder_dtypes)

    # ------------------------------
    # Encoder benchmark
    # ------------------------------
    print("Encoder warm-up...")

    with torch.inference_mode():
        for _ in range(ENCODER_WARMUP):
            encode_image(predictor, image_rgb)
            torch.cuda.synchronize()

    encoder_times_ms = []

    print("Measuring encoder...")

    with torch.inference_mode():
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

    # Keep one valid image embedding for prompt-only testing.
    with torch.inference_mode():
        encode_image(predictor, image_rgb)
        torch.cuda.synchronize()

    # ------------------------------
    # Prompt + decoder benchmark
    # ------------------------------
    print("Prompt/decoder warm-up...")

    with torch.inference_mode():
        for _ in range(PROMPT_WARMUP):
            predict_mask(
                predictor,
                input_point,
                input_label,
            )
            torch.cuda.synchronize()

    prompt_times_ms = []

    print("Measuring prompt + mask decoder...")

    with torch.inference_mode():
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

    # ------------------------------
    # End-to-end benchmark
    # ------------------------------
    print("End-to-end warm-up...")

    with torch.inference_mode():
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

    print("Measuring end-to-end SAM...")

    with torch.inference_mode():
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
                f"End-to-end {index + 1}/{END_TO_END_RUNS}",
                f"{end_to_end_times_ms[-1]:.3f} ms",
            )

    torch.cuda.synchronize()

    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    max_rss_kb = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )

    encoder_summary = summarize(encoder_times_ms)
    prompt_summary = summarize(prompt_times_ms)
    end_to_end_summary = summarize(end_to_end_times_ms)

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    result = {
        "benchmark_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "name": "SAM ViT-B",
            "checkpoint": str(CHECKPOINT.resolve()),
            "checkpoint_bytes": CHECKPOINT.stat().st_size,
            "model_type": MODEL_TYPE,
            "parameters": {
                "total": total_parameters,
                "trainable": trainable_parameters,
                "image_encoder": encoder_parameters,
                "prompt_encoder": prompt_parameters,
                "mask_decoder": decoder_parameters,
            },
            "precision": {
                "image_encoder": "FP16",
                "image_embedding_after_encoder": "FP32",
                "prompt_encoder": "FP32",
                "mask_decoder": "FP32",
                "integer_quantized": False,
            },
            "dtypes": {
                "image_encoder": encoder_dtypes,
                "prompt_encoder": prompt_dtypes,
                "mask_decoder": decoder_dtypes,
            },
        },
        "input": {
            "image_path": str(IMAGE_PATH),
            "original_size": [original_w, original_h],
            "source_size_after_manual_resize": [w, h],
            "model_internal_input": [1024, 1024],
            "batch_size": 1,
            "prompt": "one positive center point",
            "multimask_output": True,
        },
        "protocol": {
            "encoder_warmup": ENCODER_WARMUP,
            "encoder_runs": ENCODER_RUNS,
            "prompt_warmup": PROMPT_WARMUP,
            "prompt_runs": PROMPT_RUNS,
            "end_to_end_warmup": END_TO_END_WARMUP,
            "end_to_end_runs": END_TO_END_RUNS,
            "cuda_synchronization": True,
            "model_loading_excluded_from_inference": True,
            "camera_capture_included": False,
            "visualization_included": False,
        },
        "timing_ms": {
            "image_encoder": encoder_summary,
            "prompt_and_mask_decoder": prompt_summary,
            "end_to_end": end_to_end_summary,
            "end_to_end_fps":
                1000.0 / end_to_end_summary["mean"],
            "model_load_s": model_load_s,
        },
        "memory": {
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_allocated_mb":
                peak_allocated / (1024**2),
            "cuda_peak_reserved_bytes": peak_reserved,
            "cuda_peak_reserved_mb":
                peak_reserved / (1024**2),
            "process_max_rss_kb": max_rss_kb,
            "process_max_rss_mb": max_rss_kb / 1024,
        },
        "output": {
            "best_score_last_run": best_score,
            "mask_count": len(masks),
        },
        "environment": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
            "power_mode": "15W",
        },
    }

    output_path = (
        OUTPUT_DIR /
        "sam_vit_b_mixed_precision_benchmark.json"
    )
    output_path.write_text(json.dumps(result, indent=2))

    print()
    print("========================================")
    print("FINAL RESULT")
    print("========================================")
    print("Parameters:", f"{total_parameters:,}")
    print(
        "Encoder:",
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
