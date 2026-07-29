import json
import resource
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


IMAGE_PATH = "../test_image.jpg"

YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.4
YOLO_IMAGE_SIZE = 640

SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_tiny.pt"
SAM2_CONFIG = "sam2_hiera_t.yaml"

WARMUP_RUNS = 10
MEASURED_RUNS = 100

OUTPUT_DIRECTORY = Path("benchmark_results")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = (
    OUTPUT_DIRECTORY
    / "yolo_fp32_sam2_bf16_offline_pipeline_benchmark.json"
)


def synchronize_cuda():
    torch.cuda.synchronize()


def summarize(values):
    array = np.asarray(values, dtype=np.float64)

    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": (
            float(np.std(array, ddof=1))
            if len(array) > 1
            else 0.0
        ),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def get_best_person_box(result):
    best_box = None
    best_confidence = 0.0

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])

        if (
            class_name == "person"
            and confidence > best_confidence
        ):
            best_box = np.asarray(
                box.xyxy[0].detach().cpu(),
                dtype=np.float32,
            )
            best_confidence = confidence

    if best_box is None:
        raise RuntimeError(
            "YOLO did not detect a person in the test image"
        )

    return best_box, best_confidence


def run_pipeline(
    yolo,
    predictor,
    image_bgr,
    image_rgb,
):
    synchronize_cuda()
    total_start = time.perf_counter()

    # YOLO preprocessing, inference and postprocessing
    synchronize_cuda()
    yolo_start = time.perf_counter()

    results = yolo.predict(
        image_bgr,
        device=0,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        half=False,
        verbose=False,
    )

    synchronize_cuda()
    yolo_end = time.perf_counter()

    person_box, yolo_confidence = (
        get_best_person_box(results[0])
    )

    # SAM2 image encoder
    synchronize_cuda()
    encoder_start = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            predictor.set_image(image_rgb)

    synchronize_cuda()
    encoder_end = time.perf_counter()

    # SAM2 prompt encoder and mask decoder
    synchronize_cuda()
    decoder_start = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            masks, scores, _ = predictor.predict(
                box=person_box,
                multimask_output=True,
            )

    synchronize_cuda()
    decoder_end = time.perf_counter()

    best_index = int(np.argmax(scores))
    best_mask = masks[best_index].astype(bool)
    best_score = float(scores[best_index])

    synchronize_cuda()
    total_end = time.perf_counter()

    output = {
        "yolo_ms": (
            yolo_end - yolo_start
        ) * 1000.0,
        "sam2_encoder_ms": (
            encoder_end - encoder_start
        ) * 1000.0,
        "sam2_decoder_ms": (
            decoder_end - decoder_start
        ) * 1000.0,
        "sam2_complete_ms": (
            decoder_end - encoder_start
        ) * 1000.0,
        "pipeline_total_ms": (
            total_end - total_start
        ) * 1000.0,
        "yolo_confidence": yolo_confidence,
        "sam2_score": best_score,
        "mask_area_pixels": int(best_mask.sum()),
        "person_box": person_box.tolist(),
    }

    return output, best_mask


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    image_bgr = cv2.imread(IMAGE_PATH)

    if image_bgr is None:
        raise FileNotFoundError(IMAGE_PATH)

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    original_height, original_width = (
        image_bgr.shape[:2]
    )

    print("=" * 70)
    print("Offline YOLO FP32 + SAM2 BF16 Pipeline Benchmark")
    print("=" * 70)
    print(f"Image: {IMAGE_PATH}")
    print(
        f"Image size: "
        f"{original_width}x{original_height}"
    )
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Measured runs: {MEASURED_RUNS}")
    print()

    initialization_start = time.perf_counter()

    yolo = YOLO(YOLO_MODEL)

    sam2_model = build_sam2(
        SAM2_CONFIG,
        SAM2_CHECKPOINT,
        device="cuda",
    )
    sam2_model.eval()

    predictor = SAM2ImagePredictor(sam2_model)

    synchronize_cuda()

    initialization_seconds = (
        time.perf_counter()
        - initialization_start
    )

    # Warm-up
    print("Running warm-up...")

    for index in range(WARMUP_RUNS):
        run_pipeline(
            yolo,
            predictor,
            image_bgr,
            image_rgb,
        )

        print(
            f"\rWarm-up "
            f"{index + 1}/{WARMUP_RUNS}",
            end="",
            flush=True,
        )

    print()

    torch.cuda.reset_peak_memory_stats()

    yolo_times = []
    encoder_times = []
    decoder_times = []
    sam2_complete_times = []
    pipeline_total_times = []

    yolo_confidences = []
    sam2_scores = []
    mask_areas = []

    last_box = None
    last_mask = None

    start_datetime = datetime.now().isoformat()
    measurement_start = time.perf_counter()

    for index in range(MEASURED_RUNS):
        output, mask = run_pipeline(
            yolo,
            predictor,
            image_bgr,
            image_rgb,
        )

        yolo_times.append(output["yolo_ms"])
        encoder_times.append(
            output["sam2_encoder_ms"]
        )
        decoder_times.append(
            output["sam2_decoder_ms"]
        )
        sam2_complete_times.append(
            output["sam2_complete_ms"]
        )
        pipeline_total_times.append(
            output["pipeline_total_ms"]
        )

        yolo_confidences.append(
            output["yolo_confidence"]
        )
        sam2_scores.append(
            output["sam2_score"]
        )
        mask_areas.append(
            output["mask_area_pixels"]
        )

        last_box = output["person_box"]
        last_mask = mask

        if (index + 1) % 10 == 0:
            print(
                f"Measured "
                f"{index + 1}/{MEASURED_RUNS}"
            )

    synchronize_cuda()

    measurement_duration_seconds = (
        time.perf_counter()
        - measurement_start
    )
    end_datetime = datetime.now().isoformat()

    throughput_fps = (
        MEASURED_RUNS
        / measurement_duration_seconds
    )

    cuda_peak_allocated_bytes = int(
        torch.cuda.max_memory_allocated()
    )
    cuda_peak_reserved_bytes = int(
        torch.cuda.max_memory_reserved()
    )

    process_max_rss_kb = int(
        resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
    )

    result = {
        "benchmark_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "pipeline": {
            "name": (
                "YOLOv8n FP32 + "
                "SAM2 Hiera Tiny BF16"
            ),
            "input": IMAGE_PATH,
            "input_size": [
                original_width,
                original_height,
            ],
            "task": (
                "person detection followed by "
                "box-prompted segmentation"
            ),
        },
        "models": {
            "yolo": {
                "name": "YOLOv8n",
                "precision": "FP32",
                "parameters": 3151904,
                "flops_g": 8.9,
                "integer_quantized": False,
            },
            "sam2": {
                "name": "SAM2 Hiera Tiny",
                "weights": "FP32",
                "execution": "BF16 autocast",
                "parameters": 38945986,
                "integer_quantized": False,
            },
        },
        "protocol": {
            "power_mode": "15W",
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
            "cuda_synchronization": True,
            "image_loading_included": False,
            "color_conversion_included": False,
            "yolo_preprocessing_included": True,
            "yolo_inference_included": True,
            "yolo_postprocessing_included": True,
            "box_selection_included": True,
            "sam2_image_encoding_included": True,
            "sam2_prompt_and_decoder_included": True,
            "camera_capture_included": False,
            "display_included": False,
            "model_loading_included": False,
        },
        "run": {
            "start_time": start_datetime,
            "end_time": end_datetime,
            "duration_s": (
                measurement_duration_seconds
            ),
            "inference_count": MEASURED_RUNS,
            "throughput_fps": throughput_fps,
        },
        "timing_ms": {
            "yolo_end_to_end": summarize(
                yolo_times
            ),
            "sam2_image_encoder": summarize(
                encoder_times
            ),
            "sam2_prompt_and_decoder": summarize(
                decoder_times
            ),
            "sam2_complete": summarize(
                sam2_complete_times
            ),
            "complete_pipeline": summarize(
                pipeline_total_times
            ),
        },
        "predictions": {
            "yolo_confidence": summarize(
                yolo_confidences
            ),
            "sam2_score": summarize(
                sam2_scores
            ),
            "mask_area_pixels": summarize(
                mask_areas
            ),
            "last_person_box": last_box,
        },
        "memory": {
            "cuda_peak_allocated_bytes": (
                cuda_peak_allocated_bytes
            ),
            "cuda_peak_allocated_mb": (
                cuda_peak_allocated_bytes
                / (1024 ** 2)
            ),
            "cuda_peak_reserved_bytes": (
                cuda_peak_reserved_bytes
            ),
            "cuda_peak_reserved_mb": (
                cuda_peak_reserved_bytes
                / (1024 ** 2)
            ),
            "process_max_rss_kb": (
                process_max_rss_kb
            ),
            "process_max_rss_mb": (
                process_max_rss_kb / 1024.0
            ),
        },
        "environment": {
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "power_mode": "15W",
        },
        "model_initialization_s": (
            initialization_seconds
        ),
    }

    with open(OUTPUT_JSON, "w") as output_file:
        json.dump(
            result,
            output_file,
            indent=2,
        )

    mask_output_path = (
        OUTPUT_DIRECTORY
        / "yolo_sam2_offline_last_mask.png"
    )

    cv2.imwrite(
        str(mask_output_path),
        last_mask.astype(np.uint8) * 255,
    )

    print()
    print("=" * 70)
    print("Offline pipeline benchmark complete")
    print("=" * 70)
    print(
        f"Throughput: {throughput_fps:.3f} FPS"
    )
    print(
        f"Mean pipeline latency: "
        f"{np.mean(pipeline_total_times):.3f} ms"
    )
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved mask: {mask_output_path}")


if __name__ == "__main__":
    main()