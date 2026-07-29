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


# ============================================================
# Configuration
# ============================================================
CAMERA_ID = 0

YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.4
YOLO_IMAGE_SIZE = 640

SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_tiny.pt"
SAM2_CONFIG = "sam2_hiera_t.yaml"

DEVICE = "cuda"

# SAM2 runs once every N camera frames.
SAM2_INTERVAL = 60

# Warm-up frames are excluded from the final measurements.
WARMUP_FRAMES = 60

# Number of frames included in the final measurements.
MEASURED_FRAMES = 600

# Keep True to benchmark the actual live demo with OpenCV display.
DISPLAY_ENABLED = True

OUTPUT_DIRECTORY = Path("benchmark_results")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = (
    OUTPUT_DIRECTORY
    / "yolo_sam2_live_pipeline_bf16_benchmark.json"
)


# ============================================================
# Utility functions
# ============================================================
def synchronize_cuda():
    """Wait until all queued CUDA operations have completed."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(values):
    """Calculate descriptive statistics for a list of values."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    array = np.asarray(values, dtype=np.float64)

    if len(array) > 1:
        standard_deviation = float(
            np.std(array, ddof=1)
        )
    else:
        standard_deviation = 0.0

    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": standard_deviation,
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def get_best_person_box(yolo_result):
    """Return the highest-confidence person bounding box."""
    best_box = None
    best_confidence = 0.0

    for box in yolo_result.boxes:
        class_id = int(box.cls[0])
        class_name = yolo_result.names[class_id]
        confidence = float(box.conf[0])

        if (
            class_name == "person"
            and confidence > best_confidence
        ):
            x1, y1, x2, y2 = box.xyxy[0]

            best_box = np.array(
                [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ],
                dtype=np.float32,
            )

            best_confidence = confidence

    return best_box, best_confidence


def apply_mask_overlay(
    frame,
    mask,
    color=(0, 255, 0),
    alpha=0.35,
):
    """Apply a transparent mask and draw its contour."""
    overlay = frame.copy()
    overlay[mask] = color

    output = cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
    )

    mask_uint8 = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 0),
        2,
    )

    return output


def draw_person_box(frame, box, confidence):
    """Draw the YOLO person bounding box."""
    x1, y1, x2, y2 = box.astype(int)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2,
    )

    label = f"person {confidence:.3f}"

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )


# ============================================================
# Main benchmark
# ============================================================
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print("=" * 70)
    print("YOLOv8n + SAM2 Hiera Tiny Live Pipeline Benchmark")
    print("=" * 70)
    print("YOLO execution precision: FP32")
    print("SAM2 weights: FP32")
    print("SAM2 execution precision: BF16 autocast")
    print(f"SAM2 interval: {SAM2_INTERVAL} frames")
    print(f"Warm-up frames: {WARMUP_FRAMES}")
    print(f"Measured frames: {MEASURED_FRAMES}")
    print(f"OpenCV display included: {DISPLAY_ENABLED}")
    print()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------
    initialization_start = time.perf_counter()

    print("Loading YOLOv8n...")
    yolo = YOLO(YOLO_MODEL)

    print("Loading SAM2 Hiera Tiny...")
    sam2_model = build_sam2(
        SAM2_CONFIG,
        SAM2_CHECKPOINT,
        device=DEVICE,
    )
    sam2_model.eval()

    predictor = SAM2ImagePredictor(sam2_model)

    synchronize_cuda()

    initialization_seconds = (
        time.perf_counter() - initialization_start
    )

    print(
        f"Model initialization completed in "
        f"{initialization_seconds:.3f} seconds"
    )

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------
    camera = cv2.VideoCapture(CAMERA_ID)

    if not camera.isOpened():
        raise RuntimeError(
            f"Cannot open camera device {CAMERA_ID}"
        )

    camera_width = int(
        camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    camera_height = int(
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    camera_reported_fps = float(
        camera.get(cv2.CAP_PROP_FPS)
    )

    print(
        f"Camera opened: {camera_width}x{camera_height}, "
        f"reported FPS={camera_reported_fps:.2f}"
    )
    print("Stand in front of the camera during the test.")
    print("Press ESC to stop early.")
    print()

    # Reset CUDA peak counters after model initialization.
    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Timing arrays in milliseconds
    # --------------------------------------------------------
    camera_capture_times = []
    yolo_times = []
    sam2_encoder_times = []
    sam2_decoder_times = []
    sam2_complete_times = []
    overlay_times = []
    display_times = []
    complete_loop_times = []

    yolo_confidences = []
    sam2_scores = []

    # --------------------------------------------------------
    # Counters and state
    # --------------------------------------------------------
    total_frame_count = 0
    measured_frame_count = 0
    detected_person_frame_count = 0
    sam2_inference_count = 0

    last_mask = None
    last_box = None
    last_yolo_confidence = None
    last_sam2_score = None

    benchmark_start_datetime = None
    benchmark_end_datetime = None
    benchmark_start_time = None

    stop_requested = False

    # --------------------------------------------------------
    # Main camera loop
    # --------------------------------------------------------
    while not stop_requested:
        loop_start = time.perf_counter()

        # Camera capture
        capture_start = time.perf_counter()
        success, frame = camera.read()
        capture_end = time.perf_counter()

        if not success:
            print("Camera frame capture failed")
            break

        total_frame_count += 1

        is_measured_frame = (
            total_frame_count > WARMUP_FRAMES
        )

        if (
            is_measured_frame
            and benchmark_start_time is None
        ):
            benchmark_start_time = loop_start
            benchmark_start_datetime = (
                datetime.now().isoformat()
            )

        display_frame = frame.copy()

        # ----------------------------------------------------
        # YOLO FP32 detection
        # ----------------------------------------------------
        synchronize_cuda()
        yolo_start = time.perf_counter()

        yolo_results = yolo.predict(
            frame,
            device=0,
            conf=YOLO_CONFIDENCE,
            imgsz=YOLO_IMAGE_SIZE,
            half=False,
            verbose=False,
        )

        synchronize_cuda()
        yolo_end = time.perf_counter()

        yolo_result = yolo_results[0]

        person_box, person_confidence = (
            get_best_person_box(yolo_result)
        )

        # ----------------------------------------------------
        # SAM2 BF16 segmentation
        # ----------------------------------------------------
        if person_box is not None:
            last_box = person_box
            last_yolo_confidence = person_confidence

            if is_measured_frame:
                detected_person_frame_count += 1
                yolo_confidences.append(
                    person_confidence
                )

            should_run_sam2 = (
                last_mask is None
                or total_frame_count % SAM2_INTERVAL == 0
            )

            if should_run_sam2:
                image_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
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
                        masks, scores, _ = (
                            predictor.predict(
                                box=person_box,
                                multimask_output=True,
                            )
                        )

                synchronize_cuda()
                decoder_end = time.perf_counter()

                best_index = int(np.argmax(scores))

                last_mask = masks[
                    best_index
                ].astype(bool)

                last_sam2_score = float(
                    scores[best_index]
                )

                if is_measured_frame:
                    encoder_time_ms = (
                        encoder_end - encoder_start
                    ) * 1000.0

                    decoder_time_ms = (
                        decoder_end - decoder_start
                    ) * 1000.0

                    complete_sam2_time_ms = (
                        encoder_time_ms
                        + decoder_time_ms
                    )

                    sam2_encoder_times.append(
                        encoder_time_ms
                    )
                    sam2_decoder_times.append(
                        decoder_time_ms
                    )
                    sam2_complete_times.append(
                        complete_sam2_time_ms
                    )

                    sam2_scores.append(
                        last_sam2_score
                    )

                    sam2_inference_count += 1

        # ----------------------------------------------------
        # Mask overlay and bounding box
        # ----------------------------------------------------
        overlay_start = time.perf_counter()

        if last_mask is not None:
            if last_mask.shape == frame.shape[:2]:
                display_frame = apply_mask_overlay(
                    display_frame,
                    last_mask,
                )

        if (
            last_box is not None
            and last_yolo_confidence is not None
        ):
            draw_person_box(
                display_frame,
                last_box,
                last_yolo_confidence,
            )

        overlay_end = time.perf_counter()

        # ----------------------------------------------------
        # OpenCV text and display
        # ----------------------------------------------------
        display_start = time.perf_counter()

        if DISPLAY_ENABLED:
            cv2.putText(
                display_frame,
                (
                    f"Measured frames: "
                    f"{measured_frame_count}/"
                    f"{MEASURED_FRAMES}"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                "YOLO FP32 + SAM2 BF16",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if last_sam2_score is not None:
                cv2.putText(
                    display_frame,
                    (
                        f"SAM2 score: "
                        f"{last_sam2_score:.3f}"
                    ),
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(
                "YOLO + SAM2 Pipeline Benchmark",
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                stop_requested = True

        display_end = time.perf_counter()
        loop_end = time.perf_counter()

        # ----------------------------------------------------
        # Record measured frame
        # ----------------------------------------------------
        if is_measured_frame:
            measured_frame_count += 1

            camera_capture_times.append(
                (capture_end - capture_start)
                * 1000.0
            )

            yolo_times.append(
                (yolo_end - yolo_start)
                * 1000.0
            )

            overlay_times.append(
                (overlay_end - overlay_start)
                * 1000.0
            )

            display_times.append(
                (display_end - display_start)
                * 1000.0
            )

            complete_loop_times.append(
                (loop_end - loop_start)
                * 1000.0
            )

            if measured_frame_count % 50 == 0:
                print(
                    f"Measured "
                    f"{measured_frame_count}/"
                    f"{MEASURED_FRAMES}; "
                    f"SAM2 runs="
                    f"{sam2_inference_count}"
                )

            if measured_frame_count >= MEASURED_FRAMES:
                stop_requested = True

    benchmark_end_datetime = datetime.now().isoformat()

    if benchmark_start_time is not None:
        measured_duration_seconds = (
            time.perf_counter()
            - benchmark_start_time
        )
    else:
        measured_duration_seconds = 0.0

    camera.release()
    cv2.destroyAllWindows()
    synchronize_cuda()

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------
    if measured_duration_seconds > 0:
        measured_throughput_fps = (
            measured_frame_count
            / measured_duration_seconds
        )
    else:
        measured_throughput_fps = 0.0

    if complete_loop_times:
        mean_complete_loop_ms = float(
            np.mean(complete_loop_times)
        )
        fps_from_mean_loop_time = (
            1000.0 / mean_complete_loop_ms
        )
    else:
        fps_from_mean_loop_time = 0.0

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

    # Linux reports ru_maxrss in KiB.
    process_max_rss_mb = (
        process_max_rss_kb / 1024.0
    )

    # --------------------------------------------------------
    # Build JSON result
    # --------------------------------------------------------
    result = {
        "benchmark_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "pipeline": {
            "name": (
                "YOLOv8n FP32 + "
                "SAM2 Hiera Tiny BF16"
            ),
            "task": (
                "person detection followed by "
                "box-prompted segmentation"
            ),
        },
        "models": {
            "yolo": {
                "name": "YOLOv8n",
                "checkpoint": YOLO_MODEL,
                "precision": "FP32",
                "integer_quantized": False,
                "parameters": 3151904,
                "flops_g": 8.9,
            },
            "sam2": {
                "name": "SAM2 Hiera Tiny",
                "generation": "SAM2 original",
                "checkpoint": SAM2_CHECKPOINT,
                "config": SAM2_CONFIG,
                "weight_precision": "FP32",
                "execution_precision": (
                    "BF16 autocast"
                ),
                "integer_quantized": False,
                "parameters": 38945986,
            },
        },
        "camera": {
            "device": CAMERA_ID,
            "width": camera_width,
            "height": camera_height,
            "reported_fps": camera_reported_fps,
        },
        "protocol": {
            "power_mode": "15W",
            "warmup_frames": WARMUP_FRAMES,
            "requested_measured_frames": (
                MEASURED_FRAMES
            ),
            "actual_measured_frames": (
                measured_frame_count
            ),
            "sam2_interval_frames": (
                SAM2_INTERVAL
            ),
            "camera_capture_included": True,
            "yolo_preprocessing_included": True,
            "yolo_inference_included": True,
            "yolo_postprocessing_included": True,
            "sam2_image_encoding_included": True,
            "sam2_prompt_and_decoder_included": True,
            "mask_overlay_included": True,
            "opencv_display_included": (
                DISPLAY_ENABLED
            ),
            "model_loading_excluded": True,
            "yolo_runs_every_frame": True,
            "sam2_requires_person_detection": True,
        },
        "run": {
            "start_time": benchmark_start_datetime,
            "end_time": benchmark_end_datetime,
            "duration_s": measured_duration_seconds,
            "measured_frames": (
                measured_frame_count
            ),
            "detected_person_frames": (
                detected_person_frame_count
            ),
            "sam2_inference_count": (
                sam2_inference_count
            ),
            "throughput_fps": (
                measured_throughput_fps
            ),
            "fps_from_mean_loop_time": (
                fps_from_mean_loop_time
            ),
        },
        "timing_ms": {
            "camera_capture": summarize(
                camera_capture_times
            ),
            "yolo_end_to_end": summarize(
                yolo_times
            ),
            "sam2_image_encoder": summarize(
                sam2_encoder_times
            ),
            "sam2_prompt_and_decoder": summarize(
                sam2_decoder_times
            ),
            "sam2_complete": summarize(
                sam2_complete_times
            ),
            "mask_overlay_and_box": summarize(
                overlay_times
            ),
            "opencv_display": summarize(
                display_times
            ),
            "complete_frame_loop": summarize(
                complete_loop_times
            ),
        },
        "predictions": {
            "yolo_confidence": summarize(
                yolo_confidences
            ),
            "sam2_score": summarize(
                sam2_scores
            ),
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
                process_max_rss_mb
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

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("Benchmark complete")
    print("=" * 70)

    print(
        f"Measured frames: "
        f"{measured_frame_count}"
    )

    print(
        f"Person-detection frames: "
        f"{detected_person_frame_count}"
    )

    print(
        f"SAM2 executions: "
        f"{sam2_inference_count}"
    )

    print(
        f"Measured throughput: "
        f"{measured_throughput_fps:.3f} FPS"
    )

    peak_cuda_mb = result["memory"][
        "cuda_peak_allocated_mb"
    ]

    print(
        f"Peak CUDA allocated: "
        f"{peak_cuda_mb:.2f} MB"
    )

    print(f"Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()