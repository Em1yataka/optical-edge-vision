import ctypes
import datetime
import gc
import json
import os
import resource
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode
from transformers import Sam3Config, Sam3Model, Sam3Processor


MODEL_PATH = "models/sam3_no_text_bf16"
PROCESSOR_PATH = "models/facebook_sam3"

TEXT_CACHE_PATH = "sam3_person_text_cache.pt"

IMAGE_PATH = (
    "/home/xia/jetson_projects/"
    "sam2_test/test_image.jpg"
)

OUTPUT_JSON = "benchmark_results/sam3_bf16_336_flops.json"

OUTPUT_MASK = "benchmark_results/sam3_bf16_336_flops_mask.png"
OUTPUT_OVERLAY = "benchmark_results/sam3_bf16_336_flops_overlay.png"

IMAGE_SIZE = 336
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
WARMUP_RUNS = 1
MEASURED_RUNS = 1

LIBC = ctypes.CDLL("libc.so.6")


def release_cpu_memory():
    gc.collect()
    LIBC.malloc_trim(0)


def get_memory():
    free_bytes, total_bytes = (
        torch.cuda.mem_get_info()
    )

    return {
        "free_gib": free_bytes / 1024**3,
        "total_gib": total_bytes / 1024**3,
    }


def tensor_is_finite(tensor):
    return bool(
        torch.isfinite(tensor.float()).all()
    )


def move_incrementally(module, device):
    moved_parameter_tensors = 0
    moved_buffer_tensors = 0
    moved_bytes = 0

    for submodule in module.modules():
        parameter_names = list(
            submodule._parameters.keys()
        )

        for name in parameter_names:
            parameter = submodule._parameters[name]

            if parameter is None:
                continue

            gpu_parameter = parameter.detach().to(
                device=device,
                non_blocking=False,
            )

            submodule._parameters[name] = (
                torch.nn.Parameter(
                    gpu_parameter,
                    requires_grad=False,
                )
            )

            moved_parameter_tensors += 1
            moved_bytes += (
                gpu_parameter.numel()
                * gpu_parameter.element_size()
            )

            del parameter
            del gpu_parameter

            if moved_parameter_tensors % 10 == 0:
                release_cpu_memory()

            if moved_parameter_tensors % 50 == 0:
                memory = get_memory()

                print(
                    f"  moved "
                    f"{moved_parameter_tensors} "
                    f"parameter tensors; "
                    f"{moved_bytes / 1024**2:.2f} MiB; "
                    f"{memory['free_gib']:.2f} GiB free"
                )

        buffer_names = list(
            submodule._buffers.keys()
        )

        for name in buffer_names:
            buffer = submodule._buffers[name]

            if buffer is None:
                continue

            if buffer.device.type == "meta":
                raise RuntimeError(
                    f"Meta buffer cannot be moved: {name}"
                )

            gpu_buffer = buffer.to(
                device=device,
                non_blocking=False,
            )

            submodule._buffers[name] = gpu_buffer

            moved_buffer_tensors += 1
            moved_bytes += (
                gpu_buffer.numel()
                * gpu_buffer.element_size()
            )

            del buffer
            del gpu_buffer

    release_cpu_memory()

    return {
        "parameter_tensors": (
            moved_parameter_tensors
        ),
        "buffer_tensors": moved_buffer_tensors,
        "bytes": moved_bytes,
        "mib": moved_bytes / 1024**2,
    }


def create_overlay(image, binary_mask):
    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    mask_bool = binary_mask.astype(bool)

    red = np.zeros_like(image_array)
    red[..., 0] = 255.0

    overlay = image_array.copy()

    overlay[mask_bool] = (
        image_array[mask_bool] * 0.55
        + red[mask_bool] * 0.45
    )

    return Image.fromarray(
        np.clip(
            overlay,
            0,
            255,
        ).astype(np.uint8)
    )


print("=" * 72)
print("SAM3 BF16 336px operator-counted FLOPs profile")
print("=" * 72)

checkpoint_path = os.path.join(
    MODEL_PATH,
    "model.safetensors",
)

if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(checkpoint_path)

if not os.path.isfile(TEXT_CACHE_PATH):
    raise FileNotFoundError(TEXT_CACHE_PATH)

if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(IMAGE_PATH)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")


# ------------------------------------------------------------------
# 1. CUDA initialization
# ------------------------------------------------------------------

print("\n[1/9] Initializing CUDA...")

torch.cuda.init()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

memory_initial = get_memory()

print(
    f"Shared memory: "
    f"{memory_initial['free_gib']:.2f} GiB free / "
    f"{memory_initial['total_gib']:.2f} GiB total"
)


# ------------------------------------------------------------------
# 2. Text-feature cache
# ------------------------------------------------------------------

print("\n[2/9] Loading cached text features...")

text_cache = torch.load(
    TEXT_CACHE_PATH,
    map_location="cpu",
    weights_only=False,
)

required_keys = {
    "text_prompt",
    "text_embeds",
    "attention_mask",
}

missing_keys = (
    required_keys - set(text_cache.keys())
)

if missing_keys:
    raise RuntimeError(
        f"Missing cache keys: {missing_keys}"
    )

text_prompt = text_cache["text_prompt"]
text_embeds_cpu = text_cache["text_embeds"]
attention_mask_cpu = text_cache["attention_mask"]

if list(text_embeds_cpu.shape) != [1, 32, 256]:
    raise RuntimeError(
        "Unexpected text_embeds shape: "
        f"{list(text_embeds_cpu.shape)}"
    )

if list(attention_mask_cpu.shape) != [1, 32]:
    raise RuntimeError(
        "Unexpected attention_mask shape: "
        f"{list(attention_mask_cpu.shape)}"
    )

if not tensor_is_finite(text_embeds_cpu):
    raise RuntimeError(
        "text_embeds contains NaN or Inf"
    )

print(f"Prompt: {text_prompt!r}")
print(
    f"text_embeds: "
    f"{list(text_embeds_cpu.shape)}, "
    f"{text_embeds_cpu.dtype}"
)
print(
    f"attention_mask: "
    f"{list(attention_mask_cpu.shape)}, "
    f"{attention_mask_cpu.dtype}"
)


# ------------------------------------------------------------------
# 3. Image preprocessing
# ------------------------------------------------------------------

print("\n[3/9] Preprocessing image...")

processor = Sam3Processor.from_pretrained(
    PROCESSOR_PATH,
    size={
        "height": IMAGE_SIZE,
        "width": IMAGE_SIZE,
    },
    local_files_only=True,
)

image = Image.open(
    IMAGE_PATH
).convert("RGB")

original_width, original_height = image.size

image_inputs = processor(
    images=image,
    return_tensors="pt",
)

pixel_values_cpu = image_inputs["pixel_values"]

expected_shape = [
    1,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
]

if list(pixel_values_cpu.shape) != expected_shape:
    raise RuntimeError(
        "Unexpected pixel_values shape: "
        f"{list(pixel_values_cpu.shape)}"
    )

if not tensor_is_finite(pixel_values_cpu):
    raise RuntimeError(
        "pixel_values contains NaN or Inf"
    )

print(
    f"Original image: "
    f"{original_width}x{original_height}"
)
print(
    f"Model input: "
    f"{list(pixel_values_cpu.shape)}"
)
print(
    f"Input range: "
    f"{pixel_values_cpu.min().item():.4f} to "
    f"{pixel_values_cpu.max().item():.4f}"
)


# ------------------------------------------------------------------
# 4. Load reduced SAM3
# ------------------------------------------------------------------

print("\n[4/9] Loading reduced SAM3 BF16 model...")

config = Sam3Config.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

config.image_size = IMAGE_SIZE

load_start = time.perf_counter()

model, loading_info = Sam3Model.from_pretrained(
    MODEL_PATH,
    config=config,
    dtype=torch.bfloat16,
    device_map={"": "cpu"},
    local_files_only=True,
    low_cpu_mem_usage=True,
    output_loading_info=True,
)

model.eval()

model_load_time = (
    time.perf_counter() - load_start
)

missing_weight_keys = loading_info.get(
    "missing_keys",
    [],
)

unexpected_weight_keys = loading_info.get(
    "unexpected_keys",
    [],
)

mismatched_weight_keys = loading_info.get(
    "mismatched_keys",
    [],
)

allowed_missing_prefixes = (
    "text_encoder.",
    "text_projection.",
)

invalid_missing_keys = [
    key
    for key in missing_weight_keys
    if not key.startswith(
        allowed_missing_prefixes
    )
]

if invalid_missing_keys:
    raise RuntimeError(
        "Unexpected missing model weights: "
        f"{invalid_missing_keys[:20]}"
    )

if mismatched_weight_keys:
    raise RuntimeError(
        "Mismatched model weights: "
        f"{mismatched_weight_keys[:20]}"
    )

print(
    f"CPU model load time: "
    f"{model_load_time:.2f} s"
)
print(
    f"Missing placeholder text keys: "
    f"{len(missing_weight_keys)}"
)
print(
    f"Unexpected source keys: "
    f"{len(unexpected_weight_keys)}"
)

parameters_before_removal = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    f"Parameters before placeholder removal: "
    f"{parameters_before_removal:,}"
)


# ------------------------------------------------------------------
# 5. Delete tiny placeholder Text Encoder
# ------------------------------------------------------------------

print("\n[5/9] Removing placeholder text modules...")

if model.text_encoder is None:
    raise RuntimeError(
        "text_encoder was not constructed"
    )

if model.text_projection is None:
    raise RuntimeError(
        "text_projection was not constructed"
    )

placeholder_text_parameters = sum(
    parameter.numel()
    for parameter in model.text_encoder.parameters()
)

placeholder_projection_parameters = sum(
    parameter.numel()
    for parameter in model.text_projection.parameters()
)

model.text_encoder = None
model.text_projection = None

release_cpu_memory()

gpu_model_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

memory_after_removal = get_memory()

print(
    f"Removed placeholder Text Encoder: "
    f"{placeholder_text_parameters:,}"
)
print(
    f"Removed placeholder projection: "
    f"{placeholder_projection_parameters:,}"
)
print(
    f"GPU-resident model parameters: "
    f"{gpu_model_parameters:,}"
)
print(
    f"Free after placeholder removal: "
    f"{memory_after_removal['free_gib']:.2f} GiB"
)

if gpu_model_parameters != 486_651_702:
    raise RuntimeError(
        "Unexpected GPU model parameter count: "
        f"{gpu_model_parameters:,}; expected "
        f"486,651,702"
    )

if memory_after_removal["free_gib"] < 3.5:
    raise RuntimeError(
        "Insufficient shared memory before transfer: "
        f"{memory_after_removal['free_gib']:.2f} GiB"
    )


# ------------------------------------------------------------------
# 6. Incremental GPU transfer
# ------------------------------------------------------------------

print(
    "\n[6/9] Moving Vision + DETR + "
    "Mask Decoder to GPU..."
)

transfer_start = time.perf_counter()

transfer = move_incrementally(
    model,
    torch.device("cuda:0"),
)

torch.cuda.synchronize()

transfer_time = (
    time.perf_counter() - transfer_start
)

memory_after_transfer = get_memory()

parameter_devices = sorted({
    str(parameter.device)
    for parameter in model.parameters()
})

if parameter_devices != ["cuda:0"]:
    raise RuntimeError(
        "Parameters are on mixed devices: "
        f"{parameter_devices}"
    )

print(
    f"Transferred: "
    f"{transfer['mib']:.2f} MiB"
)
print(
    f"Transfer time: "
    f"{transfer_time:.3f} s"
)
print(
    f"Free after transfer: "
    f"{memory_after_transfer['free_gib']:.2f} GiB"
)


# ------------------------------------------------------------------
# 7. Warm-up and measured inference
# ------------------------------------------------------------------

print("\n[7/9] Preparing GPU inputs...")

pixel_values = pixel_values_cpu.to(
    device="cuda:0",
    dtype=torch.bfloat16,
)
text_embeds = text_embeds_cpu.to(
    device="cuda:0",
    dtype=torch.bfloat16,
)
attention_mask = attention_mask_cpu.to(
    device="cuda:0",
    dtype=torch.int64,
)
text_output = SimpleNamespace(
    pooler_output=text_embeds,
)

del pixel_values_cpu
del text_embeds_cpu
del attention_mask_cpu
del text_cache
release_cpu_memory()


def forward_once():
    with torch.inference_mode():
        return model(
            pixel_values=pixel_values,
            text_embeds=text_output,
            attention_mask=attention_mask,
            return_dict=True,
        )


print(
    f"\nWarm-up runs: {WARMUP_RUNS}; "
    f"measured runs: {MEASURED_RUNS}"
)

for index in range(WARMUP_RUNS):
    warmup_output = forward_once()
    torch.cuda.synchronize()
    del warmup_output
    print(f"Warm-up {index + 1}/{WARMUP_RUNS}")

release_cpu_memory()
memory_before_measured = get_memory()
latencies_ms = []
score_history = []
active_start_iso = datetime.datetime.now().isoformat()
active_start_epoch_s = time.time()

print("\n[8/9] Running measured inference...")

last_outputs = None
flop_counter = FlopCounterMode(
    mods=model,
    depth=3,
    display=False,
)

for index in range(MEASURED_RUNS):
    torch.cuda.synchronize()
    start = time.perf_counter()

    with flop_counter:
        outputs = forward_once()

    torch.cuda.synchronize()
    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    scores = torch.sigmoid(
        outputs.pred_logits.float()
    )
    if outputs.presence_logits is not None:
        scores = scores * torch.sigmoid(
            outputs.presence_logits.float()
        )

    score_history.append(
        float(scores.max().item())
    )
    latencies_ms.append(elapsed_ms)

    if last_outputs is not None:
        del last_outputs
    last_outputs = outputs

    print(
        f"Measured {index + 1}/{MEASURED_RUNS}: "
        f"{elapsed_ms:.2f} ms"
    )

active_end_epoch_s = time.time()
active_end_iso = datetime.datetime.now().isoformat()
measured_duration_s = (
    active_end_epoch_s - active_start_epoch_s
)
memory_after_measured = get_memory()

total_flops = int(
    flop_counter.get_total_flops()
)

if total_flops <= 0:
    raise RuntimeError(
        "FLOP counter returned a non-positive value"
    )

operator_counted_gflops = (
    total_flops / 1_000_000_000.0
)

raw_flop_counts = (
    flop_counter.get_flop_counts()
    if hasattr(
        flop_counter,
        "get_flop_counts",
    )
    else {}
)

flop_breakdown = {}

for module_name, operation_counts in (
    raw_flop_counts.items()
):
    serialized_operations = {}

    for operation, count in (
        operation_counts.items()
    ):
        serialized_operations[
            str(operation)
        ] = int(count)

    flop_breakdown[
        str(module_name)
    ] = serialized_operations

pred_masks = last_outputs.pred_masks
pred_boxes = last_outputs.pred_boxes
pred_logits = last_outputs.pred_logits
presence_logits = last_outputs.presence_logits

scores = torch.sigmoid(pred_logits.float())
if presence_logits is not None:
    scores = scores * torch.sigmoid(
        presence_logits.float()
    )

scores_flat = scores.reshape(-1)
best_flat_index = int(
    torch.argmax(scores_flat).item()
)
query_count = int(pred_masks.shape[1])
best_query_index = (
    best_flat_index % query_count
)
best_score = float(
    scores_flat[best_flat_index].item()
)

mask_logits = pred_masks[
    0,
    best_query_index,
].float()
resized_mask_logits = F.interpolate(
    mask_logits[None, None, ...],
    size=(original_height, original_width),
    mode="bilinear",
    align_corners=False,
)[0, 0]
binary_mask = (
    torch.sigmoid(resized_mask_logits)
    >= MASK_THRESHOLD
).cpu().numpy().astype(np.uint8)

mask_area_pixels = int(binary_mask.sum())
best_box = pred_boxes[
    0,
    best_query_index,
].float().cpu().tolist()
detected_queries = int(
    (scores_flat >= SCORE_THRESHOLD)
    .sum()
    .item()
)


def timing_statistics(values):
    array = np.asarray(
        values,
        dtype=np.float64,
    )
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "sample_std": float(
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(
            np.percentile(array, 95)
        ),
        "max": float(array.max()),
    }


latency_statistics = timing_statistics(
    latencies_ms
)
mean_latency_s = (
    latency_statistics["mean"] / 1000.0
)
throughput_fps = (
    MEASURED_RUNS / measured_duration_s
)

print("\nMeasured inference completed")
print(
    f"Mean latency: "
    f"{latency_statistics['mean']:.2f} ms"
)
print(
    f"Median latency: "
    f"{latency_statistics['median']:.2f} ms"
)
print(
    f"P95 latency: "
    f"{latency_statistics['p95']:.2f} ms"
)
print(f"Throughput: {throughput_fps:.3f} FPS")


# ------------------------------------------------------------------
# 8. Save final mask and overlay
# ------------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_JSON),
    exist_ok=True,
)

Image.fromarray(
    binary_mask * 255,
    mode="L",
).save(OUTPUT_MASK)

create_overlay(
    image,
    binary_mask,
).save(OUTPUT_OVERLAY)


# ------------------------------------------------------------------
# 9. Save benchmark JSON
# ------------------------------------------------------------------

print("\n[9/9] Saving benchmark metadata...")

process_max_rss_kb = resource.getrusage(
    resource.RUSAGE_SELF
).ru_maxrss

result = {
    "benchmark_version": "1.0",
    "timestamp": (
        datetime.datetime.now().isoformat()
    ),
    "model": {
        "name": "SAM3 reduced no-text runtime",
        "source_generation": "SAM3",
        "path": MODEL_PATH,
        "complete_original_parameters": 840_376_374,
        "gpu_runtime_parameters": gpu_model_parameters,
        "precision": {
            "weights": "BF16",
            "execution": "BF16",
            "cached_text_embedding": "BF16",
            "integer_quantized": False,
            "quantization_bit_width": None,
        },
        "deployment_note": (
            "Text encoder is executed offline on CPU; "
            "cached projected text features are supplied "
            "to the GPU segmentation runtime."
        ),
    },
    "input": {
        "image": IMAGE_PATH,
        "original_size": [
            original_width,
            original_height,
        ],
        "model_input_shape": list(
            pixel_values.shape
        ),
        "prompt": text_prompt,
        "text_embedding_shape": list(
            text_embeds.shape
        ),
        "batch_size": 1,
    },
    "protocol": {
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "cuda_synchronization": True,
        "flop_counter": (
            "torch.utils.flop_counter."
            "FlopCounterMode"
        ),
        "profiled_latency_valid_for_performance": False,
        "model_loading_included_in_latency": False,
        "gpu_transfer_included_in_latency": False,
        "image_preprocessing_included_in_latency": False,
        "mask_postprocessing_included_in_latency": False,
        "camera_capture_included": False,
        "visualization_included": False,
    },
    "flops": {
        "operator_counted_flops": total_flops,
        "operator_counted_gflops": (
            operator_counted_gflops
        ),
        "counting_scope": (
            "One complete forward pass of the "
            "reduced SAM3 GPU runtime at batch 1 "
            "and 336x336 input, including vision "
            "encoder, DETR components, scoring, "
            "and mask decoder; offline text encoder "
            "is excluded."
        ),
        "convention": (
            "PyTorch FlopCounterMode convention. "
            "Only operations with registered FLOP "
            "formulas are counted; the result is an "
            "operator-counted estimate, not an exact "
            "count of every scalar operation."
        ),
        "module_operation_breakdown": (
            flop_breakdown
        ),
    },
    "active_window": {
        "start_iso": active_start_iso,
        "end_iso": active_end_iso,
        "start_epoch_s": active_start_epoch_s,
        "end_epoch_s": active_end_epoch_s,
        "duration_s": measured_duration_s,
        "inference_count": MEASURED_RUNS,
    },
    "timing_ms": {
        "inference": latency_statistics,
        "raw_values": latencies_ms,
        "throughput_fps": throughput_fps,
        "mean_fps_from_latency": (
            1.0 / mean_latency_s
        ),
        "cpu_model_load_s": model_load_time,
        "gpu_transfer_s": transfer_time,
    },
    "prediction_consistency": {
        "best_scores": score_history,
        "score_min": min(score_history),
        "score_max": max(score_history),
        "last_best_query": best_query_index,
        "last_best_score": best_score,
        "last_best_box": best_box,
        "last_detected_queries": detected_queries,
        "last_mask_area_pixels": mask_area_pixels,
    },
    "memory": {
        "total_shared_gib": (
            memory_initial["total_gib"]
        ),
        "free_initial_gib": (
            memory_initial["free_gib"]
        ),
        "free_after_placeholder_removal_gib": (
            memory_after_removal["free_gib"]
        ),
        "free_after_gpu_transfer_gib": (
            memory_after_transfer["free_gib"]
        ),
        "free_before_measured_gib": (
            memory_before_measured["free_gib"]
        ),
        "free_after_measured_gib": (
            memory_after_measured["free_gib"]
        ),
        "process_max_rss_mb": (
            process_max_rss_kb / 1024
        ),
        "note": (
            "Jetson uses unified memory. Free shared "
            "memory and process RSS are reported; "
            "PyTorch allocator peak is invalid when "
            "PYTORCH_NO_CUDA_MEMORY_CACHING=1."
        ),
    },
    "output": {
        "pred_masks_shape": list(
            pred_masks.shape
        ),
        "pred_boxes_shape": list(
            pred_boxes.shape
        ),
        "pred_logits_shape": list(
            pred_logits.shape
        ),
        "mask_file": OUTPUT_MASK,
        "overlay_file": OUTPUT_OVERLAY,
    },
    "transfer": transfer,
    "environment": {
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_caching_disabled": (
            os.environ.get(
                "PYTORCH_NO_CUDA_MEMORY_CACHING"
            ) == "1"
        ),
    },
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as file:
    json.dump(result, file, indent=2)

print("\n" + "=" * 72)
print("SAM3 FLOPs profile completed")
print("=" * 72)
print(
    f"Operator-counted FLOPs: "
    f"{total_flops:,}"
)
print(
    f"Operator-counted GFLOPs: "
    f"{operator_counted_gflops:.3f}"
)
print(
    f"Mean inference: "
    f"{latency_statistics['mean']:.2f} ms"
)
print(
    f"Std: "
    f"{latency_statistics['sample_std']:.2f} ms"
)
print(
    f"P95: "
    f"{latency_statistics['p95']:.2f} ms"
)
print(f"Throughput: {throughput_fps:.3f} FPS")
print(f"Best score: {best_score:.6f}")
print(f"JSON: {OUTPUT_JSON}")
print(f"Mask: {OUTPUT_MASK}")
print(f"Overlay: {OUTPUT_OVERLAY}")
