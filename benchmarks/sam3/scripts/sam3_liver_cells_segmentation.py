import ctypes
import gc
import json
import os
import resource
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Sam3Config, Sam3Model, Sam3Processor


MODEL_PATH = "models/sam3_no_text_bf16"
PROCESSOR_PATH = "models/facebook_sam3"

TEXT_CACHE_PATH = "sam3_cell_text_cache.pt"

IMAGE_PATH = "microscope_liver_cells.bmp"

OUTPUT_JSON = (
    "sam3_full_336_segmentation_test.json"
)

OUTPUT_MASK = "sam3_full_336_mask.png"
OUTPUT_OVERLAY = "sam3_full_336_overlay.png"

IMAGE_SIZE = 336
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

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
print("Reduced SAM3 BF16 336px full segmentation")
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
# 7. Complete inference
# ------------------------------------------------------------------

print("\n[7/9] Running complete SAM3 inference...")

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

del pixel_values_cpu
del text_embeds_cpu
del attention_mask_cpu
del text_cache

release_cpu_memory()

torch.cuda.synchronize()

inference_start = time.perf_counter()

with torch.inference_mode():
    outputs = model(
        pixel_values=pixel_values,
        text_embeds=SimpleNamespace(
            pooler_output=text_embeds,
        ),
        attention_mask=attention_mask,
        return_dict=True,
    )

torch.cuda.synchronize()

inference_time = (
    time.perf_counter() - inference_start
)

pred_masks = outputs.pred_masks
pred_boxes = outputs.pred_boxes
pred_logits = outputs.pred_logits
presence_logits = outputs.presence_logits

for name, tensor in (
    ("pred_masks", pred_masks),
    ("pred_boxes", pred_boxes),
    ("pred_logits", pred_logits),
):
    if tensor is None:
        raise RuntimeError(
            f"{name} was not returned"
        )

    if not tensor_is_finite(tensor):
        raise RuntimeError(
            f"{name} contains NaN or Inf"
        )

scores = torch.sigmoid(
    pred_logits.float()
)

if presence_logits is not None:
    if not tensor_is_finite(presence_logits):
        raise RuntimeError(
            "presence_logits contains NaN or Inf"
        )

    presence_scores = torch.sigmoid(
        presence_logits.float()
    )

    scores = scores * presence_scores

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
    mask_logits[
        None,
        None,
        ...,
    ],
    size=(
        original_height,
        original_width,
    ),
    mode="bilinear",
    align_corners=False,
)[0, 0]

mask_probability = torch.sigmoid(
    resized_mask_logits
)

binary_mask = (
    mask_probability >= MASK_THRESHOLD
).cpu().numpy().astype(np.uint8)

mask_area_pixels = int(
    binary_mask.sum()
)

best_box = pred_boxes[
    0,
    best_query_index,
].float().cpu().tolist()

detected_queries = int(
    (
        scores_flat >= SCORE_THRESHOLD
    ).sum().item()
)

memory_after_inference = get_memory()

print("Complete inference succeeded")
print(
    f"Inference time: "
    f"{inference_time:.3f} s"
)
print(
    f"pred_masks: "
    f"{list(pred_masks.shape)}"
)
print(
    f"pred_boxes: "
    f"{list(pred_boxes.shape)}"
)
print(
    f"pred_logits: "
    f"{list(pred_logits.shape)}"
)
print(f"Best query: {best_query_index}")
print(f"Best score: {best_score:.6f}")
print(f"Best box: {best_box}")
print(f"Detected queries: {detected_queries}")
print(f"Mask pixels: {mask_area_pixels}")


# ------------------------------------------------------------------
# 8. Save images
# ------------------------------------------------------------------

print("\n[8/9] Saving mask and overlay...")

mask_image = Image.fromarray(
    binary_mask * 255,
    mode="L",
)

mask_image.save(OUTPUT_MASK)

overlay_image = create_overlay(
    image,
    binary_mask,
)

overlay_image.save(OUTPUT_OVERLAY)

print(f"Mask: {OUTPUT_MASK}")
print(f"Overlay: {OUTPUT_OVERLAY}")


# ------------------------------------------------------------------
# 9. Save JSON
# ------------------------------------------------------------------

print("\n[9/9] Saving test metadata...")

process_max_rss_kb = resource.getrusage(
    resource.RUSAGE_SELF
).ru_maxrss

result = {
    "test": (
        "Reduced SAM3 BF16 full segmentation"
    ),
    "success": True,
    "model": {
        "path": MODEL_PATH,
        "precision": "BF16",
        "integer_quantized": False,
        "gpu_parameters": (
            gpu_model_parameters
        ),
        "removed_placeholder_text_parameters": (
            placeholder_text_parameters
        ),
        "removed_placeholder_projection_parameters": (
            placeholder_projection_parameters
        ),
        "custom_resolution": IMAGE_SIZE,
    },
    "input": {
        "image": IMAGE_PATH,
        "original_size": [
            original_width,
            original_height,
        ],
        "pixel_values_shape": list(
            pixel_values.shape
        ),
        "prompt": text_prompt,
        "text_embeds_shape": list(
            text_embeds.shape
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
        "best_query_index": best_query_index,
        "best_score": best_score,
        "best_box": best_box,
        "detected_queries": detected_queries,
        "mask_area_pixels": mask_area_pixels,
        "mask_file": OUTPUT_MASK,
        "overlay_file": OUTPUT_OVERLAY,
    },
    "timing": {
        "cpu_model_load_s": model_load_time,
        "gpu_transfer_s": transfer_time,
        "inference_s": inference_time,
        "inference_ms": (
            inference_time * 1000
        ),
        "throughput_fps": (
            1.0 / inference_time
        ),
    },
    "memory": {
        "total_shared_gib": (
            memory_initial["total_gib"]
        ),
        "free_initial_gib": (
            memory_initial["free_gib"]
        ),
        "free_after_model_removal_gib": (
            memory_after_removal["free_gib"]
        ),
        "free_after_transfer_gib": (
            memory_after_transfer["free_gib"]
        ),
        "free_after_inference_gib": (
            memory_after_inference["free_gib"]
        ),
        "process_max_rss_mb": (
            process_max_rss_kb / 1024
        ),
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
    json.dump(
        result,
        file,
        indent=2,
    )

print("\n" + "=" * 72)
print("Reduced SAM3 segmentation succeeded")
print("=" * 72)
print(f"Prompt: {text_prompt!r}")
print(
    f"Inference: "
    f"{inference_time * 1000:.2f} ms"
)
print(
    f"Throughput: "
    f"{1.0 / inference_time:.3f} FPS"
)
print(f"Best score: {best_score:.6f}")
print(f"Mask pixels: {mask_area_pixels}")
print(
    f"Free after inference: "
    f"{memory_after_inference['free_gib']:.2f} GiB"
)
print(f"JSON: {OUTPUT_JSON}")
print(f"Mask: {OUTPUT_MASK}")
print(f"Overlay: {OUTPUT_OVERLAY}")