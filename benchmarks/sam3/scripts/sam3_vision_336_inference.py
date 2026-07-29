import gc
import json
import os
import resource
import time

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import Sam3Config, Sam3VisionModel

MODEL_PATH = "models/facebook_sam3"
VISION_CHECKPOINT = "models/sam3_vision_bf16.safetensors"
IMAGE_PATH = "/home/xia/jetson_projects/sam2_test/test_image.jpg"
OUTPUT_PATH = "sam3_vision_bf16_336_inference_test.json"

EXPECTED_PARAMETERS = 454_038_784
IMAGE_SIZE = 336

# Standard CLIP/SAM3 image normalization values.
IMAGE_MEAN = np.array(
    [0.48145466, 0.4578275, 0.40821073],
    dtype=np.float32,
)

IMAGE_STD = np.array(
    [0.26862954, 0.26130258, 0.27577711],
    dtype=np.float32,
)


def cuda_memory():
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    return {
        "free_gib": free_bytes / 1024**3,
        "total_gib": total_bytes / 1024**3,
    }


def move_incrementally(module, device):
    moved_parameters = 0
    moved_buffers = 0
    moved_bytes = 0

    for submodule in module.modules():
        for name, parameter in list(
            submodule._parameters.items()
        ):
            if parameter is None:
                continue

            gpu_parameter = parameter.detach().to(
                device=device,
                non_blocking=False,
            )

            submodule._parameters[name] = torch.nn.Parameter(
                gpu_parameter,
                requires_grad=False,
            )

            moved_parameters += 1
            moved_bytes += (
                gpu_parameter.numel()
                * gpu_parameter.element_size()
            )

            del parameter
            del gpu_parameter

            if moved_parameters % 50 == 0:
                free_bytes, _ = torch.cuda.mem_get_info()

                print(
                    f"  moved {moved_parameters} parameters; "
                    f"{moved_bytes / 1024**2:.2f} MiB; "
                    f"{free_bytes / 1024**3:.2f} GiB free"
                )

        for name, buffer in list(
            submodule._buffers.items()
        ):
            if buffer is None:
                continue

            gpu_buffer = buffer.to(
                device=device,
                non_blocking=False,
            )

            submodule._buffers[name] = gpu_buffer

            moved_buffers += 1
            moved_bytes += (
                gpu_buffer.numel()
                * gpu_buffer.element_size()
            )

            del buffer
            del gpu_buffer

    return moved_parameters, moved_buffers, moved_bytes


def prepare_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")

    original_size = {
        "width": image.width,
        "height": image.height,
    }

    image = image.resize(
        (image_size, image_size),
        Image.Resampling.BICUBIC,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGE_MEAN) / IMAGE_STD
    array = np.transpose(array, (2, 0, 1))
    array = np.ascontiguousarray(array)

    pixel_values = torch.from_numpy(array).unsqueeze(0)

    return pixel_values, original_size


print("=" * 72)
print("SAM3 standalone Vision Encoder BF16 — 336px inference")
print("=" * 72)

if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(IMAGE_PATH)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("\n[1/7] Initializing CUDA...")

torch.cuda.init()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

memory_initial = cuda_memory()

print(
    f"Initial shared memory: "
    f"{memory_initial['free_gib']:.2f} GiB free / "
    f"{memory_initial['total_gib']:.2f} GiB total"
)

print("\n[2/7] Preparing 336x336 image...")

pixel_values_cpu, original_size = prepare_image(
    IMAGE_PATH,
    IMAGE_SIZE,
)

print(f"Original image size: {original_size}")
print(f"Input tensor shape: {list(pixel_values_cpu.shape)}")
print(f"Input tensor dtype: {pixel_values_cpu.dtype}")
print(
    f"Input range: "
    f"{pixel_values_cpu.min().item():.4f} to "
    f"{pixel_values_cpu.max().item():.4f}"
)

print("\n[3/7] Creating standalone BF16 vision model...")

full_config = Sam3Config.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

vision_config = full_config.vision_config

if hasattr(vision_config, "backbone_config"):
    vision_config.backbone_config.image_size = IMAGE_SIZE

original_default_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)

try:
    vision_model = Sam3VisionModel(vision_config)
finally:
    torch.set_default_dtype(original_default_dtype)

vision_model.eval()

print("CPU vision model created")

print("\n[4/7] Loading extracted vision checkpoint...")

state_dict = load_file(
    VISION_CHECKPOINT,
    device="cpu",
)

load_result = vision_model.load_state_dict(
    state_dict,
    strict=True,
    assign=True,
)

missing_keys = list(load_result.missing_keys)
unexpected_keys = list(load_result.unexpected_keys)

if missing_keys or unexpected_keys:
    raise RuntimeError(
        f"Checkpoint mismatch: missing={missing_keys[:5]}, "
        f"unexpected={unexpected_keys[:5]}"
    )

parameter_count = sum(
    parameter.numel()
    for parameter in vision_model.parameters()
)

if parameter_count != EXPECTED_PARAMETERS:
    raise RuntimeError(
        f"Parameter mismatch: {parameter_count:,}"
    )

del state_dict
gc.collect()

print(f"Loaded parameters: {parameter_count:,}")

print("\n[5/7] Moving vision model incrementally to GPU...")

transfer_start = time.perf_counter()

moved_parameters, moved_buffers, moved_bytes = (
    move_incrementally(
        vision_model,
        torch.device("cuda:0"),
    )
)

torch.cuda.synchronize()

transfer_time = time.perf_counter() - transfer_start
memory_after_transfer = cuda_memory()

print(f"GPU transfer time: {transfer_time:.3f} s")
print(
    f"Free after model transfer: "
    f"{memory_after_transfer['free_gib']:.2f} GiB"
)

print("\n[6/7] Running one real image-encoder inference...")

pixel_values = pixel_values_cpu.to(
    device="cuda:0",
    dtype=torch.bfloat16,
)

del pixel_values_cpu
gc.collect()

torch.cuda.synchronize()
inference_start = time.perf_counter()

with torch.inference_mode():
    output = vision_model(
        pixel_values=pixel_values,
        return_dict=True,
    )

torch.cuda.synchronize()
inference_time = time.perf_counter() - inference_start

if hasattr(output, "last_hidden_state"):
    output_tensor = output.last_hidden_state
elif isinstance(output, (tuple, list)):
    output_tensor = output[0]
else:
    raise RuntimeError(
        f"Unknown output type: {type(output)}"
    )

if not torch.isfinite(output_tensor.float()).all():
    raise RuntimeError(
        "Vision output contains NaN or Inf"
    )

memory_after_inference = cuda_memory()
peak_cuda_bytes = torch.cuda.max_memory_allocated()

output_shape = list(output_tensor.shape)
output_dtype = str(output_tensor.dtype)
output_mean = output_tensor.float().mean().item()
output_std = output_tensor.float().std().item()
output_min = output_tensor.float().min().item()
output_max = output_tensor.float().max().item()

print("Real vision inference succeeded")
print(f"Inference time: {inference_time:.3f} s")
print(f"Output shape: {output_shape}")
print(f"Output dtype: {output_dtype}")
print(
    f"Output statistics: mean={output_mean:.6f}, "
    f"std={output_std:.6f}, "
    f"min={output_min:.6f}, "
    f"max={output_max:.6f}"
)

print("\n[7/7] Saving results...")

max_rss_kb = resource.getrusage(
    resource.RUSAGE_SELF
).ru_maxrss

result = {
    "test": "SAM3 standalone vision BF16 336px inference",
    "success": True,
    "model": {
        "name": "SAM3 Vision Encoder",
        "checkpoint": VISION_CHECKPOINT,
        "parameters": parameter_count,
        "precision": "BF16",
        "integer_quantized": False,
    },
    "input": {
        "image_path": IMAGE_PATH,
        "original_size": original_size,
        "model_input_shape": list(pixel_values.shape),
        "model_input_dtype": str(pixel_values.dtype),
        "image_size": IMAGE_SIZE,
        "batch_size": 1,
    },
    "output": {
        "shape": output_shape,
        "dtype": output_dtype,
        "finite": True,
        "mean": output_mean,
        "std": output_std,
        "min": output_min,
        "max": output_max,
    },
    "timing": {
        "gpu_transfer_s": transfer_time,
        "vision_inference_s": inference_time,
        "vision_inference_ms": inference_time * 1000,
        "vision_fps": 1.0 / inference_time,
    },
    "memory": {
        "shared_total_gib": memory_initial["total_gib"],
        "shared_free_initial_gib": memory_initial["free_gib"],
        "shared_free_after_transfer_gib": (
            memory_after_transfer["free_gib"]
        ),
        "shared_free_after_inference_gib": (
            memory_after_inference["free_gib"]
        ),
        "cuda_peak_allocated_mb": (
            peak_cuda_bytes / 1024**2
        ),
        "process_max_rss_mb": max_rss_kb / 1024,
    },
    "transfer": {
        "parameter_tensors": moved_parameters,
        "buffer_tensors": moved_buffers,
        "total_mib": moved_bytes / 1024**2,
    },
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

with open(OUTPUT_PATH, "w") as file:
    json.dump(result, file, indent=2)

print("\n" + "=" * 72)
print("SAM3 Vision Encoder inference test succeeded")
print("=" * 72)
print(f"Inference time: {inference_time * 1000:.2f} ms")
print(f"Throughput: {1.0 / inference_time:.3f} FPS")
print(f"Output shape: {output_shape}")
print(
    f"CUDA peak allocated: "
    f"{peak_cuda_bytes / 1024**2:.2f} MiB"
)
print(
    f"Free shared memory after inference: "
    f"{memory_after_inference['free_gib']:.2f} GiB"
)
print(f"Saved result: {OUTPUT_PATH}")