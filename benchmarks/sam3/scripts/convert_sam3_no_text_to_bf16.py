import gc
import json
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MODEL_DIRECTORY = "models/sam3_no_text_bf16"

INPUT_CHECKPOINT = os.path.join(
    MODEL_DIRECTORY,
    "model.safetensors",
)

TEMP_CHECKPOINT = os.path.join(
    MODEL_DIRECTORY,
    "model_bf16.safetensors.tmp",
)

METADATA_PATH = os.path.join(
    MODEL_DIRECTORY,
    "checkpoint_metadata.json",
)

EXPECTED_PARAMETER_COUNT = 506_197_688


print("=" * 72)
print("Converting reduced SAM3 checkpoint from FP32 to BF16")
print("=" * 72)

if not os.path.isfile(INPUT_CHECKPOINT):
    raise FileNotFoundError(INPUT_CHECKPOINT)

start_time = time.perf_counter()

converted_tensors = {}
tensor_count = 0
parameter_count = 0

floating_tensor_count = 0
integer_tensor_count = 0

floating_parameter_count = 0
integer_parameter_count = 0

source_dtypes = {}
output_dtypes = {}


# ---------------------------------------------------------------------
# 1. Read and convert tensors
# ---------------------------------------------------------------------

print("\n[1/4] Reading reduced checkpoint...")

with safe_open(
    INPUT_CHECKPOINT,
    framework="pt",
    device="cpu",
) as source:
    keys = list(source.keys())

    print(f"Tensor count: {len(keys)}")

    for index, key in enumerate(
        keys,
        start=1,
    ):
        tensor = source.get_tensor(key)

        tensor_count += 1
        parameter_count += tensor.numel()

        source_dtype = str(tensor.dtype)

        source_dtypes[source_dtype] = (
            source_dtypes.get(source_dtype, 0)
            + tensor.numel()
        )

        if tensor.is_floating_point():
            floating_tensor_count += 1
            floating_parameter_count += tensor.numel()

            if tensor.dtype != torch.bfloat16:
                tensor = tensor.to(
                    dtype=torch.bfloat16
                )
        else:
            integer_tensor_count += 1
            integer_parameter_count += tensor.numel()

        output_dtype = str(tensor.dtype)

        output_dtypes[output_dtype] = (
            output_dtypes.get(output_dtype, 0)
            + tensor.numel()
        )

        converted_tensors[key] = tensor

        if index % 200 == 0:
            print(
                f"  converted "
                f"{index}/{len(keys)} tensors"
            )


# ---------------------------------------------------------------------
# 2. Validate before saving
# ---------------------------------------------------------------------

print("\n[2/4] Validating BF16 conversion...")

if parameter_count != EXPECTED_PARAMETER_COUNT:
    raise RuntimeError(
        "Unexpected parameter count: "
        f"{parameter_count:,}; expected "
        f"{EXPECTED_PARAMETER_COUNT:,}"
    )

incorrect_floating_tensors = []

for key, tensor in converted_tensors.items():
    if (
        tensor.is_floating_point()
        and tensor.dtype != torch.bfloat16
    ):
        incorrect_floating_tensors.append(
            (key, str(tensor.dtype))
        )

if incorrect_floating_tensors:
    raise RuntimeError(
        "Some floating tensors are not BF16: "
        f"{incorrect_floating_tensors[:10]}"
    )

protected_scoring_keys = {
    "detector_model."
    "dot_product_scoring."
    "text_proj.weight",
    "detector_model."
    "dot_product_scoring."
    "text_proj.bias",
}

missing_protected_keys = (
    protected_scoring_keys
    - set(converted_tensors.keys())
)

if missing_protected_keys:
    raise RuntimeError(
        "Missing detector scoring weights: "
        f"{missing_protected_keys}"
    )

for key in protected_scoring_keys:
    tensor = converted_tensors[key]

    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{key} is not BF16"
        )

print(f"Parameters: {parameter_count:,}")
print(f"Floating tensors: {floating_tensor_count}")
print(f"Integer tensors: {integer_tensor_count}")
print(f"Source dtypes: {source_dtypes}")
print(f"Output dtypes: {output_dtypes}")


# ---------------------------------------------------------------------
# 3. Save temporary checkpoint and replace original
# ---------------------------------------------------------------------

print("\n[3/4] Saving BF16 checkpoint...")

if os.path.exists(TEMP_CHECKPOINT):
    os.remove(TEMP_CHECKPOINT)

save_file(
    converted_tensors,
    TEMP_CHECKPOINT,
    metadata={
        "format": "pt",
        "precision": "BF16",
        "description": (
            "SAM3 checkpoint without Text Encoder; "
            "all floating tensors converted to BF16"
        ),
    },
)

temporary_size = os.path.getsize(
    TEMP_CHECKPOINT
)

print(
    f"Temporary BF16 size: "
    f"{temporary_size / 1024**2:.2f} MiB"
)

# Reject an unexpectedly large output before replacing the source.
if temporary_size > 1_200 * 1024**2:
    raise RuntimeError(
        "Converted checkpoint is unexpectedly large: "
        f"{temporary_size / 1024**2:.2f} MiB"
    )

os.replace(
    TEMP_CHECKPOINT,
    INPUT_CHECKPOINT,
)

del converted_tensors
gc.collect()


# ---------------------------------------------------------------------
# 4. Verify saved checkpoint and update metadata
# ---------------------------------------------------------------------

print("\n[4/4] Verifying saved checkpoint...")

saved_tensor_count = 0
saved_parameter_count = 0
saved_dtypes = {}
invalid_saved_tensors = []

with safe_open(
    INPUT_CHECKPOINT,
    framework="pt",
    device="cpu",
) as saved:
    for key in saved.keys():
        tensor = saved.get_tensor(key)

        saved_tensor_count += 1
        saved_parameter_count += tensor.numel()

        dtype_name = str(tensor.dtype)

        saved_dtypes[dtype_name] = (
            saved_dtypes.get(dtype_name, 0)
            + tensor.numel()
        )

        if (
            tensor.is_floating_point()
            and tensor.dtype != torch.bfloat16
        ):
            invalid_saved_tensors.append(
                (key, dtype_name)
            )

        del tensor

if saved_parameter_count != EXPECTED_PARAMETER_COUNT:
    raise RuntimeError(
        "Saved parameter count mismatch: "
        f"{saved_parameter_count:,}"
    )

if invalid_saved_tensors:
    raise RuntimeError(
        "Saved checkpoint contains non-BF16 "
        "floating tensors: "
        f"{invalid_saved_tensors[:10]}"
    )

final_size = os.path.getsize(
    INPUT_CHECKPOINT
)

elapsed_time = time.perf_counter() - start_time

if os.path.isfile(METADATA_PATH):
    with open(
        METADATA_PATH,
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)
else:
    metadata = {}

metadata["precision_conversion"] = {
    "source_floating_precision": (
        "FP32 or original checkpoint dtype"
    ),
    "output_floating_precision": "BF16",
    "floating_tensor_count": (
        floating_tensor_count
    ),
    "integer_tensor_count": (
        integer_tensor_count
    ),
    "floating_parameter_count": (
        floating_parameter_count
    ),
    "integer_parameter_count": (
        integer_parameter_count
    ),
    "source_dtypes": source_dtypes,
    "output_dtypes": saved_dtypes,
    "verified": True,
}

metadata["checkpoint"] = {
    "size_bytes": final_size,
    "size_mib": final_size / 1024**2,
    "size_gib": final_size / 1024**3,
    "precision": "BF16",
}

metadata["bf16_conversion_time_s"] = (
    elapsed_time
)

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=2,
    )

print("\n" + "=" * 72)
print("BF16 checkpoint conversion succeeded")
print("=" * 72)
print(f"Tensors: {saved_tensor_count}")
print(f"Parameters: {saved_parameter_count:,}")
print(f"Saved dtypes: {saved_dtypes}")
print(
    f"Final size: "
    f"{final_size / 1024**2:.2f} MiB"
)
print(f"Conversion time: {elapsed_time:.2f} s")
print(f"Checkpoint: {INPUT_CHECKPOINT}")
print(f"Metadata: {METADATA_PATH}")