import gc
import json
import os
import shutil
import time

from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Sam3Config


SOURCE_MODEL_PATH = "models/facebook_sam3"
SOURCE_CHECKPOINT = os.path.join(
    SOURCE_MODEL_PATH,
    "model.safetensors",
)

OUTPUT_MODEL_PATH = "models/sam3_no_text_bf16"

OUTPUT_CHECKPOINT = os.path.join(
    OUTPUT_MODEL_PATH,
    "model.safetensors",
)

TEMP_CHECKPOINT = os.path.join(
    OUTPUT_MODEL_PATH,
    "model.safetensors.tmp",
)

METADATA_PATH = os.path.join(
    OUTPUT_MODEL_PATH,
    "checkpoint_metadata.json",
)

# These are the exact prefixes found in the original Meta checkpoint.
EXCLUDED_PREFIXES = (
    "detector_model.text_encoder.",
    "detector_model.text_projection.",
)

# Do not remove:
# detector_model.dot_product_scoring.text_proj.*
# It is part of the detector scoring module.

EXPECTED_EXCLUDED_PARAMETERS = 353_724_672

FILES_TO_COPY = (
    "processor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)


def matches_excluded_prefix(key):
    return any(
        key.startswith(prefix)
        for prefix in EXCLUDED_PREFIXES
    )


print("=" * 72)
print("Creating SAM3 checkpoint without Text Encoder")
print("=" * 72)

if not os.path.isfile(SOURCE_CHECKPOINT):
    raise FileNotFoundError(
        SOURCE_CHECKPOINT
    )

os.makedirs(
    OUTPUT_MODEL_PATH,
    exist_ok=True,
)

start_time = time.perf_counter()


# ---------------------------------------------------------------------
# 1. Inspect and filter checkpoint
# ---------------------------------------------------------------------

print("\n[1/5] Reading original SAM3 checkpoint...")

kept_tensors = {}

original_tensor_count = 0
original_parameter_count = 0

kept_tensor_count = 0
kept_parameter_count = 0

excluded_tensor_count = 0
excluded_parameter_count = 0

excluded_keys = []
kept_text_scoring_keys = []

with safe_open(
    SOURCE_CHECKPOINT,
    framework="pt",
    device="cpu",
) as source:
    checkpoint_keys = list(source.keys())
    original_tensor_count = len(checkpoint_keys)

    print(
        f"Original tensor count: "
        f"{original_tensor_count}"
    )

    for index, key in enumerate(
        checkpoint_keys,
        start=1,
    ):
        tensor = source.get_tensor(key)
        parameter_count = tensor.numel()

        original_parameter_count += parameter_count

        if matches_excluded_prefix(key):
            excluded_tensor_count += 1
            excluded_parameter_count += parameter_count
            excluded_keys.append(key)

            del tensor
        else:
            kept_tensors[key] = tensor
            kept_tensor_count += 1
            kept_parameter_count += parameter_count

            if key.startswith(
                "detector_model."
                "dot_product_scoring."
                "text_proj."
            ):
                kept_text_scoring_keys.append(key)

        if index % 200 == 0:
            print(
                f"  processed "
                f"{index}/{original_tensor_count} tensors"
            )


# ---------------------------------------------------------------------
# 2. Validate filtering before writing output
# ---------------------------------------------------------------------

print("\n[2/5] Validating filtered checkpoint...")

if excluded_tensor_count == 0:
    raise RuntimeError(
        "No Text Encoder tensors were excluded. "
        "Checkpoint prefixes are incorrect."
    )

if (
    excluded_parameter_count
    != EXPECTED_EXCLUDED_PARAMETERS
):
    raise RuntimeError(
        "Unexpected excluded parameter count: "
        f"{excluded_parameter_count:,}; expected "
        f"{EXPECTED_EXCLUDED_PARAMETERS:,}"
    )

if (
    original_parameter_count
    != kept_parameter_count
    + excluded_parameter_count
):
    raise RuntimeError(
        "Parameter accounting mismatch"
    )

if (
    original_tensor_count
    != kept_tensor_count
    + excluded_tensor_count
):
    raise RuntimeError(
        "Tensor accounting mismatch"
    )

remaining_excluded_keys = [
    key
    for key in kept_tensors
    if matches_excluded_prefix(key)
]

if remaining_excluded_keys:
    raise RuntimeError(
        "Some excluded tensors remain: "
        f"{remaining_excluded_keys[:5]}"
    )

expected_scoring_keys = {
    "detector_model."
    "dot_product_scoring."
    "text_proj.weight",
    "detector_model."
    "dot_product_scoring."
    "text_proj.bias",
}

if set(kept_text_scoring_keys) != expected_scoring_keys:
    raise RuntimeError(
        "Detector text-scoring weights were not "
        "preserved correctly: "
        f"{kept_text_scoring_keys}"
    )

print(
    f"Excluded tensors: "
    f"{excluded_tensor_count}"
)
print(
    f"Excluded parameters: "
    f"{excluded_parameter_count:,}"
)
print(
    f"Kept tensors: "
    f"{kept_tensor_count}"
)
print(
    f"Kept parameters: "
    f"{kept_parameter_count:,}"
)
print(
    "Preserved detector scoring keys:"
)

for key in kept_text_scoring_keys:
    print(f"  {key}")


# ---------------------------------------------------------------------
# 3. Save reduced checkpoint safely
# ---------------------------------------------------------------------

print("\n[3/5] Saving reduced checkpoint...")

if os.path.exists(TEMP_CHECKPOINT):
    os.remove(TEMP_CHECKPOINT)

save_file(
    kept_tensors,
    TEMP_CHECKPOINT,
    metadata={
        "format": "pt",
        "description": (
            "SAM3 BF16 checkpoint without "
            "detector_model.text_encoder and "
            "detector_model.text_projection"
        ),
    },
)

# Atomically replace the previous invalid checkpoint.
os.replace(
    TEMP_CHECKPOINT,
    OUTPUT_CHECKPOINT,
)

checkpoint_size_bytes = os.path.getsize(
    OUTPUT_CHECKPOINT
)

print(
    f"Saved checkpoint size: "
    f"{checkpoint_size_bytes / 1024**2:.2f} MiB"
)

del kept_tensors
gc.collect()


# ---------------------------------------------------------------------
# 4. Create reduced-memory config
# ---------------------------------------------------------------------

print("\n[4/5] Creating reduced-memory configuration...")

config = Sam3Config.from_pretrained(
    SOURCE_MODEL_PATH,
    local_files_only=True,
)

# Official custom-resolution setting.
config.image_size = 336

# The Text Encoder will not be used because cached text_embeds
# are passed directly to model.forward().
#
# A tiny placeholder configuration prevents Transformers from
# constructing a 353M-parameter text model for missing weights.
text_config = config.text_config

text_config.hidden_size = 32
text_config.intermediate_size = 64
text_config.num_hidden_layers = 1
text_config.num_attention_heads = 4
text_config.max_position_embeddings = 32

if hasattr(text_config, "projection_dim"):
    text_config.projection_dim = 256

config.save_pretrained(
    OUTPUT_MODEL_PATH,
)

print(
    "Saved reduced-memory configuration: "
    f"{OUTPUT_MODEL_PATH}/config.json"
)


# ---------------------------------------------------------------------
# 5. Copy processor and tokenizer configuration
# ---------------------------------------------------------------------

print("\n[5/5] Copying processor/tokenizer files...")

copied_files = []

for filename in FILES_TO_COPY:
    source_path = os.path.join(
        SOURCE_MODEL_PATH,
        filename,
    )

    destination_path = os.path.join(
        OUTPUT_MODEL_PATH,
        filename,
    )

    if not os.path.isfile(source_path):
        print(f"  skipped missing file: {filename}")
        continue

    shutil.copy2(
        source_path,
        destination_path,
    )

    copied_files.append(filename)
    print(f"  copied: {filename}")


# ---------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------

elapsed_time = time.perf_counter() - start_time

metadata = {
    "source_model": SOURCE_MODEL_PATH,
    "source_checkpoint": SOURCE_CHECKPOINT,
    "output_checkpoint": OUTPUT_CHECKPOINT,
    "excluded_prefixes": list(
        EXCLUDED_PREFIXES
    ),
    "protected_prefix": (
        "detector_model."
        "dot_product_scoring."
        "text_proj."
    ),
    "original": {
        "tensor_count": original_tensor_count,
        "parameter_count": (
            original_parameter_count
        ),
    },
    "excluded": {
        "tensor_count": excluded_tensor_count,
        "parameter_count": (
            excluded_parameter_count
        ),
        "expected_parameter_count": (
            EXPECTED_EXCLUDED_PARAMETERS
        ),
        "first_keys": excluded_keys[:10],
    },
    "kept": {
        "tensor_count": kept_tensor_count,
        "parameter_count": kept_parameter_count,
    },
    "preserved_detector_scoring_keys": (
        kept_text_scoring_keys
    ),
    "checkpoint": {
        "size_bytes": checkpoint_size_bytes,
        "size_mib": (
            checkpoint_size_bytes / 1024**2
        ),
        "size_gib": (
            checkpoint_size_bytes / 1024**3
        ),
    },
    "configuration": {
        "custom_image_size": 336,
        "text_encoder_placeholder": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "max_position_embeddings": 32,
            "projection_dim": getattr(
                text_config,
                "projection_dim",
                None,
            ),
        },
    },
    "copied_files": copied_files,
    "creation_time_s": elapsed_time,
}

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


# ---------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------

print("\n" + "=" * 72)
print("Reduced SAM3 checkpoint created successfully")
print("=" * 72)
print(
    f"Original parameters: "
    f"{original_parameter_count:,}"
)
print(
    f"Excluded parameters: "
    f"{excluded_parameter_count:,}"
)
print(
    f"Kept parameters: "
    f"{kept_parameter_count:,}"
)
print(
    f"Excluded tensors: "
    f"{excluded_tensor_count}"
)
print(
    f"Kept tensors: "
    f"{kept_tensor_count}"
)
print(
    f"Checkpoint size: "
    f"{checkpoint_size_bytes / 1024**2:.2f} MiB"
)
print(
    f"Creation time: "
    f"{elapsed_time:.2f} s"
)
print(f"Output directory: {OUTPUT_MODEL_PATH}")
print(f"Checkpoint: {OUTPUT_CHECKPOINT}")
print(f"Metadata: {METADATA_PATH}")