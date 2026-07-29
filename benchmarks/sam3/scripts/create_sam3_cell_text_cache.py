import json
import time
from pathlib import Path

import torch
from transformers import Sam3Model, Sam3Processor


MODEL_PATH = Path("models/facebook_sam3")
TEXT_PROMPT = "cell"

CACHE_PATH = Path("sam3_cell_text_cache.pt")
METADATA_PATH = Path("sam3_cell_text_cache.json")


def main():
    print("=" * 72)
    print("SAM3 BF16 cell text-feature cache")
    print("=" * 72)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SAM3 model directory not found: {MODEL_PATH}"
        )

    print(f"Model: {MODEL_PATH}")
    print(f"Prompt: {TEXT_PROMPT!r}")

    print("\n[1/3] Loading processor...")

    processor = Sam3Processor.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    print("\n[2/3] Loading complete SAM3 model on CPU...")

    load_start = time.perf_counter()

    model = Sam3Model.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model.eval()

    load_time = time.perf_counter() - load_start

    print(f"Model loaded in {load_time:.3f} seconds")

    print(f"\n[3/3] Encoding prompt: {TEXT_PROMPT!r}")

    text_inputs = processor(
        text=TEXT_PROMPT,
        return_tensors="pt",
    )

    input_ids = text_inputs["input_ids"].to("cpu")
    attention_mask = text_inputs["attention_mask"].to("cpu")

    encode_start = time.perf_counter()

    with torch.inference_mode():
        text_output = model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_hidden_state = text_output.last_hidden_state

        text_embeds = model.text_projection(
            text_hidden_state
        )

    encode_time = time.perf_counter() - encode_start

    text_embeds = (
        text_embeds.detach()
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
    )

    attention_mask = (
        attention_mask.detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous()
    )

    if not torch.isfinite(text_embeds.float()).all():
        raise RuntimeError(
            "Text embedding contains NaN or infinite values"
        )

    cache = {
        "text_prompt": TEXT_PROMPT,
        "text_embeds": text_embeds,
        "attention_mask": attention_mask,
    }

    torch.save(cache, CACHE_PATH)

    metadata = {
        "model_path": str(MODEL_PATH.resolve()),
        "text_prompt": TEXT_PROMPT,
        "model_precision": "BF16",
        "text_embedding_shape": list(text_embeds.shape),
        "text_embedding_dtype": str(text_embeds.dtype),
        "attention_mask_shape": list(attention_mask.shape),
        "attention_mask_dtype": str(attention_mask.dtype),
        "model_load_time_s": load_time,
        "text_encoding_time_s": encode_time,
        "cache_path": str(CACHE_PATH.resolve()),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 72)
    print("Cell text-feature cache created successfully")
    print("=" * 72)
    print(f"Text embedding: {list(text_embeds.shape)}")
    print(f"Embedding dtype: {text_embeds.dtype}")
    print(f"Attention mask: {list(attention_mask.shape)}")
    print(f"Encoding time: {encode_time:.3f} seconds")
    print(f"Saved cache: {CACHE_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()