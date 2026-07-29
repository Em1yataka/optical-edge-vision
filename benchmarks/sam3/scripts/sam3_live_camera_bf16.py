#!/usr/bin/env python3
"""
Live USB-camera demo for the reduced SAM3 BF16 runtime on Jetson Orin Nano.

The camera preview runs continuously in the main thread. A background worker
segments the most recent frame. Because SAM3 takes about 1.4 seconds per
inference on this Jetson, the preview can be smooth while the mask updates at
approximately 0.7 Hz.

Required local artifacts:
    models/sam3_no_text_bf16/
    models/facebook_sam3/
    sam3_person_text_cache.pt

Run with the same memory-related environment variables used by the successful
offline SAM3 deployment:

PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=-1 \
HF_ENABLE_PARALLEL_LOADING=false \
CUDA_LAUNCH_BLOCKING=0 \
python sam3_live_camera_bf16.py
"""

import argparse
import ctypes
import gc
import os
import queue
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Sam3Config, Sam3Model, Sam3Processor


EXPECTED_GPU_PARAMETERS = 486_651_702
LIBC = ctypes.CDLL("libc.so.6")


@dataclass
class SegmentationResult:
    frame_id: int
    mask: np.ndarray
    score: float
    inference_ms: float
    completed_at: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reduced SAM3 BF16 live USB-camera segmentation on Jetson"
    )
    parser.add_argument(
        "--camera",
        default="/dev/video0",
        help="V4L2 camera device, default: /dev/video0",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera capture width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera capture height",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Requested camera FPS",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=336,
        help="SAM3 square model input size; checkpoint was tested at 336",
    )
    parser.add_argument(
        "--model-path",
        default="models/sam3_no_text_bf16",
        help="Reduced no-text SAM3 model directory",
    )
    parser.add_argument(
        "--processor-path",
        default="models/facebook_sam3",
        help="Original SAM3 processor directory",
    )
    parser.add_argument(
        "--text-cache",
        default="sam3_person_text_cache.pt",
        help="Cached projected text features",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.50,
        help="Minimum selected query score",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.50,
        help="Mask probability threshold",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.42,
        help="Red mask-overlay opacity",
    )
    parser.add_argument(
        "--save-dir",
        default="live_results",
        help="Directory used when S is pressed",
    )
    return parser.parse_args()


def release_cpu_memory():
    gc.collect()
    LIBC.malloc_trim(0)


def get_memory():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "free_gib": free_bytes / 1024**3,
        "total_gib": total_bytes / 1024**3,
    }


def tensor_is_finite(tensor):
    return bool(torch.isfinite(tensor.float()).all())


def move_incrementally(module, device):
    """Move parameters one at a time to reduce peak unified-memory pressure."""
    moved_parameter_tensors = 0
    moved_buffer_tensors = 0
    moved_bytes = 0

    for submodule in module.modules():
        for name in list(submodule._parameters.keys()):
            parameter = submodule._parameters[name]
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
            moved_parameter_tensors += 1
            moved_bytes += (
                gpu_parameter.numel() * gpu_parameter.element_size()
            )
            del parameter
            del gpu_parameter

            if moved_parameter_tensors % 10 == 0:
                release_cpu_memory()

            if moved_parameter_tensors % 50 == 0:
                memory = get_memory()
                print(
                    f"  moved {moved_parameter_tensors} parameter tensors; "
                    f"{moved_bytes / 1024**2:.2f} MiB; "
                    f"{memory['free_gib']:.2f} GiB free"
                )

        for name in list(submodule._buffers.keys()):
            buffer = submodule._buffers[name]
            if buffer is None:
                continue
            if buffer.device.type == "meta":
                raise RuntimeError(f"Meta buffer cannot be moved: {name}")

            gpu_buffer = buffer.to(
                device=device,
                non_blocking=False,
            )
            submodule._buffers[name] = gpu_buffer
            moved_buffer_tensors += 1
            moved_bytes += gpu_buffer.numel() * gpu_buffer.element_size()
            del buffer
            del gpu_buffer

    release_cpu_memory()
    return {
        "parameter_tensors": moved_parameter_tensors,
        "buffer_tensors": moved_buffer_tensors,
        "bytes": moved_bytes,
        "mib": moved_bytes / 1024**2,
    }


def load_text_cache(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    required = {"text_prompt", "text_embeds", "attention_mask"}
    missing = required - set(cache.keys())
    if missing:
        raise RuntimeError(f"Missing text-cache keys: {sorted(missing)}")

    text_embeds = cache["text_embeds"]
    attention_mask = cache["attention_mask"]
    if list(text_embeds.shape) != [1, 32, 256]:
        raise RuntimeError(
            f"Unexpected text_embeds shape: {list(text_embeds.shape)}"
        )
    if list(attention_mask.shape) != [1, 32]:
        raise RuntimeError(
            f"Unexpected attention_mask shape: {list(attention_mask.shape)}"
        )
    if not tensor_is_finite(text_embeds):
        raise RuntimeError("Cached text embedding contains NaN or Inf")

    return (
        str(cache["text_prompt"]),
        text_embeds,
        attention_mask,
    )


def load_reduced_model(model_path, image_size):
    checkpoint = os.path.join(model_path, "model.safetensors")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    config = Sam3Config.from_pretrained(
        model_path,
        local_files_only=True,
    )
    config.image_size = image_size

    start = time.perf_counter()
    model, loading_info = Sam3Model.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        local_files_only=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    model.eval()

    missing_keys = loading_info.get("missing_keys", [])
    mismatched_keys = loading_info.get("mismatched_keys", [])
    allowed_missing_prefixes = (
        "text_encoder.",
        "text_projection.",
    )
    invalid_missing = [
        key
        for key in missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if invalid_missing:
        raise RuntimeError(
            f"Unexpected missing model weights: {invalid_missing[:20]}"
        )
    if mismatched_keys:
        raise RuntimeError(
            f"Mismatched model weights: {mismatched_keys[:20]}"
        )

    if model.text_encoder is None or model.text_projection is None:
        raise RuntimeError("Placeholder text modules were not constructed")

    model.text_encoder = None
    model.text_projection = None
    release_cpu_memory()

    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != EXPECTED_GPU_PARAMETERS:
        raise RuntimeError(
            f"Unexpected runtime parameter count: {parameter_count:,}; "
            f"expected {EXPECTED_GPU_PARAMETERS:,}"
        )

    print(
        f"CPU model prepared in {time.perf_counter() - start:.2f} s; "
        f"runtime parameters: {parameter_count:,}"
    )
    return model


def open_camera(device, width, height, fps):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {device}")

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Camera opened but returned no frame: {device}")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera: {device}; "
        f"{actual_width}x{actual_height}; "
        f"{actual_fps:.1f} FPS"
    )
    return cap, frame


def preprocess_frame(processor, frame_bgr, image_size):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    inputs = processor(
        images=image,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"]
    expected = [1, 3, image_size, image_size]
    if list(pixel_values.shape) != expected:
        raise RuntimeError(
            f"Unexpected camera tensor shape: {list(pixel_values.shape)}"
        )
    return pixel_values


def infer_one(
    model,
    processor,
    frame_bgr,
    frame_id,
    image_size,
    text_embeds,
    attention_mask,
    score_threshold,
    mask_threshold,
):
    original_height, original_width = frame_bgr.shape[:2]
    pixel_values_cpu = preprocess_frame(
        processor,
        frame_bgr,
        image_size,
    )
    pixel_values = pixel_values_cpu.to(
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    del pixel_values_cpu

    torch.cuda.synchronize()
    start = time.perf_counter()

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
    inference_ms = (time.perf_counter() - start) * 1000.0

    pred_masks = outputs.pred_masks
    pred_logits = outputs.pred_logits
    presence_logits = outputs.presence_logits

    if pred_masks is None or pred_logits is None:
        raise RuntimeError("SAM3 did not return masks and logits")
    if not tensor_is_finite(pred_masks):
        raise RuntimeError("pred_masks contains NaN or Inf")
    if not tensor_is_finite(pred_logits):
        raise RuntimeError("pred_logits contains NaN or Inf")

    scores = torch.sigmoid(pred_logits.float())
    if presence_logits is not None:
        if not tensor_is_finite(presence_logits):
            raise RuntimeError("presence_logits contains NaN or Inf")
        scores = scores * torch.sigmoid(presence_logits.float())

    scores_flat = scores.reshape(-1)
    best_flat_index = int(torch.argmax(scores_flat).item())
    query_count = int(pred_masks.shape[1])
    best_query = best_flat_index % query_count
    best_score = float(scores_flat[best_flat_index].item())

    if best_score < score_threshold:
        binary_mask = np.zeros(
            (original_height, original_width),
            dtype=np.uint8,
        )
    else:
        mask_logits = pred_masks[0, best_query].float()
        resized_logits = F.interpolate(
            mask_logits[None, None, ...],
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        binary_mask = (
            torch.sigmoid(resized_logits) >= mask_threshold
        ).cpu().numpy().astype(np.uint8)

    del outputs
    del pred_masks
    del pred_logits
    del presence_logits
    del scores
    del scores_flat
    del pixel_values

    return SegmentationResult(
        frame_id=frame_id,
        mask=binary_mask,
        score=best_score,
        inference_ms=inference_ms,
        completed_at=time.time(),
    )


def apply_overlay(frame, mask, alpha):
    if mask is None or mask.shape != frame.shape[:2]:
        return frame.copy()

    output = frame.copy()
    active = mask.astype(bool)
    if not np.any(active):
        return output

    red_bgr = np.zeros_like(output)
    red_bgr[..., 2] = 255
    output[active] = cv2.addWeighted(
        output[active],
        1.0 - alpha,
        red_bgr[active],
        alpha,
        0.0,
    )

    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(output, contours, -1, (0, 255, 255), 1)
    return output


class InferenceWorker(threading.Thread):
    def __init__(
        self,
        model,
        processor,
        image_size,
        text_embeds,
        attention_mask,
        score_threshold,
        mask_threshold,
        input_queue,
        output_queue,
        stop_event,
    ):
        super().__init__(daemon=True)
        self.model = model
        self.processor = processor
        self.image_size = image_size
        self.text_embeds = text_embeds
        self.attention_mask = attention_mask
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.error: Optional[BaseException] = None

    def run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    frame_id, frame = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Discard older queued frames and always process the newest.
                while True:
                    try:
                        frame_id, frame = self.input_queue.get_nowait()
                    except queue.Empty:
                        break

                result = infer_one(
                    model=self.model,
                    processor=self.processor,
                    frame_bgr=frame,
                    frame_id=frame_id,
                    image_size=self.image_size,
                    text_embeds=self.text_embeds,
                    attention_mask=self.attention_mask,
                    score_threshold=self.score_threshold,
                    mask_threshold=self.mask_threshold,
                )

                while True:
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break
                self.output_queue.put_nowait(result)
        except BaseException as error:
            self.error = error
            self.stop_event.set()


def put_latest(input_queue, item):
    while True:
        try:
            input_queue.get_nowait()
        except queue.Empty:
            break
    try:
        input_queue.put_nowait(item)
    except queue.Full:
        pass


def main():
    args = parse_args()

    print("=" * 76)
    print("SAM3 reduced BF16 live USB-camera segmentation")
    print("=" * 76)
    print("Keys: Q or ESC = quit; S = save current frame/mask/overlay")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.image_size != 336:
        print(
            "WARNING: this runtime was validated at 336 x 336; "
            f"requested size is {args.image_size}."
        )

    torch.cuda.init()
    torch.cuda.empty_cache()
    memory = get_memory()
    print(
        f"Shared memory before loading: "
        f"{memory['free_gib']:.2f} GiB free / "
        f"{memory['total_gib']:.2f} GiB total"
    )
    if memory["free_gib"] < 4.0:
        print(
            "WARNING: less than 4 GiB is free. Close other applications "
            "or reboot before retrying if CUDA reports out of memory."
        )

    text_prompt, text_embeds_cpu, attention_mask_cpu = load_text_cache(
        args.text_cache
    )
    print(f"Cached prompt: {text_prompt!r}")

    processor = Sam3Processor.from_pretrained(
        args.processor_path,
        size={
            "height": args.image_size,
            "width": args.image_size,
        },
        local_files_only=True,
    )

    model = load_reduced_model(
        args.model_path,
        args.image_size,
    )

    print("Moving reduced runtime to CUDA incrementally...")
    transfer_start = time.perf_counter()
    transfer = move_incrementally(
        model,
        torch.device("cuda:0"),
    )
    torch.cuda.synchronize()
    print(
        f"GPU transfer complete: {transfer['mib']:.2f} MiB in "
        f"{time.perf_counter() - transfer_start:.2f} s"
    )

    devices = sorted({str(p.device) for p in model.parameters()})
    if devices != ["cuda:0"]:
        raise RuntimeError(f"Model parameters are on mixed devices: {devices}")

    text_embeds = text_embeds_cpu.to(
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    attention_mask = attention_mask_cpu.to(
        device="cuda:0",
        dtype=torch.int64,
    )
    del text_embeds_cpu
    del attention_mask_cpu
    release_cpu_memory()

    cap, first_frame = open_camera(
        args.camera,
        args.width,
        args.height,
        args.fps,
    )

    input_queue = queue.Queue(maxsize=1)
    output_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    worker = InferenceWorker(
        model=model,
        processor=processor,
        image_size=args.image_size,
        text_embeds=text_embeds,
        attention_mask=attention_mask,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        input_queue=input_queue,
        output_queue=output_queue,
        stop_event=stop_event,
    )
    worker.start()

    frame_id = 0
    latest_result: Optional[SegmentationResult] = None
    last_frame = first_frame
    last_preview_time = time.perf_counter()
    preview_counter = 0
    preview_fps = 0.0
    save_index = 0
    os.makedirs(args.save_dir, exist_ok=True)

    put_latest(input_queue, (frame_id, first_frame.copy()))

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                print("WARNING: camera frame read failed")
                time.sleep(0.02)
                continue

            frame_id += 1
            last_frame = frame
            put_latest(input_queue, (frame_id, frame.copy()))

            while True:
                try:
                    latest_result = output_queue.get_nowait()
                except queue.Empty:
                    break

            preview_counter += 1
            now = time.perf_counter()
            elapsed = now - last_preview_time
            if elapsed >= 1.0:
                preview_fps = preview_counter / elapsed
                preview_counter = 0
                last_preview_time = now

            mask = latest_result.mask if latest_result else None
            display = apply_overlay(frame, mask, args.alpha)

            status = (
                f"Preview {preview_fps:.1f} FPS | "
                f"Prompt: {text_prompt}"
            )
            cv2.putText(
                display,
                status,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if latest_result:
                age = time.time() - latest_result.completed_at
                info = (
                    f"SAM3 {latest_result.inference_ms:.0f} ms | "
                    f"score {latest_result.score:.3f} | "
                    f"mask age {age:.1f} s"
                )
            else:
                info = "SAM3: waiting for first mask..."

            cv2.putText(
                display,
                info,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "Q/ESC quit | S save",
                (10, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("SAM3 Live Camera - Reduced BF16", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                stop_event.set()
                break

            if key in (ord("s"), ord("S")):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                prefix = os.path.join(
                    args.save_dir,
                    f"sam3_live_{timestamp}_{save_index:03d}",
                )
                save_index += 1
                cv2.imwrite(prefix + "_frame.jpg", last_frame)
                cv2.imwrite(prefix + "_overlay.jpg", display)
                if latest_result is not None:
                    cv2.imwrite(
                        prefix + "_mask.png",
                        latest_result.mask * 255,
                    )
                print(f"Saved: {prefix}_*")

        if worker.error is not None:
            raise RuntimeError("SAM3 worker failed") from worker.error

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        worker.join(timeout=3.0)

    print("Live SAM3 camera session ended cleanly")


if __name__ == "__main__":
    main()
