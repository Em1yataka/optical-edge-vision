# Edge Vision Model Benchmark Report

## Platform and scope

These measurements were collected on an **NVIDIA Jetson Orin** in
**15 W power mode**. They must not be presented as Jetson Nano results.

The repository contains independently produced YOLOv8n, SAM ViT-B,
SAM2 Hiera Tiny, YOLO+SAM2, and reduced SAM3 benchmark artifacts.
Confidential MicroFoundation source code and model files are excluded.

## Performance summary

| Model | Precision | Parameters | GFLOPs | Mean latency (ms) | Throughput (FPS) | Active VDD_IN (W) | Gross energy/inference (J) | Integer quantized |
|---|---|---|---|---|---|---|---|---|
| YOLOv8n | FP32 | 3,151,904 | 8.900 | 27.746 | 36.041 | 9.371 | 0.2355 | No |
| YOLOv8n | FP16 | 3,151,904 | 8.900 | 28.792 | 34.732 | 7.938 | 0.2099 | No |
| SAM ViT-B | Encoder FP16; prompt/decoder FP32 | 93,735,472 | Not measured | 1002.981 | 0.997 | 11.062 | 10.9223 | No |
| SAM2 Hiera Tiny | FP32 | 38,945,986 | Not measured | 635.708 | 1.573 | 11.243 | 7.0092 | No |
| SAM2 Hiera Tiny | BF16 autocast; FP32 weights | 38,945,986 | Not measured | 268.343 | 3.727 | 10.992 | 2.8263 | No |
| YOLOv8n FP32 + SAM2 Tiny BF16 | Mixed floating point | 42,097,890 combined | Not measured | 319.520 | 3.122 | 10.530 | 3.3875 | No |
| SAM3 reduced no-text runtime | BF16 weights and execution | 486,651,702 runtime (840,376,374 complete) | 615.778 | 1379.342 | 0.724 | 7.401 | 10.2259 | No |

`Not measured` means that no validated artifact is available; it does
not mean zero.

## Workloads

### YOLOv8n

YOLOv8n uses a 640-pixel input and batch size 1. Its wall-clock timing
includes preprocessing, inference, and postprocessing. Camera capture,
display, and model loading are excluded.

### SAM ViT-B

SAM ViT-B uses FP16 for the image encoder and FP32 for its embedding,
prompt encoder, and mask decoder. The reported timing includes image
encoding and prompt-based mask decoding.

### SAM2 Hiera Tiny

SAM2 was measured in FP32 and BF16 execution modes. BF16 autocast with
FP32 source weights is floating-point mixed precision, not integer
quantization. FP32/BF16 mask consistency is reported separately.

### YOLOv8n plus SAM2

The combined offline pipeline performs YOLO detection followed by
box-prompted SAM2 segmentation. Three independent repetitions quantify
cross-run stability.

### Reduced SAM3 BF16

The original SAM3 model contains 840,376,374 parameters. The reduced
GPU runtime contains 486,651,702 parameters. The text encoder runs
offline on CPU and its projected BF16 embedding is cached.

The deployed runtime uses 336 x 336 input, batch size 1, BF16 weights
and execution, and no integer quantization.

The measured 615.778 GFLOPs are an operator-counted estimate from
PyTorch `FlopCounterMode`. They cover the reduced GPU runtime and
exclude the offline text encoder. Profiler-instrumented latency is not
used as normal inference latency.

## Power and energy method

Power was sampled from the instantaneous `VDD_IN` value reported by
`tegrastats` at 100 ms intervals.

- Active power is mean board input power during inference.
- Idle power is measured in a separate idle window.
- Gross energy includes total board input energy.
- Incremental energy subtracts idle-equivalent energy.
- Camera capture, display, and model loading are excluded unless an
  individual result explicitly states otherwise.

Gross and incremental energy answer different questions and must not
be mixed in one comparison.

## Reproduction

Repository layout:

```text
benchmarks/
  yolov8/
  sam_vit_b/
  sam2_hiera_tiny/
  yolo_sam2_pipeline/
  sam3/
results/
  microscope/
docs/
  BENCHMARK_REPORT.md
```

Model checkpoints are intentionally excluded because they are large or
externally licensed. Obtain them from their official providers.

Some scripts and historical JSON records contain original absolute
paths under `/home/xia/jetson_projects`. Update configurable script
paths for a new machine, but preserve JSON result files as provenance.

The common protocol uses batch size 1, model warm-up, CUDA
synchronization, model loading outside measured latency, and separate
idle/active power collection. Refer to each JSON file for exact
exceptions.

## Microscope integration

Derived microscope segmentation results are included for integration
testing; raw biological images are excluded. A successful mask does
not establish biological accuracy. Proper validation requires
expert-reviewed annotations, multiple samples and fields of view, and
metrics such as IoU, Dice, precision, recall, and object-level error.

## Limitations

- Results apply to Jetson Orin, not Jetson Nano.
- SAM and SAM2 FLOPs remain unreported without validated profiles.
- SAM3 FLOPs exclude offline text encoding.
- `VDD_IN` is board-level rather than isolated GPU power.
- Unified memory complicates CUDA-memory and RSS interpretation.
- Microscope output has not been validated against biological ground
  truth.
- Results with different preprocessing pipelines are not necessarily
  directly comparable.

## Artifact checks

Collected artifacts were checked for Python syntax, JSON validity,
large files, model weights, common credential patterns, and private
MicroFoundation paths.
