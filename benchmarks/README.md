# Benchmark artifacts

This directory contains scripts and structured results collected on an
NVIDIA Jetson Orin in 15 W mode.

- `yolov8`: YOLOv8n FP32 and FP16
- `sam_vit_b`: SAM ViT-B mixed precision
- `sam2_hiera_tiny`: SAM2 FP32 and BF16
- `yolo_sam2_pipeline`: combined detection and segmentation
- `sam3`: reduced SAM3 BF16 deployment and profiling

Model weights are excluded. Obtain checkpoints from their official
providers and update local script paths.

See [the benchmark report](../docs/BENCHMARK_REPORT.md).
