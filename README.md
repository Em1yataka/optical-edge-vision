
# Optical Edge Vision

### Edge AI Deployment, Object Detection and Foundation-Model Segmentation on NVIDIA Jetson Orin

**Research Project | University of California, Irvine | Summer 2026**

Optical Edge Vision is a research-oriented computer vision project exploring the deployment, benchmarking and integration of object detection and image segmentation models on resource-constrained NVIDIA Jetson hardware.

The project combines real-time camera processing, YOLO-based object detection, Segment Anything models, performance profiling and microscopy image analysis.

Its primary objective is to investigate the computational feasibility, latency, power consumption and deployment challenges of modern vision models at the edge.

## 1. Research Overview

Deploying advanced computer vision models on embedded platforms introduces challenges related to limited computational resources, software compatibility, memory usage and energy efficiency.

This project investigates these challenges through:

- Camera-based object detection using YOLOv8n
- Prompt-based image segmentation using SAM and SAM2
- Detection-guided segmentation using YOLOv8n + SAM2
- Reduced-runtime SAM3 benchmarking
- FP32, FP16 and BF16 execution comparisons
- Latency, throughput, power and energy measurements
- Microscopy segmentation integration experiments

The research was conducted during a summer research programme at the University of California, Irvine.

## 2. Hardware and Software

### Hardware
- NVIDIA Jetson Orin platform
- 15 W power mode for the reported benchmarks
- Camera-based image acquisition
- OpenUC2 microscopy integration

### Software
- Ubuntu Linux
- Python
- PyTorch
- CUDA
- OpenCV
- Ultralytics YOLOv8
- Segment Anything Model (SAM)
- Segment Anything 2 (SAM2)
- Reduced SAM3 runtime

**Note:** Benchmark results were collected on Jetson Orin hardware, not the original Jetson Nano. Performance should not be generalized to other Jetson configurations.

## 3. System Architecture

The repository includes a modular camera-processing pipeline:

```text
Camera Input
     |
     v
Frame Acquisition
     |
     v
YOLOv8 Object Detection
     |
     +----> Object Counting
     |
     +----> Image Quality Analysis
     |
     v
Visualisation and Annotation
     |
     v
FPS / Detection / Quality Logging
```

Additional offline experiments evaluate prompt-based segmentation and combined detection-segmentation workflows.

### Core Functionalities

**Object Detection**
- YOLOv8-based detection
- Object counting
- Frame annotation

**Image Analysis**
- Image-quality measurements
- Frame processing
- Performance monitoring

**Segmentation**
- SAM ViT-B
- SAM2 Hiera Tiny
- YOLO-assisted SAM2 segmentation
- Reduced SAM3 inference

**Performance Evaluation**
- Inference latency
- Throughput (FPS)
- Power consumption
- Energy per inference
- Execution precision comparisons

## 4. Benchmark Results

The following measurements were collected on an NVIDIA Jetson Orin operating in 15 W power mode.

| Model | Precision | Mean Latency (ms) | Throughput (FPS) |
|---|---|---:|---:|
| YOLOv8n | FP32 | 27.746 | 36.041 |
| YOLOv8n | FP16 | 28.792 | 34.732 |
| SAM ViT-B | Mixed FP16/FP32 | 1002.981 | 0.997 |
| SAM2 Hiera Tiny | FP32 | 635.708 | 1.573 |
| SAM2 Hiera Tiny | BF16 autocast | 268.343 | 3.727 |
| YOLOv8n + SAM2 Tiny | Mixed precision | 319.520 | 3.122 |
| Reduced SAM3 | BF16 | 1379.342 | 0.724 |

### Observations

- YOLOv8n achieved approximately 36 FPS under the measured detection workload.
- SAM2 Hiera Tiny showed a substantial latency reduction when using BF16 autocast compared with FP32.
- The YOLOv8n + SAM2 pipeline demonstrated the feasibility of detection-guided segmentation on embedded hardware.
- Reduced SAM3 remained computationally demanding on the tested platform.
- The results highlight the trade-offs between model complexity, inference latency and energy consumption.

These figures represent different model workloads and preprocessing configurations. They should not be interpreted as a strictly controlled, like-for-like comparison of segmentation accuracy or end-to-end camera performance.

For the full experimental methodology, power measurements, FLOP estimates and limitations, see:

**[Full Benchmark Report](docs/BENCHMARK_REPORT.md)**

**[Benchmark Artifacts](benchmarks/README.md)**

## 5. Microscopy Integration

The project also explored the application of edge-deployed segmentation models to microscopy image analysis.

This involved integrating image acquisition and processing components, investigating model deployment constraints and generating segmentation outputs for experimental evaluation.

Selected derived microscopy results are included in `results/microscope/`. Raw biological images and confidential external model resources are excluded.

These experiments demonstrate pipeline integration rather than clinically or biologically validated segmentation accuracy.

## 6. Repository Structure

```text
optical-edge-vision/
|
|-- analysis/         # Object counting, logging and image analysis
|-- benchmarks/       # Model benchmarks and structured results
|-- camera/           # Camera acquisition components
|-- detection/        # YOLO detection implementation
|-- docs/             # Experimental reports
|-- logs/             # Logging artifacts
|-- results/          # Experimental outputs
|-- src/              # Supporting source code
|-- utils/            # Visualisation utilities
|
|-- config.py         # Camera and detection configuration
|-- main.py           # Modular camera pipeline entry point
|-- yolo_camera.py    # Original YOLO camera implementation
|-- README.md
```

## 7. Getting Started

### Clone the Repository

```bash
git clone https://github.com/Em1yataka/optical-edge-vision.git
cd optical-edge-vision
```

### Configure the Application

The main camera pipeline uses settings in `config.py`:

```python
MODEL_PATH = "models/yolov8n.pt"
CAMERA_ID = 0
CONFIDENCE = 0.5
```

Set the model checkpoint path and camera device index to match your environment.

### Run the Camera Pipeline

After installing the appropriate PyTorch, OpenCV, Ultralytics and Jetson-compatible dependencies, and obtaining the required model checkpoint:

```bash
python3 main.py
```

The camera interface requires a supported video device and a graphical display environment. Headless systems may require modification of the OpenCV display functionality.

### Reproducing Benchmarks

Benchmark scripts and measurement artifacts are organised under `benchmarks/`.

Model checkpoints are intentionally excluded. Obtain them from their official providers and comply with their respective licences.

Some experimental scripts reference original local paths. Adapt these paths to your system before execution.

Exact software versions, model checkpoints and environment compatibility must be checked before attempting to reproduce the results. The repository is not currently distributed as a fully automated, one-command installation package.

## 8. Experimental Methodology

The reported benchmarks generally use:

- Batch size 1
- Model warm-up before measurement
- CUDA synchronisation for GPU timing
- Model loading outside the timed inference window
- Separate idle and active power measurements
- `tegrastats` for board-level power monitoring

Power values are based on `VDD_IN` board input measurements rather than isolated GPU power.

For model-specific methodology and exceptions, consult the detailed benchmark report and associated result files.

## 9. Research Challenges and Limitations

The work involved practical investigation of:

- Jetson ARM64 software compatibility
- PyTorch and CUDA execution
- Model precision and runtime performance
- Camera integration and visualisation
- Resource-constrained segmentation
- Measurement reproducibility

Current limitations include:

- Different model workloads are not directly comparable in every respect.
- Some checkpoints and external dependencies are not redistributed.
- Hardware-specific environment setup is required.
- Microscopy outputs lack expert-reviewed ground-truth validation.
- Power measurements represent board-level consumption.
- Some benchmark scripts require local path adjustments.

## 10. Research Context

**Researcher:** Dongchi Xu  
**Institution:** University of Southampton  
**Summer Research Host:** University of California, Irvine  
**Research Supervisor:** Professor Fei Xia  
**Research Period:** Summer 2026

This project reflects an interest in combining artificial intelligence, optical imaging and embedded computing, with a particular emphasis on translating computational models into practical hardware-based systems.

## 11. Documentation

- [Benchmark Report](docs/BENCHMARK_REPORT.md)
- [Benchmark Artifact Guide](benchmarks/README.md)

## 12. Disclaimer

This repository is intended for academic research and technical demonstration.

All reported results are specific to the experimental environment and should be interpreted alongside the documented measurement procedures and limitations.

External model weights, confidential repositories and raw biological image data are not included.
