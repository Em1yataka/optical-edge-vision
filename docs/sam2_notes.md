# SAM2 Integration Notes

## Current Status

- SAM2 repository has been cloned and installed.
- Required dependencies have been installed.
- SAM2 tiny checkpoint has been downloaded.
- Single-image SAM2 inference works on CPU.
- Test result saved to `results/sam2_test_result.jpg`.

## Result

- Best segmentation score: 0.9079

## Current Issue

- GPU inference on Jetson Orin Nano currently fails with `CUBLAS_STATUS_ALLOC_FAILED`.
- This appears to be related to CUDA/cuBLAS memory allocation or PyTorch compatibility.

## Next Step

- Keep SAM2 as an experimental segmentation module.
- Investigate low-memory GPU inference or isolated environment.
- Integrate SAM2 into the Smart Camera pipeline after GPU inference is stable.