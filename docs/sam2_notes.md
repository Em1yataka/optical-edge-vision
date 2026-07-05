# SAM2 Integration Notes

## Current Status

- SAM2 has been installed on the Jetson.
- SAM2 tiny checkpoint has been downloaded.
- Single-image CPU inference works successfully.
- Test result saved to `results/sam2_test_result.jpg`.

## Result

- SAM2 successfully segmented the prompted object.
- Best segmentation score: 0.9079.

## Current Issue

- GPU inference currently fails with `CUBLAS_STATUS_ALLOC_FAILED`.
- This is likely related to CUDA/cuBLAS memory allocation or PyTorch compatibility on Jetson Orin Nano.

## Next Step

- Integrate SAM2 as a segmentation module.
- Use YOLO bounding boxes as prompts for SAM2.
- Keep SAM2 integration separate from the stable Smart Camera demo until GPU inference is resolved.