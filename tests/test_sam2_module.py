import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import cv2

from segmentation.sam2_segmenter import SAM2Segmenter
image = cv2.imread("results/sam2_test_result.jpg")

segmenter = SAM2Segmenter(device="cpu")

h, w = image.shape[:2]

mask, score = segmenter.segment_from_point(
    image,
    (w // 2, h // 2),
)

print("Score:", score)
print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)