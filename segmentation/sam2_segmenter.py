import sys
from pathlib import Path

import cv2
import numpy as np
import torch


SAM2_DIR = Path.home() / "jetson_projects" / "sam2_test" / "sam2"
CHECKPOINT = SAM2_DIR / "checkpoints" / "sam2_hiera_tiny.pt"
CONFIG = "sam2_hiera_t.yaml"

sys.path.append(str(SAM2_DIR))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Segmenter:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = build_sam2(CONFIG, str(CHECKPOINT), device=device)
        self.predictor = SAM2ImagePredictor(self.model)

    def segment_from_point(self, frame, point):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        input_point = np.array([point])
        input_label = np.array([1])

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(bool)
        score = float(scores[best_idx])

        return mask, score

    def segment_from_box_center(self, frame, box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        return self.segment_from_point(frame, (cx, cy))