import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


CONFIG = "sam2_hiera_t.yaml"
CHECKPOINT = "checkpoints/sam2_hiera_tiny.pt"
IMAGE_PATH = "../test_image.jpg"
OUTPUT_DIR = Path("benchmark_results")

OUTPUT_DIR.mkdir(exist_ok=True)

image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    raise FileNotFoundError(IMAGE_PATH)

image_bgr = cv2.resize(image_bgr, (512, 512))
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

h, w = image_rgb.shape[:2]
point_coords = np.array([[w / 2, h / 2]], dtype=np.float32)
point_labels = np.array([1], dtype=np.int32)

model = build_sam2(CONFIG, CHECKPOINT, device="cuda")
model.eval()
predictor = SAM2ImagePredictor(model)


def predict(use_bf16):
    torch.cuda.synchronize()

    with torch.inference_mode():
        if use_bf16:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            ):
                predictor.set_image(image_rgb)
                masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
        else:
            predictor.set_image(image_rgb)
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

    torch.cuda.synchronize()

    return (
        np.asarray(masks).copy(),
        np.asarray(scores).copy(),
        np.asarray(logits).copy(),
    )


fp32_masks, fp32_scores, _ = predict(False)
bf16_masks, bf16_scores, _ = predict(True)

fp32_index = int(np.argmax(fp32_scores))
bf16_index = int(np.argmax(bf16_scores))

fp32_best = fp32_masks[fp32_index].astype(bool)
bf16_best = bf16_masks[bf16_index].astype(bool)

intersection = int(np.logical_and(fp32_best, bf16_best).sum())
union = int(np.logical_or(fp32_best, bf16_best).sum())

fp32_area = int(fp32_best.sum())
bf16_area = int(bf16_best.sum())

iou = intersection / union if union else 1.0
dice_denominator = fp32_area + bf16_area
dice = (
    2.0 * intersection / dice_denominator
    if dice_denominator else 1.0
)

pixel_agreement = float(np.mean(fp32_best == bf16_best))
pixel_disagreement = 1.0 - pixel_agreement

pairwise_iou = []

for fp_mask in fp32_masks:
    row = []

    for bf_mask in bf16_masks:
        a = fp_mask.astype(bool)
        b = bf_mask.astype(bool)

        pair_intersection = np.logical_and(a, b).sum()
        pair_union = np.logical_or(a, b).sum()

        row.append(
            float(pair_intersection / pair_union)
            if pair_union else 1.0
        )

    pairwise_iou.append(row)

result = {
    "model": "SAM2 Hiera Tiny",
    "comparison": "FP32 versus BF16 autocast with FP32 weights",
    "image": IMAGE_PATH,
    "input_size": [w, h],
    "prompt": {
        "type": "one positive center point",
        "coordinates": point_coords.tolist(),
    },
    "fp32": {
        "best_mask_index": fp32_index,
        "scores": fp32_scores.tolist(),
        "best_score": float(fp32_scores[fp32_index]),
        "best_mask_area_pixels": fp32_area,
    },
    "bf16": {
        "best_mask_index": bf16_index,
        "scores": bf16_scores.tolist(),
        "best_score": float(bf16_scores[bf16_index]),
        "best_mask_area_pixels": bf16_area,
    },
    "best_mask_comparison": {
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": float(iou),
        "dice": float(dice),
        "pixel_agreement": pixel_agreement,
        "pixel_disagreement": pixel_disagreement,
        "absolute_area_difference_pixels": abs(
            bf16_area - fp32_area
        ),
        "score_difference": float(
            bf16_scores[bf16_index]
            - fp32_scores[fp32_index]
        ),
    },
    "pairwise_mask_iou_matrix": pairwise_iou,
}

json_path = (
    OUTPUT_DIR /
    "sam2_fp32_bf16_mask_consistency.json"
)

with open(json_path, "w") as file:
    json.dump(result, file, indent=2)

cv2.imwrite(
    str(OUTPUT_DIR / "sam2_fp32_best_mask.png"),
    fp32_best.astype(np.uint8) * 255,
)

cv2.imwrite(
    str(OUTPUT_DIR / "sam2_bf16_best_mask.png"),
    bf16_best.astype(np.uint8) * 255,
)

difference = np.logical_xor(
    fp32_best,
    bf16_best
).astype(np.uint8) * 255

cv2.imwrite(
    str(OUTPUT_DIR / "sam2_fp32_bf16_difference.png"),
    difference,
)

print(json.dumps(result, indent=2))
print(f"\nSaved: {json_path}")
