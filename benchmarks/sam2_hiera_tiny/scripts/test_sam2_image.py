import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
config = "sam2_hiera_t.yaml"

device = "cuda"

model = build_sam2(config, checkpoint, device=device)
predictor = SAM2ImagePredictor(model)

image_path = "../test_image.jpg"
image = cv2.imread(image_path)

if image is None:
    raise RuntimeError("Cannot load test image. Please put test_image.jpg in ~/jetson_projects/sam2_test/")
image = cv2.resize(image, (512, 512))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image_rgb)

h, w = image.shape[:2]

input_point = np.array([[w // 2, h // 2]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

best_idx = np.argmax(scores)
mask = masks[best_idx].astype(bool)

overlay = image.copy()
overlay[mask] = (0, 255, 0)

result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

cv2.circle(result, (w // 2, h // 2), 6, (0, 0, 255), -1)

cv2.imwrite("sam2_test_result.jpg", result)

print("SAM2 segmentation finished")
print("Best score:", scores[best_idx])
print("Saved result to sam2_test_result.jpg")