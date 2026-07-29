import time
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
config = "sam2_hiera_t.yaml"
device = "cuda"

image_path = "../test_image.jpg"

image = cv2.imread(image_path)
if image is None:
    raise RuntimeError("Cannot load test image.")

image = cv2.resize(image, (512, 512))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Device:", device)
print("CUDA available:", torch.cuda.is_available())
print("CUDA memory before:", torch.cuda.mem_get_info())

start_load = time.time()
model = build_sam2(config, checkpoint, device=device)
predictor = SAM2ImagePredictor(model)
torch.cuda.synchronize()
print("Model load time:", round(time.time() - start_load, 3), "s")
print("CUDA memory after load:", torch.cuda.mem_get_info())

start_set = time.time()
predictor.set_image(image_rgb)
torch.cuda.synchronize()
set_image_time = time.time() - start_set
print("Set image time:", round(set_image_time, 3), "s")
print("CUDA memory after set_image:", torch.cuda.mem_get_info())

h, w = image.shape[:2]
input_point = np.array([[w // 2, h // 2]])
input_label = np.array([1])

start_pred = time.time()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
torch.cuda.synchronize()
predict_time = time.time() - start_pred

best_idx = int(np.argmax(scores))
mask = masks[best_idx].astype(bool)

print("Prediction time:", round(predict_time, 3), "s")
print("Best score:", float(scores[best_idx]))
print("Mask area:", int(mask.sum()))
print("CUDA memory after predict:", torch.cuda.mem_get_info())

overlay = image.copy()
overlay[mask] = (0, 255, 0)
result = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
cv2.imwrite("sam2_gpu_profile_result.jpg", result)

print("Saved result to sam2_gpu_profile_result.jpg")