import time
import cv2
import numpy as np
import torch

from segment_anything import sam_model_registry, SamPredictor


checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"

image_path = "~/jetson_projects/sam2_test/test_image.jpg"
image_path = image_path.replace("~", "/home/xia")

image = cv2.imread(image_path)

if image is None:
    raise RuntimeError("Cannot load test image.")

image = cv2.resize(image, (256, 256))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Device:", device)
print("CUDA available:", torch.cuda.is_available())
print("CUDA memory before:", torch.cuda.mem_get_info())

start_load = time.time()

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)
sam.half()
sam.prompt_encoder.float()
sam.mask_decoder.float()
predictor = SamPredictor(sam)

load_time = time.time() - start_load
print(f"Model load time: {load_time:.2f} s")
print("CUDA memory after load:", torch.cuda.mem_get_info())

start_set_image = time.time()
print("Input image shape:", image_rgb.shape)
torch.cuda.empty_cache()
predictor.set_image(image_rgb)
predictor.features = predictor.features.float()
set_image_time = time.time() - start_set_image

h, w = image.shape[:2]

input_point = np.array([[w // 2, h // 2]])
input_label = np.array([1])

start_infer = time.time()

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

infer_time = time.time() - start_infer

best_idx = int(np.argmax(scores))
mask = masks[best_idx].astype(bool)

overlay = image.copy()
overlay[mask] = (0, 255, 0)

result = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
cv2.circle(result, (w // 2, h // 2), 8, (0, 0, 255), -1)

cv2.imwrite("sam_gpu_test_result.jpg", result)

print("SAM GPU test finished")
print("Best score:", float(scores[best_idx]))
print(f"Set image time: {set_image_time:.2f} s")
print(f"Inference time: {infer_time:.2f} s")
print("CUDA memory after inference:", torch.cuda.mem_get_info())
print("Saved result to sam_gpu_test_result.jpg")