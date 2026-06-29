import cv2
import numpy as np


def compute_image_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "brightness": brightness,
        "contrast": contrast,
        "blur": blur,
        "brightness_status": evaluate_brightness(brightness),
        "contrast_status": evaluate_contrast(contrast),
        "blur_status": evaluate_blur(blur),
    }


def evaluate_brightness(value):
    if value < 60:
        return "Too Dark"
    if value > 200:
        return "Too Bright"
    return "Good"


def evaluate_contrast(value):
    if value < 30:
        return "Low"
    if value > 90:
        return "High"
    return "Good"


def evaluate_blur(value):
    if value < 100:
        return "Blurry"
    return "Sharp"