import os
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from utils.constants import OUTPUTS_DIR, MODEL_PATH

def load_sam_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=MODEL_PATH).to(device)
    return SamPredictor(sam)

def generate_masks(cropped_img_path, pixel):
    # Load image
    img_bgr = cv2.imread(cropped_img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Set up predictor
    predictor = load_sam_model()
    predictor.set_image(img_rgb)

    px, py = pixel
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[px, py]]),
        point_labels=np.array([1]),
        multimask_output=True
    )

    binary_masks = [(m > 0.5).astype(np.uint8) * 255 for m in masks]

    # Convert each binary mask into an overlaid image for display
    annotated_images = []
    for b_mask in binary_masks:
        masked = cv2.bitwise_and(img_bgr, img_bgr, mask=b_mask)
        annotated_images.append(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))  # Convert for Streamlit

    return binary_masks, annotated_images
