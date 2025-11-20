import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib
from utils.constants import OUTPUTS_DIR
from utils.image_utils import extract_lbp_features, RESIZE_DIMS

# Model paths
MODEL_DIR = os.path.join("models")
LBP_MODEL_PATH = os.path.join(MODEL_DIR, "lbp_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
ROOFTOP_ENCODER_PATH = os.path.join(MODEL_DIR, "rooftop_encoder.pkl")
AREA_SCALER_PATH = os.path.join(MODEL_DIR, "area_scaler.pkl")
TARGET_ENCODERS_PATH = os.path.join(MODEL_DIR, "target_encoders.pkl")
AREA_MODEL_PATH = os.path.join(MODEL_DIR, "area_predictor_model.pt")

class MultiHeadClassifier(nn.Module):
    def __init__(self, input_dim, output_dims):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(64, out_dim) for out_dim in output_dims])

    def forward(self, x):
        shared_out = self.shared(x)
        return [head(shared_out) for head in self.heads]

def predict_from_mask(mask_binary):
    try:
        # Read and apply mask to cropped image
        cropped_path = os.path.join(OUTPUTS_DIR, "cropped_house_area.png")
        img = cv2.imread(cropped_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Find white patch and crop
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No white region found in the mask.")
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        chunk_w, chunk_h = 200, 200
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= chunk_w and h >= chunk_h:
                cx, cy = x + w // 2, y + h // 2
                sx, sy = max(cx - chunk_w // 2, 0), max(cy - chunk_h // 2, 0)
                sx = min(sx, img.shape[1] - chunk_w)
                sy = min(sy, img.shape[0] - chunk_h)
                patch = img[sy:sy + chunk_h, sx:sx + chunk_w]
                patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
                break
        else:
            raise RuntimeError("No white region large enough for patch size.")

        # Calculate area
        pixel_area = np.sum(mask_binary > 0)
        
        # Load models and encoders
        lbp_model = joblib.load(LBP_MODEL_PATH)
        lbp_encoder = joblib.load(LABEL_ENCODER_PATH)
        rooftop_encoder = joblib.load(ROOFTOP_ENCODER_PATH)
        scaler = joblib.load(AREA_SCALER_PATH)
        target_encoders = joblib.load(TARGET_ENCODERS_PATH)

        # Extract LBP features and predict rooftop type with confidence
        lbp_features = extract_lbp_features(patch)
        lbp_probs = lbp_model.predict_proba([lbp_features])[0]
        lbp_pred = lbp_model.predict([lbp_features])[0]
        rooftop_label = lbp_encoder.inverse_transform([lbp_pred])[0]
        rooftop_encoded = rooftop_encoder.transform([rooftop_label])[0]
        rooftop_confidence = float(lbp_probs[lbp_pred])

        # Load and prepare area predictor model
        output_dims = [len(enc.classes_) for enc in target_encoders.values()]
        model = MultiHeadClassifier(input_dim=2, output_dims=output_dims)
        state_dict = torch.load(AREA_MODEL_PATH, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()

        # Make final predictions
        X = np.array([[rooftop_encoded, float(pixel_area)]], dtype=np.float32)
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(X_tensor)

        final_preds = {}
        final_confidences = {}
        
        for logits, (target, encoder) in zip(outputs, target_encoders.items()):
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = float(probs[0][pred_class])
            
            decoded = encoder.inverse_transform([pred_class])[0]
            final_preds[target] = decoded
            final_confidences[target] = confidence

        # Add rooftop type to predictions with confidence
        final_preds['rooftop_type'] = rooftop_label
        final_confidences['rooftop_type'] = rooftop_confidence

        return {
            'predictions': final_preds,
            'confidences': final_confidences,
        }

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise
