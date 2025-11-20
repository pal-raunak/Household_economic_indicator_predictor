import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Parameters for LBP
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"
RESIZE_DIMS = (128, 128)

def extract_lbp_features(img):
    """Extract Local Binary Pattern features from an image."""
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Resize to match script3_predict.py
    gray = cv2.resize(gray, RESIZE_DIMS)
    
    # Calculate LBP
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
    
    # Calculate histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                          range=(0, n_bins), density=True)
    return hist 