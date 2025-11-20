# app/utils/constants.py

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sam_vit_h_4b8939.pth")

MAP_MANIFEST_PATH = os.path.join(BASE_DIR, "app", "static", "map_manifest.json")
LOCAL_MODELS_DIR = os.path.join(BASE_DIR, "models")
LOCAL_MAPS_DIR = os.path.join(BASE_DIR, "app", "static", "stored_maps")

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "palrono/economic-indicator-assets")
HF_DATASET_REVISION = os.getenv("HF_DATASET_REVISION", "main")
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")

USE_LOCAL_ASSETS = os.getenv("USE_LOCAL_ASSETS", "0").lower() in ("1", "true", "yes")