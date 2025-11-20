import json
import os
from functools import lru_cache
from typing import Dict, List

from huggingface_hub import hf_hub_download

from utils.constants import (
    HF_CACHE_DIR,
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
    LOCAL_MAPS_DIR,
    LOCAL_MODELS_DIR,
    MAP_MANIFEST_PATH,
    MODEL_PATH,
    USE_LOCAL_ASSETS,
)


def _hf_download(path_in_repo: str) -> str:
    """Download a file from the configured Hugging Face dataset repo and return the local path."""
    return hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=path_in_repo,
        repo_type="dataset",
        revision=HF_DATASET_REVISION,
        cache_dir=HF_CACHE_DIR or None,
    )


@lru_cache
def _load_map_manifest() -> List[Dict[str, str]]:
    if not os.path.exists(MAP_MANIFEST_PATH):
        return []
    with open(MAP_MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_map_names() -> List[str]:
    """Return the display names for all maps defined in the manifest."""
    return [entry["name"] for entry in _load_map_manifest()]


def _get_map_entry(name: str) -> Dict[str, str]:
    for entry in _load_map_manifest():
        if entry["name"] == name:
            return entry
    raise ValueError(f"Map '{name}' not found in manifest at {MAP_MANIFEST_PATH}")


def get_map_local_path(name: str) -> str:
    """Return a local filesystem path for the requested map, downloading it if necessary."""
    entry = _get_map_entry(name)
    rel_path = entry.get("path")
    if not rel_path:
        raise ValueError(f"Manifest entry for '{name}' is missing 'path'")

    filename = entry.get("filename") or os.path.basename(rel_path)
    if USE_LOCAL_ASSETS:
        candidate = os.path.join(LOCAL_MAPS_DIR, filename)
        if os.path.exists(candidate):
            return candidate

    return _hf_download(rel_path)


def get_model_file(filename: str) -> str:
    """Return a local path to a model/asset stored under the models/ folder."""
    if USE_LOCAL_ASSETS:
        candidate = os.path.join(LOCAL_MODELS_DIR, filename)
        if os.path.exists(candidate):
            return candidate
    return _hf_download(f"models/{filename}")


def get_sam_checkpoint_path() -> str:
    """Return the SAM checkpoint path, preferring a local file if available."""
    if USE_LOCAL_ASSETS and os.path.exists(MODEL_PATH):
        return MODEL_PATH
    return get_model_file(os.path.basename(MODEL_PATH))

