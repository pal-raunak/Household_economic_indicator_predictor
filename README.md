<!-- ---
title: Household Economic Indicator Predictor
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.51.0
app_file: app.py
pinned: false
--- -->

# Household Level Economic Indicator Predictor

This Streamlit app predicts economic indicators for households based on satellite imagery analysis.

## Features

- Upload or select from pre-stored geospatial maps (TIFF files)
- Process household locations using coordinates
- Segment house rooftops using SAM (Segment Anything Model)
- Predict multiple economic indicators including:
  - Rooftop type
  - Floor and wall materials
  - Water supply status
  - Government housing schemes
  - Occupation
  - Ration card status
  - MGNREGA participation
  - Vehicle ownership

## Usage

1. Select or upload a map (TIFF file)
2. Upload an Excel/CSV file with household coordinates (longitude, latitude, and ID)
3. Select a household ID to analyze
4. Review the cropped house image
5. Generate segmentation masks
6. Select the best mask
7. View predictions and compare with ground truth data

## Assets

Large model files and maps are stored on Hugging Face Datasets:
- Dataset: `palrono/economic-indicator-assets`
- Models: SAM checkpoint, LBP models, encoders, scalers
- Maps: Pre-processed TIFF files for different regions

## Environment Variables

Set these in Hugging Face Spaces Settings → Variables and secrets:

- `HF_DATASET_REPO`: `palrono/economic-indicator-assets`
- `HF_DATASET_REVISION`: `main` (optional)
- `USE_LOCAL_ASSETS`: `0` (use remote assets from HF)

