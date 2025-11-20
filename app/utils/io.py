import os
import pandas as pd

STORED_MAP_PATH = "app/static/stored_maps"

def get_stored_tiffs():
    return [f for f in os.listdir(STORED_MAP_PATH) if f.endswith((".tif", ".tiff"))]

def load_excel(excel_file):
    return pd.read_excel(excel_file)
