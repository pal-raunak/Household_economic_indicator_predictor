import pandas as pd

from utils.assets import get_map_local_path, get_map_names


def get_stored_tiffs():
    return get_map_names()


def get_local_map_path(map_name: str) -> str:
    return get_map_local_path(map_name)


def load_excel(excel_file):
    return pd.read_excel(excel_file)
