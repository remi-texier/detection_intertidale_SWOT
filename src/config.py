# src/config.py
import os
import json
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GLOBAL_CONFIG: Dict[str, Any] = {
    "data_path": os.path.join(PROJECT_ROOT, "data"),
    "data_type": "HR",  # "HR" ou "LR"
    "zone_info_filepath": os.path.join(PROJECT_ROOT, "data", 'zone_info.json'),
    "mnt_filepath": os.path.join(PROJECT_ROOT, "data", "franceRgeAltiLitto3D5m.nc"),
    "mnt_alt": "Band1",
    "mnt_lon": "lon",
    "mnt_lat": "lat",
    "perm_water": {"method": "mnt_min_plus_offset", "offset": 0.1},
    "depression_depth_for_isolated_source": 0.1,
    "inundation_margin": 0.1,
    "results_base_path": os.path.join(PROJECT_ROOT, "results"),
    "rasterization_downsampling_factor": 1,
    "num_processes": 8,
}

def load_zone_configurations(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier d'information de zone non trouvé : {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)