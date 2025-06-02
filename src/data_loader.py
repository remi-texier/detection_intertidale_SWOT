# src/data_loader.py
import xarray as xr
import numpy as np
import pandas as pd
import os
import re
import h5py
from typing import Dict, Any, Optional, Union

def parse_tide_gauge_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, comment='#', delimiter=';', header=None, names=['DateTimeStr', 'Value', 'Source'], encoding='utf-8')
    df = df.assign(DateTime=pd.to_datetime(df['DateTimeStr'], format='%d/%m/%Y %H:%M:%S'),
                   Value=pd.to_numeric(df['Value'], errors='coerce'))
    return df[['DateTime', 'Value']].dropna(subset=['Value']).copy()

def load_elevation_data(filepath: str, alt_var_name: str) -> xr.DataArray:
    with xr.open_dataset(filepath, decode_coords='all') as ds:
        if alt_var_name not in ds:
            raise ValueError(f"Altitude variable '{alt_var_name}' not found in {filepath}. Available: {list(ds.variables.keys())}")
        elevation_da = ds[alt_var_name].copy()
    
    has_rio = hasattr(elevation_da, 'rio')
    crs_found = False
    if has_rio:
        try:
            if elevation_da.rio.crs is not None:
                crs_found = True
        except Exception:
            pass
            
    if not crs_found:
        if 'lat' in elevation_da.coords and 'lon' in elevation_da.coords:
            print(f"Warning (load_elevation_data): CRS not found via rioxarray in '{filepath}'. Assuming geographic (lat/lon).")
        else:
            print(f"Warning (load_elevation_data): CRS not found via rioxarray in '{filepath}' and standard lat/lon coords not detected.")
            
    return elevation_da

def find_swot_files(base_path: str, cycle_id: str, pass_id: str, product_type: str) -> Optional[str]:
    target_pattern_part = f"SWOT_L2_LR_SSH_{product_type}_{cycle_id}_{pass_id}_"
    for filename in os.listdir(base_path):
        if filename.startswith(target_pattern_part) and filename.endswith(".nc"):
            return os.path.join(base_path, filename)
    print(f"Aucun fichier {product_type} trouvé pour cycle {cycle_id}, pass {pass_id} avec le motif {target_pattern_part}")
    return None

def read_swot_datafile(filepath: str, is_expert: bool = False) -> Union[xr.Dataset, Dict[str, xr.Dataset], None]:
    if not os.path.exists(filepath):
        print(f"Erreur (read_swot_datafile): Fichier non trouvé : {filepath}")
        return None
    
    try:
        if is_expert:
            return xr.open_dataset(filepath, engine='netcdf4')
        else: 
            with h5py.File(filepath, 'r') as f:
                available_top_level_keys = list(f.keys())

            expected_groups = ["left", "right"]
            groups_to_load = [g for g in expected_groups if g in available_top_level_keys]

            if not groups_to_load:
                print(f"Warning (read_swot_datafile): Groupes {expected_groups} non trouvés dans {os.path.basename(filepath)}. Tentative de lecture de la racine.")
                try:
                    ds_root = xr.open_dataset(filepath, engine='netcdf4')
                    return {'data_root': ds_root} if ds_root.data_vars or ds_root.coords else {}
                except Exception as e_root:
                    print(f"Erreur (read_swot_datafile): Impossible de charger la racine de {os.path.basename(filepath)}: {e_root}")
                    return None

            data_dict = {}
            for group_name in groups_to_load:
                try:
                    data_dict[group_name] = xr.open_dataset(filepath, group=group_name, engine='netcdf4')
                except Exception as e_group:
                    print(f"Warning (read_swot_datafile): Impossible de charger le groupe '{group_name}' depuis {filepath}: {e_group}")
            
            if not data_dict:
                print(f"Erreur (read_swot_datafile): Aucun groupe attendu n'a pu être chargé depuis {os.path.basename(filepath)}.")
                return None
            return data_dict
    except Exception as e:
        print(f"Erreur (read_swot_datafile): Erreur générale lors de la lecture de {filepath}: {e}")
        return None


def extract_swot_filename_info(filename: str) -> Optional[Dict[str, Any]]:
    basename = os.path.basename(filename)
    pattern = r"SWOT_L2_LR_SSH_(Unsmoothed|Expert)_(\d{3})_(\d{3})_(\d{8}T\d{6})_(\d{8}T\d{6})_(\w+)_(\d{2})\.nc"
    match = re.match(pattern, basename)
    if match:
        return {
            "product_type": match.group(1), "cycle": int(match.group(2)), 
            "pass_id": int(match.group(3)), "start_time": match.group(4), 
            "end_time": match.group(5), "processing_version": match.group(6), 
            "processing_counter": int(match.group(7))
        }
    return None