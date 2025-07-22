# src/data_loader.py
import xarray as xr
import numpy as np
import pandas as pd
import os
import logging
import re
import h5py
from typing import Dict, Any, Optional, Union

log = logging.getLogger("rich_app")

def parse_tide_gauge_data(filepath: str) -> Optional[pd.DataFrame]: 
    try:
        df = pd.read_csv(filepath, comment='#', delimiter=';', header=None, names=['DateTimeStr', 'Value', 'Source'], encoding='utf-8') 
        df = df.assign(DateTime=pd.to_datetime(df['DateTimeStr'], format='%d/%m/%Y %H:%M:%S'), 
                       Value=pd.to_numeric(df['Value'], errors='coerce')) 
        if df['DateTime'].dt.tz is not None: 
            df['DateTime'] = df['DateTime'].dt.tz_localize(None) 
        
        clean_df = df[['DateTime', 'Value']].dropna(subset=['Value']).copy()
        if clean_df.empty:
            return None
        return clean_df
    except FileNotFoundError: 
        return None
    except Exception: 
        return None

def load_elevation_data(filepath: str, alt_var: str) -> xr.DataArray:
    with xr.open_dataset(filepath, decode_coords='all') as ds:
        if alt_var not in ds:
            raise ValueError(f"Altitude variable '{alt_var}' not found in {filepath}. Available: {list(ds.variables.keys())}")
        elev_da = ds[alt_var].copy()
    
    has_rio = hasattr(elev_da, 'rio')
    crs_found = False
    if has_rio:
        try:
            if elev_da.rio.crs is not None:
                crs_found = True
        except Exception:
            pass
            
    if not crs_found:
        if 'lat' in elev_da.coords and 'lon' in elev_da.coords:
            log.warning(f"CRS not found via rioxarray in '{filepath}'. Assuming geographic (lat/lon).")
        else:
            log.warning(f"CRS not found via rioxarray in '{filepath}' and standard lat/lon coords not detected.")
            
    return elev_da

def find_swot_files(base_path: str, cycle_id: str, pass_id: str, product_type: str) -> Optional[str]:
    pattern_part = f"SWOT_L2_LR_SSH_{product_type}_{cycle_id}_{pass_id}_"
    for filename in os.listdir(base_path):
        if filename.startswith(pattern_part) and filename.endswith(".nc"):
            return os.path.join(base_path, filename)
    log.warning(f"Aucun fichier {product_type} trouvé pour cycle {cycle_id}, pass {pass_id} avec le motif {pattern_part}")
    return None

def read_swot_datafile(filepath: str, is_expert: bool = False) -> Union[xr.Dataset, Dict[str, xr.Dataset], None]:
    if not os.path.exists(filepath):
        log.warning(f"Erreur (read_swot_datafile): Fichier non trouvé : {filepath}")
        return None
    
    try:
        if is_expert:
            return xr.open_dataset(filepath, engine='netcdf4')
        else: 
            with h5py.File(filepath, 'r') as f:
                top_keys = list(f.keys())

            groups = ["left", "right"]
            load_groups = [g for g in groups if g in top_keys]

            if not load_groups:
                log.warning(f"Warning (read_swot_datafile): Groupes {groups} non trouvés dans {os.path.basename(filepath)}. Tentative de lecture de la racine.")
                try:
                    ds_root = xr.open_dataset(filepath, engine='netcdf4')
                    return {'data_root': ds_root} if ds_root.data_vars or ds_root.coords else {}
                except Exception as e:
                    log.warning(f"Erreur (read_swot_datafile): Impossible de charger la racine de {os.path.basename(filepath)}: {e}")
                    return None

            data_dict = {}
            for group in load_groups:
                try:
                    data_dict[group] = xr.open_dataset(filepath, group=group, engine='netcdf4')
                except Exception as e:
                    log.warning(f"Warning (read_swot_datafile): Impossible de charger le groupe '{group}' depuis {filepath}: {e}")
            
            if not data_dict:
                log.warning(f"Erreur (read_swot_datafile): Aucun groupe attendu n'a pu être chargé depuis {os.path.basename(filepath)}.")
                return None
            return data_dict
    except Exception as e:
        log.warning(f"Erreur (read_swot_datafile): Erreur générale lors de la lecture de {filepath}: {e}")
        return None
