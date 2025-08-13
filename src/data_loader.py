# src/data_loader.py
import xarray as xr
import numpy as np
import pandas as pd
import os
import logging
import traceback
import re
import h5py
from typing import Dict, Any, Optional, Union, List

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
    if product_type == "HR":
        hr_path = os.path.join(base_path, "L2_HR")
        if os.path.exists(hr_path):
            hr_files = []
            for filename in os.listdir(hr_path):
                if not filename.endswith(".nc"):
                    continue
                info = extract_tile_info_from_filename(filename)
                if not info:
                    continue
                if info.get('cycle') == str(cycle_id) and info.get('pass') == str(pass_id):
                    hr_files.append(os.path.join(hr_path, filename))
            if hr_files:
                log.info(f"Trouvé {len(hr_files)} fichier(s) HR pour cycle {cycle_id}, pass {pass_id}")
                return hr_files[0]
        log.warning(f"Aucun fichier HR trouvé pour cycle {cycle_id}, pass {pass_id} dans {hr_path}")
        return None
    else:
        lr_path = os.path.join(base_path, "L2_LR")
        if os.path.exists(lr_path):
            pattern_part = f"SWOT_L2_LR_SSH_{product_type}_{cycle_id}_{pass_id}_"
            for filename in os.listdir(lr_path):
                if filename.startswith(pattern_part) and filename.endswith(".nc"):
                    return os.path.join(lr_path, filename)
        log.warning(f"Aucun fichier {product_type} trouvé pour cycle {cycle_id}, pass {pass_id} avec le motif {pattern_part}")
        return None

def find_hr_tiles(base_path: str, cycle_id: str, pass_id: str) -> List[str]:
    hr_files = []
    hr_path = os.path.join(base_path, "L2_HR")
    
    if os.path.exists(hr_path):
        for filename in os.listdir(hr_path):
            if not filename.endswith('.nc'):
                continue
            info = extract_tile_info_from_filename(filename)
            if not info:
                continue
            if info.get('cycle') == str(cycle_id) and info.get('pass') == str(pass_id):
                hr_files.append(os.path.join(hr_path, filename))
        
        hr_files.sort()
        
        if hr_files:
            log.info(f"Trouvé {len(hr_files)} tuile(s) HR pour cycle {cycle_id}, pass {pass_id}")
            for file in hr_files:
                basename = os.path.basename(file)
                info = extract_tile_info_from_filename(basename) or {}
                tile = info.get('tile')
                utm = info.get('utm')
                hemisphere = info.get('hemisphere')
                epsg = info.get('epsg')
                if tile:
                    log.info(f"  Tuile: {tile} | UTM: {utm}_{hemisphere} | CRS: {epsg} - {basename}")
    
    return hr_files

def get_tiles(base_path: str, cycle_id: str, pass_id: str, zone_data: Optional[Dict] = None) -> Optional[str]:
    available_tiles = find_hr_tiles(base_path, cycle_id, pass_id)
    
    if not available_tiles:
        return None
    
    if len(available_tiles) == 1:
        return available_tiles[0]
    
    if zone_data and 'tile' in zone_data:
        preferred_tiles = zone_data['tile']
        if isinstance(preferred_tiles, str):
            preferred_tiles = [preferred_tiles]
        
        for available_file in available_tiles:
            tile_info = extract_tile_info_from_filename(available_file)
            if tile_info and tile_info['tile'] in preferred_tiles:
                log.info(f"Tuile HR sélectionnée selon configuration zone: {tile_info['tile']}")
                return available_file
        
        log.warning(f"Aucune des tiles préférées {preferred_tiles} n'est disponible. Utilisation du fallback.")
    
    tile_info = extract_tile_info_from_filename(available_tiles[0])
    if tile_info:
        log.info(f"Tuile HR sélectionnée par défaut: {tile_info['tile']}")
    return available_tiles[0]

def extract_tile_info_from_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Extrait les informations (cycle, pass, tuile, UTM zone/bande, hémisphère, EPSG) d'un nom de fichier HR.
    Format attendu (généralisé):
      SWOT_L2_HR_Raster_100m_UTM{zone}{band}_{hemisphere}_x_x_x_{cycle}_{pass}_{tile}_...
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')

    if len(parts) < 13:
        return None
    if not (parts[0] == "SWOT" and parts[1] == "L2" and parts[2] == "HR" and parts[3] == "Raster" and parts[4] == "100m"):
        return None
    utm_part = parts[5]
    hemisphere = parts[6] if len(parts) > 6 else None
    if not (utm_part.startswith("UTM") and hemisphere in ("N", "S")):
        return None
    if any(p != 'x' for p in parts[7:10]):
        return None

    cycle = parts[10]
    pas = parts[11]
    tile = parts[12] if len(parts) > 12 else None

    utm_code = utm_part[3:]  # e.g., '30T', '31U'
    zone_str = ''
    for ch in utm_code:
        if ch.isdigit():
            zone_str += ch
        else:
            break
    epsg = None
    if zone_str and hemisphere in ("N", "S"):
        try:
            zone_num = int(zone_str)
            epsg_num = (32600 if hemisphere == 'N' else 32700) + zone_num
            epsg = f"EPSG:{epsg_num}"
        except ValueError:
            epsg = None

    return {
        'cycle': cycle,
        'pass': pas,
        'tile': tile,
        'utm': utm_code,
        'hemisphere': hemisphere,
        'epsg': epsg,
        'filename': basename
    }

def read_swot_datafile(filepath: str, is_expert: bool = False) -> Union[xr.Dataset, Dict[str, xr.Dataset], None]:
    if not os.path.exists(filepath):
        log.warning(f"Erreur (read_swot_datafile): Fichier non trouvé : {filepath}")
        return None
    
    try:
        if is_expert:
            return xr.open_dataset(filepath, engine='netcdf4')
        
        is_hr_file = "HR_Raster" in os.path.basename(filepath)
        
        if is_hr_file:
            try:
                ds = xr.open_dataset(filepath, engine='netcdf4')
                info = extract_tile_info_from_filename(os.path.basename(filepath))
                if info and info.get('epsg'):
                    ds.attrs['source_epsg'] = info['epsg']
                return {"main": ds}  
            except Exception as e:
                log.warning(f"Erreur (read_swot_datafile): Impossible de charger le fichier HR {filepath}: {e}")
                return None
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
        tb_str = traceback.format_exc()
        log.error(f"Erreur générale lors de la lecture de {filepath}: {e}\n\nTraceback:\n{tb_str}", exc_info=True)
        return None
