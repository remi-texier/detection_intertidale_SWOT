# src/processing/swot_processing.py
import logging
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from matplotlib.path import Path as MplPath
from typing import Dict, Any, Tuple, Optional, List, Union
from datetime import datetime
import pandas as pd

from .. import data_loader

log = logging.getLogger("rich_app")


LR250M_VARIABLES = ["time", "longitude", "latitude", "ssh_karin_2", "ssh_karin_qual", "height_cor_xover", "sig0_karin_2", "sig0_karin_qual"]
HR100M_VARIABLES = ["illumination_time", "longitude", "latitude", "wse", "geoid", "wse_qual", "height_cor_xover", "sig0", "sig0_qual"]

def clean_swot_dataset_variables(data: xr.Dataset, key_variables: List[str] = LR250M_VARIABLES) -> xr.Dataset:
    def _clean_single_ds(ds: xr.Dataset) -> xr.Dataset:
        if not isinstance(ds, xr.Dataset): return ds
        vars_to_drop = [var for var in ds.variables if var not in key_variables and var not in ds.dims]
        return ds.drop_vars(vars_to_drop, errors='ignore')

    if isinstance(data, dict):
        return {group: _clean_single_ds(ds) for group, ds in data.items()}
    elif isinstance(data, xr.Dataset):
        return _clean_single_ds(data)
    return data

def _normalize_longitude_array(lon_array: xr.DataArray) -> xr.DataArray:
    return xr.where(lon_array > 180, lon_array - 360, lon_array)


def apply_roi_to_swot_dataset(dataset: xr.Dataset, roi_config: Dict[str, Any], lon_coord_name: str = "longitude", lat_coord_name: str = "latitude") -> xr.Dataset:
    if roi_config is None:
        return dataset

    if not (lon_coord_name in dataset.coords and lat_coord_name in dataset.coords):
        log.warning(f"Coords '{lon_coord_name}' or '{lat_coord_name}' manquantes.")
        return dataset

    if isinstance(roi_config, dict) and all(k in roi_config for k in ['lon', 'lat']):
        lon_bounds = sorted(roi_config['lon'])
        lat_bounds = sorted(roi_config['lat'])

        lon_norm = _normalize_longitude_array(dataset[lon_coord_name])
        mask = ((dataset[lat_coord_name] >= lat_bounds[0]) & (dataset[lat_coord_name] <= lat_bounds[1]) & (lon_norm >= lon_bounds[0]) & (lon_norm <= lon_bounds[1]))
        
        valid_lines = mask.any(dim='num_pixels')
        valid_pixels = mask.any(dim='num_lines')
        
        if not valid_lines.any() or not valid_pixels.any():
            return dataset.isel(num_lines=slice(0), num_pixels=slice(0))

        subset = dataset.sel(num_lines=valid_lines, num_pixels=valid_pixels)
        
        return subset.where(mask.sel(num_lines=valid_lines, num_pixels=valid_pixels))
    else:
        log.warning("Config ROI invalide.")
        return dataset

def apply_roi_to_swot_data_groups(swot_groups: Dict[str, xr.Dataset], roi_config: Dict[str, List[float]], lon_coord_name: str = "longitude", lat_coord_name: str = "latitude") -> Dict[str, xr.Dataset]:
    if roi_config is None: return swot_groups
    processed = {}
    for group_name, ds in swot_groups.items():
        if not isinstance(ds, xr.Dataset):
            processed[group_name] = ds; continue
        if not (lon_coord_name in ds and lat_coord_name in ds and \
                ds.sizes.get('num_lines', 0) > 0 and ds.sizes.get('num_pixels', 0) > 0):
            log.warning(f"Données ou dims manquantes pour ROI sur groupe {group_name}.")
            processed[group_name] = ds 
            continue
        processed_ds = apply_roi_to_swot_dataset(ds, roi_config, lon_coord_name, lat_coord_name)
        processed[group_name] = processed_ds
    return processed

def apply_ssh_correction(unsmoothed_groups: Dict[str, xr.Dataset], expert_data: Optional[xr.Dataset]) -> Dict[str, xr.Dataset]:
    corrected = {k: v.copy(deep=True) for k, v in unsmoothed_groups.items() if isinstance(v, xr.Dataset)}

    if expert_data is None:
        log.warning("Données Expert non fournies. Aucune correction SSH appliquée.")
        return None

    required_vars = ["longitude", "latitude", "height_cor_xover"]
    if not all(v in expert_data for v in required_vars):
        log.warning("Variables requises manquantes dans les données Expert. Aucune correction SSH appliquée.")
        return None

    lon_expert = _normalize_longitude_array(expert_data.longitude.values).flatten()
    lat_expert = expert_data.latitude.values.flatten()
    correction_vals = expert_data['height_cor_xover'].values.flatten()
    valid_idx = ~np.isnan(lon_expert) & ~np.isnan(lat_expert) & ~np.isnan(correction_vals)
    
    if not np.any(valid_idx):
        log.warning("Aucune donnée de correction valide dans les données Expert. Aucune correction SSH appliquée.")
        return None
        
    expert_points = np.vstack((lon_expert[valid_idx], lat_expert[valid_idx])).T
    expert_values = correction_vals[valid_idx]

    for group_name, ds in corrected.items():
        required_vars = ["longitude", "latitude", "ssh_karin_2"]
        if not isinstance(ds, xr.Dataset) or not all(v in ds for v in required_vars):
            break
        
        lon_unsmoothed = _normalize_longitude_array(ds.longitude.values).flatten()
        lat_unsmoothed = ds.latitude.values.flatten()
        target_points = np.vstack((lon_unsmoothed, lat_unsmoothed)).T
        
        interpolated = griddata(expert_points, expert_values, target_points, method='linear', fill_value=0.0)
        correction_grid = np.reshape(interpolated, ds.longitude.shape)
        ds["ssh_karin_2_corrected"] = ds["ssh_karin_2"] + correction_grid
        log.info(f"Correction SSH appliquée au groupe {group_name}.")
    return corrected

def load_and_process_swot_data(config: dict, cycle: str, pass_id: str, zone_data: dict, ui_queue, report_queue, task_name, pid):
    """Charge et traite les données SWOT."""
    ui_queue.put((pid, "status", f"{task_name} | Chargement SWOT..."))
    expert_file = data_loader.find_swot_files(config["data_path"], cycle, pass_id, "Expert") 
    unsmoothed_file = data_loader.find_swot_files(config["data_path"], cycle, pass_id, "Unsmoothed")
    
    if not expert_file:
        msg = f"Fichier SWOT Expert manquant pour la passe {pass_id} / cycle {cycle}. Le traitement est abandonné car la correction SSH est requise."
        report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': msg})
        return None, None

    expert_data = data_loader.read_swot_datafile(expert_file, is_expert=True) if expert_file else None 
    unsmoothed_groups = data_loader.read_swot_datafile(unsmoothed_file, is_expert=False) 

    if not unsmoothed_groups or not any(isinstance(ds, xr.Dataset) for ds in unsmoothed_groups.values()): 
        msg = "Aucune donnée Unsmoothed SWOT chargée depuis le fichier. Le fichier pourrait être corrompu ou vide."
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        return None, None
    
    ui_queue.put((pid, "status", f"{task_name} | Traitement SWOT..."))
    allowed_groups = zone_data.get("data_group", []) 
    filtered = {gn: ds for gn, ds in unsmoothed_groups.items() if not allowed_groups or gn in allowed_groups}
    if not filtered: 
        msg = f"Aucun groupe de données Unsmoothed ('left'/'right') ne correspond à la configuration '{allowed_groups}'."
        report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': msg})
        return None, None

    cleaned = clean_swot_dataset_variables(filtered) 
    roi_bbox = config["analysis_roi_bbox_dict"] 
    roi_groups = apply_roi_to_swot_data_groups(cleaned, roi_bbox) 
    
    expert_roi = None 
    if expert_data is not None: 
        expert_cleaned = clean_swot_dataset_variables(expert_data) 
        expert_roi = apply_roi_to_swot_dataset(expert_cleaned, roi_bbox) 

    valid_groups = {gn: ds for gn, ds in roi_groups.items() if isinstance(ds, xr.Dataset) and ds.sizes.get('num_lines', 0) > 0}
    if not valid_groups: 
        msg = "Aucune donnée SWOT Unsmoothed ne se trouve dans la zone d'intérêt (ROI) après filtrage."
        report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': msg})
        return None, None
    
    expert_valid = (expert_roi is not None and isinstance(expert_roi, xr.Dataset) and expert_roi.sizes.get('num_lines', 0) > 0)
    final_groups = apply_ssh_correction(valid_groups, expert_roi if expert_valid else None) 

    display_group = next((g for g in (allowed_groups or final_groups.keys()) if g in final_groups and isinstance(final_groups[g], xr.Dataset) and final_groups[g].sizes.get('num_lines',0) > 0), None)
    if not display_group: 
        msg = "Aucun groupe SWOT valide ('left'/'right') ne contient de données après tous les filtrages."
        report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': msg})
        return None, None
    
    return final_groups[display_group], unsmoothed_file

def process_swot_orientation_and_time(swot_data, ui_queue, report_queue, task_name, pid):
    if "ssh_karin_2_corrected" not in swot_data: 
        msg = f"La variable requise 'ssh_karin_2_corrected' n'a pas été trouvée dans le dataset SWOT."
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        return None, None
    
    if 'latitude' in swot_data and 'num_lines' in swot_data.dims and swot_data.sizes['num_lines'] > 1:
        try:
            lat_col = swot_data['latitude'].isel(num_pixels=swot_data.sizes['num_pixels'] // 2)
            valid_lats = lat_col.dropna(dim='num_lines', how='all')
            if valid_lats.size > 1 and valid_lats.isel(num_lines=0).item() > valid_lats.isel(num_lines=-1).item():
                ui_queue.put((pid, "log", "Trace N->S, inversion..."))
                swot_data = swot_data.isel(num_lines=slice(None, None, -1))
        except (IndexError, ValueError) as e:
            report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': f"Orientation SWOT non déterminée. Erreur: {e}"})

    time_array = swot_data["time"] 
    swot_time, is_fallback = datetime.now(), True
    if time_array.size > 0: 
        valid_times = time_array.values.flatten()[~np.isnat(time_array.values.flatten())]
        if valid_times.size > 0: 
            median_ns = np.median(valid_times.astype('datetime64[ns]').astype(np.int64)) 
            swot_time = pd.to_datetime(median_ns, unit='ns') 
            is_fallback = False 
    ui_queue.put((pid, "log", f"Heure SWOT: {swot_time.strftime('%H:%M:%S')}{' (Fallback)' if is_fallback else ''}"))

    return swot_data, (swot_time, is_fallback)

def rasterize_swot_data(swot_data: xr.Dataset, target_grid: xr.DataArray, config: dict) -> Dict[str, xr.DataArray]:
    result = {}
    lon_mesh, lat_mesh = np.meshgrid(target_grid[config["mnt_lon"]].values, target_grid[config["mnt_lat"]].values, indexing='xy')
    
    src_lons = _normalize_longitude_array(swot_data.longitude.values).flatten()
    src_lats = swot_data.latitude.values.flatten()

    vars_to_rasterize = {
        'swot_ssh': {'src_var': 'ssh_karin_2_corrected', 'units': 'm'},
        'swot_sig0': {'src_var': 'sig0_karin_2', 'units': 'dB'}
    }

    for out_name, params in vars_to_rasterize.items():
        src_var = params['src_var']
        if src_var not in swot_data:
            continue
            
        src_values = swot_data[src_var].values.flatten()
        valid_mask = ~np.isnan(src_lons) & ~np.isnan(src_lats) & ~np.isnan(src_values)
        
        if np.any(valid_mask):
            raster_data = griddata(np.vstack((src_lons[valid_mask], src_lats[valid_mask])).T, src_values[valid_mask], (lon_mesh, lat_mesh), method='nearest')
            result[out_name] = xr.DataArray(data=raster_data, coords=target_grid.coords, dims=target_grid.dims, name=out_name, attrs={'units': params['units']})
            
    return result