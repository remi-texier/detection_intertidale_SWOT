# src/processing/dem_processing.py
import logging
import xarray as xr
import numpy as np
from matplotlib.path import Path as MplPath
from typing import Dict, Any, Tuple, Optional, List

from .. import config as app_config

log = logging.getLogger("rich_app")

def crop_to_roi(data_array: xr.DataArray, roi_config: Optional[Any], lon_coord: str, lat_coord: str) -> Tuple[xr.DataArray, bool]:
    if roi_config is None:
        return data_array, False

    is_roi_applied = False
    if isinstance(roi_config, dict) and all(k in roi_config for k in ['min_lon', 'min_lat', 'max_lon', 'max_lat']):
        lon_slice = slice(roi_config['min_lon'], roi_config['max_lon'])
        if data_array[lon_coord].size > 1 and data_array[lon_coord][0] > data_array[lon_coord][-1]:
            lon_slice = slice(roi_config['max_lon'], roi_config['min_lon'])
        
        lat_slice = slice(roi_config['min_lat'], roi_config['max_lat'])
        if data_array[lat_coord].size > 1 and data_array[lat_coord][0] > data_array[lat_coord][-1]:
            lat_slice = slice(roi_config['max_lat'], roi_config['min_lat'])

        sl = {lon_coord: lon_slice, lat_coord: lat_slice}
        try:
            cropped_da = data_array.sel(sl)
            is_roi_applied = True
        except KeyError:
            log.warning(f"Could not select axis-aligned ROI. Coords '{lon_coord}' or '{lat_coord}' missing/invalid.")
            return data_array, False
    else:
        log.warning("Invalid ROI configuration. Using full data extent.")
        return data_array, False

    if cropped_da.size == 0: log.warning("ROI resulted in an empty data slice.")
    elif not cropped_da.notnull().any(): log.warning("ROI resulted in all NaN values.")
    
    return cropped_da, is_roi_applied

def create_permanent_water_mask(elevation_da: xr.DataArray, threshold: float) -> xr.DataArray:
    if elevation_da.size == 0:
        return xr.DataArray(np.empty(elevation_da.shape, dtype=bool), coords=elevation_da.coords, dims=elevation_da.dims, name="permanent_water_mask")
    return (elevation_da < threshold).fillna(False)

def load_and_process_dem(config: dict, ui_queue, report_queue, task_name, pid):
    """Charge et traite le MNT."""
    ui_queue.put((pid, "status", f"{task_name} | Chargement MNT..."))
    
    raw_extent_from_config = config["analysis_roi_bbox_dict"] 
    processed_roi_bbox = {
        'min_lon': min(raw_extent_from_config['lon']),
        'max_lon': max(raw_extent_from_config['lon']),
        'min_lat': min(raw_extent_from_config['lat']), 
        'max_lat': max(raw_extent_from_config['lat'])  
    }

    try:
        with xr.open_dataset(config["mnt_filepath"], decode_coords='all') as mnt_lazy:
            alt_var, lon_coord, lat_coord = config["mnt_alt"], config["mnt_lon"], config["mnt_lat"]
            
            lon_slice = slice(processed_roi_bbox['min_lon'], processed_roi_bbox['max_lon'])
            lat_slice = slice(processed_roi_bbox['max_lat'], processed_roi_bbox['min_lat']) 
            
            if mnt_lazy[lon_coord].values[0] > mnt_lazy[lon_coord].values[-1]: 
                lon_slice = slice(processed_roi_bbox['max_lon'], processed_roi_bbox['min_lon'])
            if mnt_lazy[lat_coord].values[0] < mnt_lazy[lat_coord].values[-1]: 
                lat_slice = slice(processed_roi_bbox['min_lat'], processed_roi_bbox['max_lat'])
            
            elevation_crop = mnt_lazy[alt_var].sel({lon_coord: lon_slice, lat_coord: lat_slice}).load()

        elevation_roi, _ = crop_to_roi(elevation_crop, processed_roi_bbox, lon_coord, lat_coord)

    except Exception as e:
        msg = f"Erreur critique lors du chargement/rognage du MNT : {e}"
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        return None
    
    if elevation_roi is None or elevation_roi.size == 0:
        msg = "Le MNT est vide après application de la ROI. La zone est probablement hors de la couverture du MNT."
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        return None

    return elevation_roi