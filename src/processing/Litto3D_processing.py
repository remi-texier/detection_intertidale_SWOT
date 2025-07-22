import logging
import xarray as xr
import numpy as np
import scipy.ndimage
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("rich_app")

def crop_to_roi(data_array: xr.DataArray, roi_config: Optional[Any], lon_coord: str, lat_coord: str) -> Tuple[xr.DataArray, bool]:
    """
    Crop a DataArray to a rectangular region of interest.
    Returns the cropped DataArray and a boolean indicating if cropping was applied.
    """
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
        try:
            cropped_da = data_array.sel({lon_coord: lon_slice, lat_coord: lat_slice})
            is_roi_applied = True
        except KeyError:
            log.warning(f"Could not select axis-aligned ROI. Coords '{lon_coord}' or '{lat_coord}' missing/invalid.")
            return data_array, False
    else:
        log.warning("Invalid ROI configuration. Using full data extent.")
        return data_array, False

    if cropped_da.size == 0:
        log.warning("ROI resulted in an empty data slice.")
    elif not cropped_da.notnull().any():
        log.warning("ROI resulted in all NaN values.")
    return cropped_da, is_roi_applied


def create_permanent_water_mask(elevation_da: xr.DataArray, threshold: float) -> xr.DataArray:
    """
    Generate a boolean mask of permanent water where elevation is below threshold.
    """
    if elevation_da.size == 0:
        return xr.DataArray(np.empty(elevation_da.shape, dtype=bool), coords=elevation_da.coords, dims=elevation_da.dims, name="permanent_water_mask")
    return (elevation_da < threshold).fillna(False)


def load_and_process_dem(config: Dict[str, Any], ui_queue, report_queue, task_name: str, pid: int) -> Optional[xr.DataArray]:
    """
    Load a DEM from file, crop to ROI, and return the elevation DataArray.
    """
    ui_queue.put((pid, "status", f"{task_name} | Chargement MNT..."))
    raw_extent = config.get("analysis_roi_bbox_dict", {})
    processed_roi_bbox = {
        'min_lon': min(raw_extent.get('lon', [])),
        'max_lon': max(raw_extent.get('lon', [])),
        'min_lat': min(raw_extent.get('lat', [])),
        'max_lat': max(raw_extent.get('lat', []))
    }
    try:
        with xr.open_dataset(config["mnt_filepath"], decode_coords='all') as mnt_lazy:
            alt_var = config["mnt_alt"]
            lon_coord = config["mnt_lon"]
            lat_coord = config["mnt_lat"]
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


def compute_connected_inundation(elevation_da: xr.DataArray, water_level: float, permanent_water_mask: xr.DataArray, depression_depth_for_isolated_source: float = 0.1, uncertainty_margin: float = 0.15) -> xr.DataArray:
    """
    Compute a multi-class inundation mask:
    0 = certain land, 1 = certain water, 2 = uncertain zone.
    """
    if elevation_da.size == 0:
        return xr.DataArray(np.full(elevation_da.shape, np.nan, dtype=np.float32), coords=elevation_da.coords, dims=elevation_da.dims, name='inundation_mask_multiclass')
    elev = elevation_da.data
    potential = (elev < (water_level + uncertainty_margin)) & ~np.isnan(elev)
    if not np.any(potential):
        return xr.zeros_like(elevation_da, dtype=np.int8).where(elevation_da.notnull())
    if permanent_water_mask.shape != elevation_da.shape or not all(permanent_water_mask.coords[c].equals(elevation_da.coords[c]) for c in elevation_da.dims):
        pwm = permanent_water_mask.reindex_like(elevation_da, method='nearest').fillna(False).data
    else:
        pwm = permanent_water_mask.data
    labels, num = scipy.ndimage.label(potential, structure=np.ones((3, 3)))
    if num == 0:
        return xr.zeros_like(elevation_da, dtype=np.int8).where(elevation_da.notnull())
    sources = []
    if np.any(pwm):
        lbls = labels[pwm & (labels > 0)]
        if lbls.size:
            sources.append(np.unique(lbls))
    mins = scipy.ndimage.minimum(elev, labels=labels, index=np.arange(1, num + 1))
    deep = mins < (water_level - depression_depth_for_isolated_source)
    cond2 = np.arange(1, num + 1)[deep]
    if cond2.size:
        sources.append(cond2)
    final = np.unique(np.concatenate(sources)) if sources else np.array([], dtype=int)
    conn = np.isin(labels, final)
    mask = np.full(elev.shape, 2, dtype=np.int8)
    mask[elev >= (water_level + uncertainty_margin)] = 0
    mask[(elev < (water_level - uncertainty_margin)) & conn] = 1
    return xr.DataArray(mask, coords=elevation_da.coords, dims=elevation_da.dims, name='inundation_mask_multiclass').where(elevation_da.notnull())


def compute_inundation_map(elevation_roi: xr.DataArray, tide_height: float, config: Dict[str, Any], ui_queue, pid: int, task_name: str) -> xr.DataArray:
    """
    Generate an inundation map for a given tide height.
    """
    ui_queue.put((pid, "status", f"{task_name} | Calcul inondation..."))
    perm_conf = config.get("perm_water", {})
    min_alt = float(elevation_roi.min(skipna=True).item())
    thresh = min_alt + perm_conf.get("offset", 0.1)
    pwm = create_permanent_water_mask(elevation_roi, thresh)
    margin = config.get("inundation_margin", 0.15)
    return compute_connected_inundation(elevation_roi, tide_height, pwm, config.get("depression_depth_for_isolated_source", 0.1), uncertainty_margin=margin)
