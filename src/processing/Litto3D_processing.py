import logging
import traceback
import xarray as xr
import numpy as np
import scipy.ndimage
from typing import Any, Dict, Optional, Tuple
from scipy.ndimage import binary_dilation

log = logging.getLogger("rich_app")

def crop_to_roi(data_array: xr.DataArray, roi_config: Optional[Any], lon: str, lat: str) -> Tuple[xr.DataArray, bool]:
    """
    Crop a DataArray to a rectangular region of interest.
    Returns the cropped DataArray and a boolean indicating if cropping was applied.
    """
    if roi_config is None:
        return data_array, False

    is_roi_applied = False
    if isinstance(roi_config, dict) and all(k in roi_config for k in ['min_lon', 'min_lat', 'max_lon', 'max_lat']):
        lon_slice = slice(roi_config['min_lon'], roi_config['max_lon'])
        if data_array[lon].size > 1 and data_array[lon][0] > data_array[lon][-1]:
            lon_slice = slice(roi_config['max_lon'], roi_config['min_lon'])
        lat_slice = slice(roi_config['min_lat'], roi_config['max_lat'])
        if data_array[lat].size > 1 and data_array[lat][0] > data_array[lat][-1]:
            lat_slice = slice(roi_config['max_lat'], roi_config['min_lat'])
        try:
            cropped_da = data_array.sel({lon: lon_slice, lat: lat_slice})
            is_roi_applied = True
        except KeyError:
            log.warning(f"Could not select axis-aligned ROI. Coords '{lon}' or '{lat}' missing/invalid.")
            return data_array, False
    else:
        log.warning("Invalid ROI configuration. Using full data extent.")
        return data_array, False

    if cropped_da.size == 0:
        log.warning("ROI resulted in an empty data slice.")
    elif not cropped_da.notnull().any():
        log.warning("ROI resulted in all NaN values.")
    return cropped_da, is_roi_applied


def create_perm_water(elevation: xr.DataArray, threshold: float) -> xr.DataArray:
    """
    Generate a boolean mask of permanent water where elevation is below threshold.
    """
    if elevation.size == 0:
        return xr.DataArray(np.empty(elevation.shape, dtype=bool), coords=elevation.coords, dims=elevation.dims, name="perm_water")
    return (elevation < threshold).fillna(False)


def load_and_process_dem(config: Dict[str, Any], ui_queue, report_queue, task_name: str, pid: int) -> Optional[xr.DataArray]:
    """
    Load a DEM from file, crop to ROI, and return the elevation DataArray.
    """
    ui_queue.put((pid, "status", f"{task_name} | Chargement MNT..."))
    raw_extent = config.get("analysis_roi_bbox_dict", {})
    roi = {
        'min_lon': min(raw_extent.get('lon', [])),
        'max_lon': max(raw_extent.get('lon', [])),
        'min_lat': min(raw_extent.get('lat', [])),
        'max_lat': max(raw_extent.get('lat', []))
    }
    try:
        with xr.open_dataset(config["mnt_filepath"], decode_coords='all') as mnt:
            alt_var = config["mnt_alt"]
            lon = config["mnt_lon"]
            lat = config["mnt_lat"]
            lon_slice = slice(roi['min_lon'], roi['max_lon'])
            lat_slice = slice(roi['max_lat'], roi['min_lat'])
            if mnt[lon].values[0] > mnt[lon].values[-1]:
                lon_slice = slice(roi['max_lon'], roi['min_lon'])
            if mnt[lat].values[0] < mnt[lat].values[-1]:
                lat_slice = slice(roi['min_lat'], roi['max_lat'])
            elevation_crop = mnt[alt_var].sel({lon: lon_slice, lat: lat_slice}).load()
        elevation_roi, _ = crop_to_roi(elevation_crop, roi, lon, lat)
    except Exception as e:
        tb_str = traceback.format_exc()
        msg = f"Erreur critique lors du chargement/rognage du MNT : {e}\n\nTraceback:\n{tb_str}"
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        logging.error(f"Échec du chargement du MNT pour {task_name}", exc_info=True)
        return None
    if elevation_roi is None or elevation_roi.size == 0:
        msg = "Le MNT est vide après application de la ROI. La zone est probablement hors de la couverture du MNT."
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        return None
        
    if elevation_roi.rio.crs is None:
        log.info("CRS du MNT non détecté, assignation manuelle de l'EPSG:4979 (WGS 84).")
        elevation_roi = elevation_roi.rio.write_crs("EPSG:4979", inplace=True)
    
    return elevation_roi


def compute_connected_inundation(elevation: xr.DataArray, water_level: float, perm_water: xr.DataArray, 
                                 depression: float = 0.1, margin: float = 0.15) -> xr.DataArray:
    elev = elevation.data

    # Masques
    sure_water = elev < (water_level - margin)
    sure_land = elev >= (water_level + margin)

    # Zone incertaine
    uncertain = ~(sure_water | sure_land)

    # Sources d’eau connectée
    seeds = perm_water.data | (elev < (water_level - depression))

    # Propagation 
    propagated = binary_dilation(seeds, structure=np.ones((3,3))) & uncertain

    # Masque final
    water_mask = sure_water | propagated
    mask = np.full(elev.shape, 2, dtype=np.int8)
    mask[sure_land] = 0
    mask[water_mask] = 1

    return xr.DataArray(mask, coords=elevation.coords, dims=elevation.dims, 
                        name="inundation_mask_multiclass").where(elevation.notnull())


def compute_inundation_map(elevation_roi: xr.DataArray, tide_height: float, config: Dict[str, Any], ui_queue, pid: int, task_name: str) -> xr.DataArray:
    ui_queue.put((pid, "status", f"{task_name} | Calcul inondation..."))
    perm_conf = config.get("perm_water", {})
    min_alt = float(elevation_roi.min(skipna=True).item())
    thresh = min_alt + perm_conf.get("offset", 0.1)
    pwm = create_perm_water(elevation_roi, thresh)
    margin = config.get("inundation_margin", 0.15)
    return compute_connected_inundation(elevation_roi, tide_height, pwm, config.get("depression", 0.1), margin=margin)

