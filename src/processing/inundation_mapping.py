# src/processing/inundation_mapping.py
import xarray as xr
import numpy as np
import scipy.ndimage
import geopandas as gpd
from rasterio.features import rasterize as rio_features_rasterize
import os
import logging
from typing import Dict, Any, Tuple, Optional, List

from . import dem_processing

log = logging.getLogger("rich_app")

def compute_connected_inundation(elevation_da: xr.DataArray, water_level: float, permanent_water_mask: xr.DataArray, depression_depth_for_isolated_source: float = 0.1, uncertainty_margin: float = 0.15) -> xr.DataArray:
    """
    Calcule l'inondation connectée en générant un masque à 3 classes.
    Classe 0: Terre certaine
    Classe 1: Eau certaine (connectée)
    Classe 2: Zone incertaine (autour du niveau d'eau)
    """
    if not isinstance(elevation_da, xr.DataArray):
        raise TypeError("Input 'elevation_da' must be an xarray.DataArray.")

    if elevation_da.size == 0:
        return xr.DataArray(np.full(elevation_da.shape, np.nan, dtype=np.float32), coords=elevation_da.coords, dims=elevation_da.dims, name='inundation_mask_multiclass')

    elevation_np = elevation_da.data
    
    potential_inundation_np = (elevation_np < (water_level + uncertainty_margin)) & ~np.isnan(elevation_np)

    if not np.any(potential_inundation_np):
        return xr.zeros_like(elevation_da, dtype=np.int8).where(elevation_da.notnull())

    if permanent_water_mask.shape != elevation_da.shape or not all(c in permanent_water_mask.coords and permanent_water_mask.coords[c].equals(elevation_da.coords[c]) for c in elevation_da.dims):
        permanent_water_mask_np = permanent_water_mask.reindex_like(elevation_da, method='nearest').fillna(False).data
    else:
        permanent_water_mask_np = permanent_water_mask.data

    labels_np, num_features = scipy.ndimage.label(potential_inundation_np, structure=np.ones((3, 3)))

    if num_features == 0:
         return xr.zeros_like(elevation_da, dtype=np.int8).where(elevation_da.notnull())

    source_labels_list = []
    if np.any(permanent_water_mask_np):
        labels_in_global_pwm = labels_np[permanent_water_mask_np & (labels_np > 0)]
        if labels_in_global_pwm.size > 0:
            source_labels_list.append(np.unique(labels_in_global_pwm))

    min_elevs_per_label = scipy.ndimage.minimum(elevation_np, labels=labels_np, index=np.arange(1, num_features + 1))
    deep_enough_mask = min_elevs_per_label < (water_level - depression_depth_for_isolated_source)
    source_labels_cond2 = np.arange(1, num_features + 1)[deep_enough_mask]
    if source_labels_cond2.size > 0:
        source_labels_list.append(source_labels_cond2)

    if not source_labels_list:
        final_source_labels = np.array([], dtype=int)
    else:
        final_source_labels = np.unique(np.concatenate(source_labels_list))
    
    connected_mask = np.isin(labels_np, final_source_labels)

    final_mask_np = np.full(elevation_np.shape, 2, dtype=np.int8)

    mask_terre = elevation_np >= (water_level + uncertainty_margin)
    final_mask_np[mask_terre] = 0

    mask_eau = (elevation_np < (water_level - uncertainty_margin)) & connected_mask
    final_mask_np[mask_eau] = 1

    return xr.DataArray(final_mask_np, coords=elevation_da.coords, dims=elevation_da.dims, name='inundation_mask_multiclass').where(elevation_da.notnull())

def compute_inundation_map(elevation_roi, tide_height, config, ui_queue, pid, task_name):
    """Calcule la carte d'inondation."""
    ui_queue.put((pid, "status", f"{task_name} | Calcul inondation..."))
    perm_water_config = config["perm_water"] 
    min_alt_mnt = float(elevation_roi.min(skipna=True).item()) 
    permanent_water_thresh = min_alt_mnt + perm_water_config.get("offset", 0.1)
    perm_water_mask = dem_processing.create_permanent_water_mask(elevation_roi, permanent_water_thresh) 
    inundation_margin = config.get("inundation_margin", 0.15)
    inundation_map_raw = compute_connected_inundation(
        elevation_roi, tide_height, perm_water_mask, 
        config.get("depression_depth_for_isolated_source", 0.1),
        uncertainty_margin=inundation_margin
    )
    return inundation_map_raw