# src/processing/swot_processing.py
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from matplotlib.path import Path as MplPath
from typing import Dict, Any, Tuple, Optional, List, Union

DEFAULT_KEY_VARIABLES = [
    "time", "time_tai", "longitude", "latitude",
    "ssh_karin_2", "ssh_karin_uncert",
    "sig0_karin_2", "sig0_karin_uncert",
    "height_cor_xover" # Assurez-vous qu'il est inclus si utilisé par apply_ssh_correction
]

def clean_swot_dataset_variables(data: Union[xr.Dataset, Dict[str, xr.Dataset]], key_variables: List[str] = DEFAULT_KEY_VARIABLES) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
    def _clean_single_ds(ds: xr.Dataset) -> xr.Dataset:
        if not isinstance(ds, xr.Dataset): return ds
        vars_to_drop = [var for var in ds.variables if var not in key_variables and var not in ds.dims]
        return ds.drop_vars(vars_to_drop, errors='ignore')

    if isinstance(data, dict):
        return {group: _clean_single_ds(ds_val) for group, ds_val in data.items()}
    elif isinstance(data, xr.Dataset):
        return _clean_single_ds(data)
    return data

def _normalize_longitude_array(lon_array: Union[xr.DataArray, np.ndarray]) -> Union[xr.DataArray, np.ndarray]:
    return np.where(lon_array > 180, lon_array - 360, lon_array)

def apply_roi_to_swot_dataset(dataset: xr.Dataset, roi_config: Optional[Union[Dict[str, List[float]], List[Tuple[float, float]]]], lon_coord_name: str = "longitude", lat_coord_name: str = "latitude") -> xr.Dataset:
    if roi_config is None: return dataset
    if not (lon_coord_name in dataset and lat_coord_name in dataset):
        print(f"Warning (apply_roi_to_swot_dataset): Coords {lon_coord_name} ou {lat_coord_name} manquantes.")
        return dataset

    lon_data_2d = dataset[lon_coord_name]
    lat_data_2d = dataset[lat_coord_name]

    if not (lon_data_2d.ndim == 2 and lat_data_2d.ndim == 2 and lon_data_2d.shape == lat_data_2d.shape):
        raise ValueError("Coordonnées lon/lat invalides pour ROI SWOT (doivent être 2D et de même shape).")
    if not all(s > 0 for dim_name, s in dataset.sizes.items() if dim_name in lon_data_2d.dims):
        print("Warning (apply_roi_to_swot_dataset): Dimensions de données vides pour ROI.")
        return dataset

    final_mask_2d_np = None
    if isinstance(roi_config, dict) and all(k in roi_config for k in ['lon', 'lat']):
        lon_bounds = np.array(sorted(roi_config['lon']))
        lat_bounds = np.array(sorted(roi_config['lat']))
        lon_norm_2d_np = _normalize_longitude_array(lon_data_2d.values)
        lat_2d_np = lat_data_2d.values
        lat_mask_np = (lat_2d_np >= lat_bounds[0]) & (lat_2d_np <= lat_bounds[1])
        lon_mask_np = (lon_norm_2d_np >= lon_bounds[0]) & (lon_norm_2d_np <= lon_bounds[1])
        final_mask_2d_np = lat_mask_np & lon_mask_np
    elif isinstance(roi_config, list) and len(roi_config) >= 3 and \
         all(isinstance(p, tuple) and len(p) == 2 for p in roi_config):
        lons_norm_np = _normalize_longitude_array(lon_data_2d.values)
        lats_np = lat_data_2d.values
        grid_points = np.column_stack((lons_norm_np.ravel(), lats_np.ravel()))
        valid_grid_points_mask = ~np.isnan(grid_points).any(axis=1)
        valid_grid_points = grid_points[valid_grid_points_mask]
        if valid_grid_points.shape[0] == 0:
            final_mask_2d_np = np.zeros(lon_data_2d.shape, dtype=bool)
        else:
            path = MplPath(roi_config)
            inside_mask_for_valid_points = path.contains_points(valid_grid_points)
            final_mask_flat_np = np.zeros(grid_points.shape[0], dtype=bool)
            final_mask_flat_np[valid_grid_points_mask] = inside_mask_for_valid_points
            final_mask_2d_np = final_mask_flat_np.reshape(lon_data_2d.shape)
    else:
        print("Warning (apply_roi_to_swot_dataset): Config ROI invalide.")
        return dataset

    if final_mask_2d_np is not None:
        mask_da = xr.DataArray(final_mask_2d_np, dims=lon_data_2d.dims, 
                               coords={dim: dataset[dim] for dim in lon_data_2d.dims})
        if not np.any(final_mask_2d_np):
            print("Warning (apply_roi_to_swot_dataset): ROI ne recouvre aucun pixel.")
            return dataset.where(mask_da, drop=False) 
        try:
            return dataset.where(mask_da, drop=True)
        except ValueError as e:
            if "any iteration dimensions are 0" in str(e) or "cannot reshape array of size 0" in str(e):
                print(f"Warning (apply_roi_to_swot_dataset): ValueError avec drop=True ({e}). Tentative sans drop=True.")
                return dataset.where(mask_da, drop=False)
            else: raise e 
    return dataset

def apply_roi_to_swot_data_groups(swot_data_groups: Dict[str, xr.Dataset], roi_config: Optional[Union[Dict[str, List[float]], List[Tuple[float, float]]]], lon_coord_name: str = "longitude", lat_coord_name: str = "latitude") -> Dict[str, xr.Dataset]:
    if roi_config is None: return swot_data_groups
    processed_groups = {}
    for group_name, ds in swot_data_groups.items():
        if not isinstance(ds, xr.Dataset):
            processed_groups[group_name] = ds; continue
        if not (lon_coord_name in ds and lat_coord_name in ds and \
                ds.sizes.get('num_lines', 0) > 0 and ds.sizes.get('num_pixels', 0) > 0):
            print(f"Warning (apply_roi_to_swot_data_groups): Données ou dims manquantes pour ROI sur groupe {group_name}.")
            processed_groups[group_name] = ds 
            continue
        processed_ds = apply_roi_to_swot_dataset(ds, roi_config, lon_coord_name, lat_coord_name)
        processed_groups[group_name] = processed_ds
    return processed_groups

def apply_ssh_correction(unsmoothed_groups: Dict[str, xr.Dataset], expert_data: Optional[xr.Dataset]) -> Dict[str, xr.Dataset]:
    corrected_groups = {k: v.copy(deep=True) for k, v in unsmoothed_groups.items() if isinstance(v, xr.Dataset)}
    
    def _assign_original_as_corrected(ds_group_to_modify):
        if isinstance(ds_group_to_modify, xr.Dataset) and "ssh_karin_2" in ds_group_to_modify:
            if "ssh_karin_2_corrected" not in ds_group_to_modify: # Assign only if not already present
                 ds_group_to_modify["ssh_karin_2_corrected"] = ds_group_to_modify["ssh_karin_2"]
                 print(f"Info (apply_ssh_correction): 'ssh_karin_2_corrected' assignée comme 'ssh_karin_2' (pas de correction appliquée).")


    if expert_data is None:
        print("Warning (apply_ssh_correction): Données Expert non fournies. Aucune correction SSH appliquée.")
        for group_ds_val in corrected_groups.values(): _assign_original_as_corrected(group_ds_val)
        return corrected_groups

    expert_vars_ok = all(v in expert_data for v in ["longitude", "latitude", "height_cor_xover"])
    
    if not expert_vars_ok:
        print("Warning (apply_ssh_correction): Variables requises manquantes dans les données Expert. Aucune correction SSH appliquée.")
        for group_ds_val in corrected_groups.values(): _assign_original_as_corrected(group_ds_val)
        return corrected_groups

    lon_expert_flat = _normalize_longitude_array(expert_data.longitude.values).flatten()
    lat_expert_flat = expert_data.latitude.values.flatten()
    correction_val_flat = expert_data['height_cor_xover'].values.flatten()
    valid_indices = ~np.isnan(lon_expert_flat) & ~np.isnan(lat_expert_flat) & ~np.isnan(correction_val_flat)
    
    if not np.any(valid_indices):
        print("Warning (apply_ssh_correction): Aucune donnée de correction valide dans les données Expert. Aucune correction SSH appliquée.")
        for group_ds_val in corrected_groups.values(): _assign_original_as_corrected(group_ds_val)
        return corrected_groups
        
    points_expert_for_interp = np.vstack((lon_expert_flat[valid_indices], lat_expert_flat[valid_indices])).T
    values_expert_for_interp = correction_val_flat[valid_indices]

    for group_name, ds_unsmoothed_in_loop in corrected_groups.items():
        required_vars_in_unsmoothed = ["longitude", "latitude", "ssh_karin_2"]
        if not isinstance(ds_unsmoothed_in_loop, xr.Dataset) or \
           not all(v_name in ds_unsmoothed_in_loop for v_name in required_vars_in_unsmoothed):
            _assign_original_as_corrected(ds_unsmoothed_in_loop)
            continue
        
        lon_unsmoothed_flat = _normalize_longitude_array(ds_unsmoothed_in_loop.longitude.values).flatten()
        lat_unsmoothed_flat = ds_unsmoothed_in_loop.latitude.values.flatten()
        points_to_interpolate_at = np.vstack((lon_unsmoothed_flat, lat_unsmoothed_flat)).T
        
        interpolated_corrections = griddata(
            points_expert_for_interp, values_expert_for_interp, 
            points_to_interpolate_at, method='linear', fill_value=0.0 
        )
        correction_grid = np.reshape(interpolated_corrections, ds_unsmoothed_in_loop.longitude.shape)
        ds_unsmoothed_in_loop["ssh_karin_2_corrected"] = ds_unsmoothed_in_loop["ssh_karin_2"] + correction_grid
        print(f"Info (apply_ssh_correction): Correction SSH appliquée au groupe {group_name}.")
    return corrected_groups

def compute_ssh_statistics(dataset_group: xr.Dataset) -> Optional[Dict[str, Dict[str, float]]]:
    if not (isinstance(dataset_group, xr.Dataset) and \
            "ssh_karin_2" in dataset_group and \
            "ssh_karin_2_corrected" in dataset_group):
        return None

    stats: Dict[str, Dict[str, float]] = {}
    ssh_o_data = dataset_group["ssh_karin_2"].values
    ssh_c_data = dataset_group["ssh_karin_2_corrected"].values

    if np.any(~np.isnan(ssh_o_data)):
        stats["original"] = {"min":float(np.nanmin(ssh_o_data)), "max":float(np.nanmax(ssh_o_data)),
                             "mean":float(np.nanmean(ssh_o_data)), "std":float(np.nanstd(ssh_o_data))}
    if np.any(~np.isnan(ssh_c_data)):
        stats["corrected"] = {"min":float(np.nanmin(ssh_c_data)), "max":float(np.nanmax(ssh_c_data)),
                              "mean":float(np.nanmean(ssh_c_data)), "std":float(np.nanstd(ssh_c_data))}

    if "original" in stats and "corrected" in stats:
        ssh_d_data = ssh_c_data - ssh_o_data
        if np.any(~np.isnan(ssh_d_data)):
            stats["difference"] = {"min":float(np.nanmin(ssh_d_data)), "max":float(np.nanmax(ssh_d_data)),
                                   "mean":float(np.nanmean(ssh_d_data)), "std":float(np.nanstd(ssh_d_data))}
            total_valid_diff_pixels = np.sum(~np.isnan(ssh_d_data))
            if total_valid_diff_pixels > 0:
                significant_corrections = np.sum(np.abs(ssh_d_data[~np.isnan(ssh_d_data)]) > 0.001)
                stats["difference"]["percent_corrected_significantly"] = \
                    float((significant_corrections / total_valid_diff_pixels) * 100)
    return stats if stats else None