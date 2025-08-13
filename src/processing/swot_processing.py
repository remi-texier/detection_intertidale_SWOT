# src/processing/swot_processing.py
import logging
import os
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from matplotlib.path import Path as MplPath
from typing import Dict, Any, Tuple, Optional, List, Union
from datetime import datetime
import pandas as pd
import rioxarray as rxr
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
import pyproj
from scipy.spatial import cKDTree
from rasterio.enums import Resampling
import traceback


from .. import data_loader

log = logging.getLogger("rich_app")


LR250M_VARIABLES = ["time", "longitude", "latitude", "ssh_karin_2", "ssh_karin_qual", "height_cor_xover", "sig0_karin_2", "sig0_karin_qual"]
HR100M_VARIABLES = ["illumination_time", "longitude", "latitude", "wse", "geoid", "wse_qual", "height_cor_xover", "sig0", "sig0_qual"]

def is_hr_dataset(dataset: xr.Dataset) -> bool:
    if isinstance(dataset, dict):
        for ds in dataset.values():
            if isinstance(ds, xr.Dataset):
                return "illumination_time" in ds.variables or "HR_Raster" in str(ds.attrs.get("source", ""))
        return False
    elif isinstance(dataset, xr.Dataset):
        return "illumination_time" in dataset.variables or "HR_Raster" in str(dataset.attrs.get("source", ""))
    return False

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

    if not (lon_coord_name in dataset.coords or lon_coord_name in dataset.variables):
        log.warning(f"Coord/Variable '{lon_coord_name}' manquante.")
        return dataset
    if not (lat_coord_name in dataset.coords or lat_coord_name in dataset.variables):
        log.warning(f"Coord/Variable '{lat_coord_name}' manquante.")
        return dataset

    if isinstance(roi_config, dict) and all(k in roi_config for k in ['lon', 'lat']):
        lon_bounds = sorted(roi_config['lon'])
        lat_bounds = sorted(roi_config['lat'])

        lon_norm = _normalize_longitude_array(dataset[lon_coord_name])
        mask = ((dataset[lat_coord_name] >= lat_bounds[0]) & (dataset[lat_coord_name] <= lat_bounds[1]) & (lon_norm >= lon_bounds[0]) & (lon_norm <= lon_bounds[1]))
        
        if 'y' in dataset.dims and 'x' in dataset.dims:
            valid_y = mask.any(dim='x')
            valid_x = mask.any(dim='y')
            
            if not valid_y.any() or not valid_x.any():
                return dataset.isel(y=slice(0), x=slice(0))

            subset = dataset.sel(y=valid_y, x=valid_x)
            
            result_vars = {}
            for var_name, var in subset.data_vars.items():
                if np.issubdtype(var.dtype, np.number):
                    result_vars[var_name] = var.where(mask.sel(y=valid_y, x=valid_x))
                else:
                    result_vars[var_name] = var  
            
            return xr.Dataset(result_vars, coords=subset.coords, attrs=subset.attrs)
        else:
            valid_lines = mask.any(dim='num_pixels')
            valid_pixels = mask.any(dim='num_lines')
            
            if not valid_lines.any() or not valid_pixels.any():
                return dataset.isel(num_lines=slice(0), num_pixels=slice(0))

            subset = dataset.sel(num_lines=valid_lines, num_pixels=valid_pixels)
            
            result_vars = {}
            for var_name, var in subset.data_vars.items():
                if np.issubdtype(var.dtype, np.number):
                    result_vars[var_name] = var.where(mask.sel(num_lines=valid_lines, num_pixels=valid_pixels))
                else:
                    result_vars[var_name] = var 
            
            return xr.Dataset(result_vars, coords=subset.coords, attrs=subset.attrs)
    else:
        log.warning("Config ROI invalide.")
        return dataset

def apply_roi_to_swot_data_groups(swot_groups: Dict[str, xr.Dataset], roi_config: Dict[str, List[float]], lon_coord_name: str = "longitude", lat_coord_name: str = "latitude") -> Dict[str, xr.Dataset]:
    if roi_config is None: return swot_groups
    processed = {}
    for group_name, ds in swot_groups.items():
        if not isinstance(ds, xr.Dataset):
            processed[group_name] = ds; continue
        
        has_lr_dims = ds.sizes.get('num_lines', 0) > 0 and ds.sizes.get('num_pixels', 0) > 0
        has_hr_dims = ds.sizes.get('y', 0) > 0 and ds.sizes.get('x', 0) > 0
        
        has_coords = ((lon_coord_name in ds.coords or lon_coord_name in ds.variables) and 
                     (lat_coord_name in ds.coords or lat_coord_name in ds.variables))
        
        if not (has_coords and (has_lr_dims or has_hr_dims)):
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

    lon_expert = _normalize_longitude_array(expert_data.longitude).values.flatten()
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
        
        lon_unsmoothed = _normalize_longitude_array(ds.longitude).values.flatten()
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
    
    is_hr_data = config.get("data_type") == "HR"
    main_data, expert_data, source_file = None, None, None

    try:
        if is_hr_data:
            ui_queue.put((pid, "log", "Traitement de données HR."))
            tile_files = data_loader.find_hr_tiles(config["data_path"], cycle, pass_id)
            if not tile_files:
                msg = f"Aucune tuile HR trouvée pour {task_name}."
                report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
                return None, None
            filtered_datasets = []
            for i, hr_file in enumerate(tile_files):
                try:
                    md = data_loader.read_swot_datafile(hr_file, is_expert=False)
                    if not md or "main" not in md:
                        continue
                    ds = md["main"]
                    ds = clean_swot_dataset_variables(ds, HR100M_VARIABLES)
                    ds = apply_roi_to_swot_dataset(ds, config.get("analysis_roi_bbox_dict"))
                    if not isinstance(ds, xr.Dataset) or not (ds.sizes.get('y', 0) > 0 or ds.sizes.get('num_lines', 0) > 0):
                        continue
                    ds = apply_quality_flags(ds, config)
                    if "wse" in ds and "geoid" in ds and "ssh" not in ds:
                        ds["ssh"] = ds["wse"] + ds["geoid"]
                    filtered_datasets.append(ds)
                except Exception as e:
                    report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': f"Tuile HR ignorée ({os.path.basename(hr_file)}): {e}"})
                    continue
            if not filtered_datasets:
                msg = f"Aucune donnée SWOT HR valide après cropping/qualité pour {task_name}."
                report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
                return None, None
            primary_ds = filtered_datasets[0]
            if len(filtered_datasets) > 1:
                primary_ds.attrs['extra_swot_datasets'] = filtered_datasets[1:]
            source_file = ";".join(os.path.basename(f) for f in tile_files)
            return primary_ds, source_file
        else:
            ui_queue.put((pid, "log", "Traitement de données LR."))
            expert_file = data_loader.find_swot_files(config["data_path"], cycle, pass_id, "Expert")
            unsmoothed_file = data_loader.find_swot_files(config["data_path"], cycle, pass_id, "Unsmoothed")
            
            if not unsmoothed_file:
                msg = f"Fichier SWOT LR Unsmoothed manquant pour {task_name}. Traitement abandonné."
                report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
                return None, None
            
            main_data = data_loader.read_swot_datafile(unsmoothed_file, is_expert=False)
            expert_data = data_loader.read_swot_datafile(expert_file, is_expert=True) if expert_file else None
            source_file = unsmoothed_file
    except Exception as e:
        tb_str = traceback.format_exc()
        msg = f"Erreur lors du chargement des données SWOT pour {task_name}: {e}\n{tb_str}"
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        return None, None


    if not main_data or not any(isinstance(ds, xr.Dataset) for ds in (main_data.values() if isinstance(main_data, dict) else [main_data])):
        msg = f"Aucune donnée SWOT n'a pu être chargée pour {task_name}."
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        return None, None

    ui_queue.put((pid, "status", f"{task_name} | Traitement SWOT..."))

    if isinstance(main_data, dict):
        filtered_groups = {k: v for k, v in main_data.items() if isinstance(v, xr.Dataset)}
    else:
        filtered_groups = {"main": main_data}

    if not is_hr_data and (allowed := zone_data.get("data_group")):
        filtered_groups = {gn: ds for gn, ds in filtered_groups.items() if gn in allowed}

    if not filtered_groups:
        msg = f"Aucun groupe de données ne correspond à la config pour {task_name}."
        report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': msg})
        return None, None

    key_variables = HR100M_VARIABLES if is_hr_data else LR250M_VARIABLES
    cleaned_groups = clean_swot_dataset_variables(filtered_groups, key_variables)
    roi_groups = apply_roi_to_swot_data_groups(cleaned_groups, config["analysis_roi_bbox_dict"])
    
    final_groups = {}
    for group_name, ds in roi_groups.items():
        if not isinstance(ds, xr.Dataset) or not (ds.sizes.get('y', 0) > 0 or ds.sizes.get('num_lines', 0) > 0):
            continue

        ds_filtered = apply_quality_flags(ds, config)

        if is_hr_data:
            if "wse" in ds_filtered and "geoid" in ds_filtered:
                ds_filtered["ssh"] = ds_filtered["wse"] + ds_filtered["geoid"]
                final_groups[group_name] = ds_filtered
        else: # Cas LR
            expert_roi = apply_roi_to_swot_dataset(expert_data, config["analysis_roi_bbox_dict"]) if expert_data else None
            corrected_ds = apply_ssh_correction(ds_filtered, expert_roi)
            final_groups[group_name] = corrected_ds
            
    valid_groups = {k: v for k, v in final_groups.items() if v.notnull().any()}
    if not valid_groups:
        msg = f"Aucune donnée SWOT valide après filtrage qualité et ROI pour {task_name}."
        report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': msg})
        return None, None

    display_group_name = next(iter(valid_groups.keys()), None)
    if not is_hr_data and (allowed := zone_data.get("data_group")):
        display_group_name = next((g for g in allowed if g in valid_groups), display_group_name)

    if not display_group_name:
        msg = f"Aucun groupe SWOT valide après traitement pour {task_name}."
        report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': msg})
        return None, None
        
    ui_queue.put((pid, "log", f"Traitement SWOT pour le groupe '{display_group_name}' terminé."))
    
    return valid_groups[display_group_name], source_file

def process_swot_orientation_and_time(swot_data, ui_queue, report_queue, task_name, pid):
    is_hr = is_hr_dataset(swot_data)
    
    if is_hr:
        if "ssh" not in swot_data and not ("wse" in swot_data and "geoid" in swot_data):
            msg = f"Les variables requises pour SSH ('ssh' ou 'wse'+'geoid') n'ont pas été trouvées dans le dataset SWOT HR."
            report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
            return None, None
    else:
        if "ssh_karin_2_corrected" not in swot_data: 
            msg = f"La variable requise 'ssh_karin_2_corrected' n'a pas été trouvée dans le dataset SWOT LR."
            report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
            return None, None
    
    if 'latitude' in swot_data:
        try:
            if is_hr and 'y' in swot_data.dims and swot_data.sizes['y'] > 1:
                lat_col = swot_data['latitude'].isel(x=swot_data.sizes['x'] // 2)
                valid_lats = lat_col.dropna(dim='y', how='all')
                if valid_lats.size > 1 and valid_lats.isel(y=0).item() > valid_lats.isel(y=-1).item():
                    ui_queue.put((pid, "log", "Trace N->S (HR), inversion..."))
                    swot_data = swot_data.isel(y=slice(None, None, -1))
            elif not is_hr and 'num_lines' in swot_data.dims and swot_data.sizes['num_lines'] > 1:
                lat_col = swot_data['latitude'].isel(num_pixels=swot_data.sizes['num_pixels'] // 2)
                valid_lats = lat_col.dropna(dim='num_lines', how='all')
                if valid_lats.size > 1 and valid_lats.isel(num_lines=0).item() > valid_lats.isel(num_lines=-1).item():
                    ui_queue.put((pid, "log", "Trace N->S (LR), inversion..."))
                    swot_data = swot_data.isel(num_lines=slice(None, None, -1))
        except (IndexError, ValueError) as e:
            report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': f"Orientation SWOT non déterminée. Erreur: {e}"})

    if is_hr:
        time_array = swot_data["illumination_time"] 
        swot_time, is_fallback = datetime.now(), True
        if time_array.size > 0: 
            try:
                time_values = time_array.values.flatten()
                
                if time_values.dtype.kind == 'M':  
                    valid_times = time_values[~np.isnat(time_values)]
                else:  
                    valid_times = time_values[~np.isnan(time_values)]
                
                if valid_times.size > 0:
                    if time_values.dtype.kind == 'M':  
                        median_ns = np.median(valid_times.astype('datetime64[ns]').astype(np.int64))
                        swot_time = pd.to_datetime(median_ns, unit='ns')
                    else:  
                        median_seconds = np.median(valid_times)
                        epoch_2000 = pd.Timestamp('2000-01-01 00:00:00')
                        swot_time = epoch_2000 + pd.Timedelta(seconds=median_seconds)
                    is_fallback = False
            except Exception as e:
                report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': f"Erreur calcul temps HR: {e}"})
    else:
        time_array = swot_data["time"] 
        swot_time, is_fallback = datetime.now(), True
        if time_array.size > 0: 
            valid_times = time_array.values.flatten()[~np.isnat(time_array.values.flatten())]
            if valid_times.size > 0: 
                median_ns = np.median(valid_times.astype('datetime64[ns]').astype(np.int64)) 
                swot_time = pd.to_datetime(median_ns, unit='ns') 
                is_fallback = False 

    ui_queue.put((pid, "log", f"Heure SWOT ({'HR' if is_hr else 'LR'}): {swot_time.strftime('%H:%M:%S')}{' (Fallback)' if is_fallback else ''}"))

    return swot_data, (swot_time, is_fallback)

def build_target_grid_from_swot(swot_data: xr.Dataset, extent: Dict[str, Any], config: dict) -> xr.DataArray:
    """
    Build a regular lon/lat grid (EPSG:4326) using SWOT native resolution and provided extent.
    The grid spacing is inferred from the median spacing of SWOT coordinates along each axis.
    Returns an xr.DataArray filled with NaNs with dims (lat, lon), CRS=EPSG:4326.
    """
    lon_name = config.get("mnt_lon", "lon")
    lat_name = config.get("mnt_lat", "lat")

    is_hr = is_hr_dataset(swot_data)
    lon_src = _normalize_longitude_array(swot_data["longitude"]).values
    lat_src = swot_data["latitude"].values

    try:
        if is_hr and "y" in swot_data.dims and "x" in swot_data.dims:
            ysize, xsize = swot_data.sizes["y"], swot_data.sizes["x"]
            x_center = max(0, xsize // 2)
            y_center = max(0, ysize // 2)
            lat_line = lat_src[:, x_center]
            lon_line = lon_src[y_center, :]
        elif ("num_lines" in swot_data.dims) and ("num_pixels" in swot_data.dims):
            lsize, psize = swot_data.sizes["num_lines"], swot_data.sizes["num_pixels"]
            p_center = max(0, psize // 2)
            l_center = max(0, lsize // 2)
            lat_line = lat_src[:, p_center]
            lon_line = lon_src[l_center, :]
        else:
            # Fallback: flatten and approximate
            lat_line = lat_src.flatten()
            lon_line = lon_src.flatten()

        # Compute median step (abs) ignoring NaNs
        def _median_step(arr):
            arr_valid = arr[np.isfinite(arr)]
            if arr_valid.size < 2:
                return np.nan
            diffs = np.diff(np.sort(arr_valid))
            diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
            return np.nanmedian(np.abs(diffs)) if diffs.size > 0 else np.nan

        dlat = _median_step(lat_line)
        dlon = _median_step(lon_line)

        # Bounds from extent
        min_lon, max_lon = sorted(extent.get("lon", [np.nan, np.nan]))
        min_lat, max_lat = sorted(extent.get("lat", [np.nan, np.nan]))
        if not all(np.isfinite([min_lon, max_lon, min_lat, max_lat])):
            raise ValueError("Extent invalide pour construire la grille")

        # If step undefined, infer from span and counts
        if not np.isfinite(dlat) or dlat <= 0:
            # approximate number of rows from source if available
            n_lat = lat_src.shape[0] if lat_src.ndim >= 1 else 256
            dlat = max((max_lat - min_lat) / max(n_lat - 1, 1), 1e-5)
        if not np.isfinite(dlon) or dlon <= 0:
            n_lon = lon_src.shape[-1] if lon_src.ndim >= 1 else 256
            dlon = max((max_lon - min_lon) / max(n_lon - 1, 1), 1e-5)

        lats = np.arange(min_lat, max_lat + dlat * 0.5, dlat)
        lons = np.arange(min_lon, max_lon + dlon * 0.5, dlon)
        if lats.size < 2:
            lats = np.linspace(min_lat, max_lat, 2)
        if lons.size < 2:
            lons = np.linspace(min_lon, max_lon, 2)

        da = xr.DataArray(
            data=np.full((lats.size, lons.size), np.nan, dtype=float),
            coords={lat_name: lats, lon_name: lons},
            dims=(lat_name, lon_name),
            name="target_grid"
        )
        # Set EPSG:4326 and spatial dims
        da = da.rio.write_crs("EPSG:4326", inplace=True)
        da = da.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=True)
        return da
    except Exception:
        log.error("Erreur lors de la construction de la grille SWOT cible", exc_info=True)
        # Minimal grid as last resort
        da = xr.DataArray(
            data=np.full((2, 2), np.nan, dtype=float),
            coords={lat_name: [extent['lat'][0], extent['lat'][1]], lon_name: [extent['lon'][0], extent['lon'][1]]},
            dims=(lat_name, lon_name),
            name="target_grid"
        )
        da = da.rio.write_crs("EPSG:4326", inplace=True)
        da = da.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=True)
        return da

def _infer_utm_epsg_from_coords(lons: np.ndarray, lats: np.ndarray) -> Optional[str]:
    try:
        lons_valid = lons[np.isfinite(lons)]
        lats_valid = lats[np.isfinite(lats)]
        if lons_valid.size == 0 or lats_valid.size == 0:
            return None
        lon_med = float(np.median(lons_valid))
        lat_med = float(np.median(lats_valid))
        zone = int(np.floor((lon_med + 180) / 6) + 1)
        zone = max(1, min(60, zone))
        epsg_num = (32600 if lat_med >= 0 else 32700) + zone
        return f"EPSG:{epsg_num}"
    except Exception:
        return None

def rasterize_swot_data(swot_data: xr.Dataset, target_grid: xr.DataArray, config: dict) -> Dict[str, xr.DataArray]:
    result = {}
    
    target_lon = target_grid[config["mnt_lon"]].values
    target_lat = target_grid[config["mnt_lat"]].values
    lon_mesh, lat_mesh = np.meshgrid(target_lon, target_lat, indexing='xy')
    
    src_lons_raw = _normalize_longitude_array(swot_data.longitude).values.flatten()
    src_lats_raw = swot_data.latitude.values.flatten()

    # Inclure d'autres tuiles si présentes (évite traitement par tuile)
    extra_list = swot_data.attrs.get('extra_swot_datasets')
    if isinstance(extra_list, list) and extra_list:
        lons_all = [src_lons_raw]
        lats_all = [src_lats_raw]
        sig0_all = []
        is_hr = is_hr_dataset(swot_data)
        sig0_name = 'sig0' if is_hr else 'sig0_karin_2'
        if sig0_name in swot_data:
            sig0_all.append(swot_data[sig0_name].values.flatten())
        for extra in extra_list:
            try:
                lons_all.append(_normalize_longitude_array(extra.longitude).values.flatten())
                lats_all.append(extra.latitude.values.flatten())
                if sig0_name in extra:
                    sig0_all.append(extra[sig0_name].values.flatten())
            except Exception:
                continue
        src_lons_raw = np.concatenate(lons_all, axis=0)
        src_lats_raw = np.concatenate(lats_all, axis=0)
        if sig0_all:
            swot_data = swot_data.copy()
            # Concat sig0 for joint filtering; store temporarily in attrs to avoid duplicating arrays in Dataset
            swot_data.attrs['__multi_sig0__'] = np.concatenate(sig0_all, axis=0)

    # Determine projected CRS to compute distances (prefer source file EPSG, else infer from coords)
    source_epsg = None
    if hasattr(swot_data, 'attrs'):
        source_epsg = swot_data.attrs.get('source_epsg')
    if not source_epsg:
        source_epsg = _infer_utm_epsg_from_coords(src_lons_raw, src_lats_raw) or "EPSG:32630"

    is_hr = is_hr_dataset(swot_data)
    
    if is_hr:
        vars_to_rasterize = {
            'swot_ssh': {'src_var': 'ssh', 'units': 'm'},
            'swot_sig0': {'src_var': 'sig0', 'units': 'linear'}
        }
        if 'ssh' not in swot_data and 'wse' in swot_data and 'geoid' in swot_data:
            swot_data['ssh'] = swot_data['wse'] + swot_data['geoid']
    else:
        vars_to_rasterize = {
            'swot_ssh': {'src_var': 'ssh_karin_2_corrected', 'units': 'm'},
            'swot_sig0': {'src_var': 'sig0_karin_2', 'units': 'linear'}
        }

    valid_coords_mask = ~np.isnan(src_lons_raw) & ~np.isnan(src_lats_raw)

    # Threshold on sig0 > 1000 (linear), mask these source samples for all variables
    sig0_src_name = 'sig0' if is_hr else 'sig0_karin_2'
    if '__multi_sig0__' in swot_data.attrs:
        sig0_vals = swot_data.attrs.pop('__multi_sig0__')
        valid_coords_mask &= ~(np.isfinite(sig0_vals) & (sig0_vals > 1000.0))
    elif sig0_src_name in swot_data:
        sig0_vals = swot_data[sig0_src_name].values.flatten()
        valid_coords_mask &= ~(np.isfinite(sig0_vals) & (sig0_vals > 1000.0))
    else:
        log.warning(f"Variable sig0 non trouvée dans le dataset pour le masquage des coordonnées.")

    src_lons_valid = src_lons_raw[valid_coords_mask]
    src_lats_valid = src_lats_raw[valid_coords_mask]

    if src_lons_valid.size == 0:
        log.warning("Aucune coordonnée SWOT valide trouvée pour la rastérisation après filtrage sig0.")
        return {}
    
    transformer = pyproj.Transformer.from_crs("EPSG:4326", source_epsg, always_xy=True)
    src_x_proj, src_y_proj = transformer.transform(src_lons_valid, src_lats_valid)
    src_points_proj = np.vstack((src_x_proj, src_y_proj)).T

    log.info(f"Construction de l'arbre cKDTree avec {len(src_points_proj)} points SWOT (proj: {source_epsg}).")
    kdtree = cKDTree(src_points_proj)

    target_grid_proj = target_grid.rio.reproject(source_epsg)
    target_x_mesh, target_y_mesh = np.meshgrid(target_grid_proj.x.values, target_grid_proj.y.values, indexing='xy')
    target_points_proj = np.vstack((target_x_mesh.ravel(), target_y_mesh.ravel())).T
    
    log.info("Calcul des distances sur la grille projetée...")
    distances, _ = kdtree.query(target_points_proj, k=1)
    
    distances_proj_da = xr.DataArray(
        data=distances.reshape(target_x_mesh.shape),
        coords=target_grid_proj.coords,
        dims=target_grid_proj.dims
    )
    distances_proj_da.rio.write_crs(target_grid_proj.rio.crs, inplace=True)
    
    log.info("Reprojection du masque de distance pour correspondre à la grille cible (EPSG:4326)...")
    distances_matched_da = distances_proj_da.rio.reproject_match(target_grid, resampling=Resampling.bilinear)

    max_dist = config.get("interpolation_max_distance", 250)
    distance_mask = distances_matched_da.values <= max_dist

    # Interpolate each variable using only valid source samples
    for out_name, params in vars_to_rasterize.items():
        src_var = params['src_var']
        if src_var not in swot_data:
            log.warning(f"Variable source '{src_var}' non trouvée dans le dataset, ignorée.")
            continue
        
        src_values_full = swot_data[src_var].values.flatten()
        if isinstance(extra_list, list) and extra_list and src_var in ['ssh', 'sig0']:
            # Concat en mémoire pour ces deux variables uniquement (limite RAM)
            extras_flat = []
            for extra in extra_list:
                if src_var in extra:
                    try:
                        extras_flat.append(extra[src_var].values.flatten())
                    except Exception:
                        pass
            if extras_flat:
                src_values_full = np.concatenate([src_values_full] + extras_flat, axis=0)
        src_values = src_values_full[valid_coords_mask]
        
        points = np.vstack((src_lons_valid, src_lats_valid)).T
        if src_values.size == 0 or points.size == 0:
            log.warning(f"Aucune donnée valide pour l'interpolation de '{src_var}'.")
            continue

        interpolated_full = griddata(
            points=points,
            values=src_values,
            xi=(lon_mesh, lat_mesh),
            method='linear'
        )

        interpolated_masked = np.where(distance_mask, interpolated_full, np.nan)
        da_out = xr.DataArray(
            data=interpolated_masked,
            coords=target_grid.coords,
            dims=target_grid.dims,
            name=out_name,
            attrs={'units': params['units']}
        )
        if target_grid.rio.crs is not None:
            da_out = da_out.rio.write_crs(target_grid.rio.crs, inplace=True)
            da_out = da_out.rio.set_spatial_dims(x_dim=config["mnt_lon"], y_dim=config["mnt_lat"], inplace=True)
        result[out_name] = da_out

    return result

def apply_quality_flags(swot_data: xr.Dataset, config: dict) -> xr.Dataset:
    quality_config = config.get("swot_quality_filter", {})
    if not quality_config.get("apply", False):
        log.info("Filtrage par flag de qualité désactivé dans la configuration.")
        return swot_data

    accepted_values = quality_config.get("accepted_values", [0, 1])
    variable_map = quality_config.get("variable_map", {})
    
    log.info(f"Construction du masque de qualité. Flags acceptés : {accepted_values}")
    
    ref_var_name = next((v for v in variable_map.values() if v in swot_data), None)
    if not ref_var_name:
        log.warning("Aucune variable de qualité trouvée pour créer le masque. Pas de filtrage.")
        return swot_data
        
    combined_mask = xr.ones_like(swot_data[ref_var_name], dtype=bool)

    for data_var, qual_var in variable_map.items():
        if data_var in swot_data and qual_var in swot_data:
            log.info(f" - Intégration du flag '{qual_var}' au masque global.")
            combined_mask &= swot_data[qual_var].isin(accepted_values)
    
    log.info("Application du masque de qualité global au dataset SWOT.")
    return swot_data.where(combined_mask)