# src/processing/dem_processing.py
import xarray as xr
import numpy as np
from matplotlib.path import Path as MplPath
from typing import Dict, Any, Tuple, Optional, List

def get_extent(data_array: xr.DataArray, lon_coord: str, lat_coord: str) -> Optional[Dict[str, float]]:
    if not (lon_coord in data_array.coords and lat_coord in data_array.coords and 
            data_array[lon_coord].size > 0 and data_array[lat_coord].size > 0):
        print(f"Warning (get_extent): Invalid coordinates ('{lon_coord}', '{lat_coord}') or empty coordinate arrays.")
        return None
    return {
        'min_lon': float(data_array[lon_coord].min()), 'max_lon': float(data_array[lon_coord].max()),
        'min_lat': float(data_array[lat_coord].min()), 'max_lat': float(data_array[lat_coord].max())
    }

def crop_to_roi(data_array: xr.DataArray, roi_config: Optional[Any], lon_coord: str, lat_coord: str) -> Tuple[xr.DataArray, bool]:
    if roi_config is None:
        return data_array, False

    is_roi_applied = False
    if isinstance(roi_config, dict) and all(k in roi_config for k in ['min_lon', 'min_lat', 'max_lon', 'max_lat']):
        print(f"Info (crop_to_roi): Applying axis-aligned ROI (BBox).")
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
            print(f"Warning (crop_to_roi): Could not select axis-aligned ROI. Coords '{lon_coord}' or '{lat_coord}' missing/invalid.")
            return data_array, False
            
    elif isinstance(roi_config, list) and len(roi_config) >= 3 and all(isinstance(p, tuple) and len(p) == 2 for p in roi_config):
        print(f"Info (crop_to_roi): Applying custom polygon ROI with {len(roi_config)} vertices.")
        poly_lons = [p[0] for p in roi_config]
        poly_lats = [p[1] for p in roi_config]
        bbox = {'min_lon': min(poly_lons), 'max_lon': max(poly_lons), 'min_lat': min(poly_lats), 'max_lat': max(poly_lats)}
        print(f"  Info (crop_to_roi): Polygon BBox: Lon ({bbox['min_lon']:.4f}-{bbox['max_lon']:.4f}), Lat ({bbox['min_lat']:.4f}-{bbox['max_lat']:.4f})")
        
        try:
            lon_slice_bbox = slice(bbox['min_lon'], bbox['max_lon'])
            if data_array[lon_coord].size > 1 and data_array[lon_coord][0] > data_array[lon_coord][-1]:
                lon_slice_bbox = slice(bbox['max_lon'], bbox['min_lon'])
            
            lat_slice_bbox = slice(bbox['min_lat'], bbox['max_lat'])
            if data_array[lat_coord].size > 1 and data_array[lat_coord][0] > data_array[lat_coord][-1]:
                lat_slice_bbox = slice(bbox['max_lat'], bbox['min_lat'])

            sub_array = data_array.sel({lon_coord: lon_slice_bbox, lat_coord: lat_slice_bbox})
        except KeyError:
            print(f"Warning (crop_to_roi): Could not select BBox for custom ROI. Coords '{lon_coord}' or '{lat_coord}' missing/invalid.")
            return data_array, False

        if sub_array.size == 0:
            print("Warning (crop_to_roi): BBox of custom ROI resulted in an empty slice.")
            return sub_array, True 

        lon_coords_1d = sub_array[lon_coord].data
        lat_coords_1d = sub_array[lat_coord].data

        if lon_coords_1d.ndim != 1 or lat_coords_1d.ndim != 1:
            print(f"Warning (crop_to_roi): Polygon ROI masking requires 1D coord axes for {lon_coord} (ndim: {lon_coords_1d.ndim}) and {lat_coord} (ndim: {lat_coords_1d.ndim}). Skipping polygon mask, returning BBox crop.")
            return sub_array, True

        if sub_array.dims[0] == lat_coord and sub_array.dims[1] == lon_coord:
            lons_2d, lats_2d = np.meshgrid(lon_coords_1d, lat_coords_1d, indexing='xy')
        elif sub_array.dims[0] == lon_coord and sub_array.dims[1] == lat_coord:
            _lons_tmp, _lats_tmp = np.meshgrid(lon_coords_1d, lat_coords_1d, indexing='xy')
            lons_2d, lats_2d = _lons_tmp.T, _lats_tmp.T
        else:
            print(f"Warning (crop_to_roi): Unexpected dimension order {sub_array.dims}. Assuming first dim is 'y-like' (lat) and second is 'x-like' (lon) for polygon masking.")
            lons_2d, lats_2d = np.meshgrid(lon_coords_1d, lat_coords_1d, indexing='xy')

        if lons_2d.shape != sub_array.shape or lats_2d.shape != sub_array.shape:
            raise ValueError(f"Shape mismatch after meshgrid: sub_array shape: {sub_array.shape}, lons_2d: {lons_2d.shape}, lats_2d: {lats_2d.shape}.")

        grid_points = np.column_stack((lons_2d.ravel(), lats_2d.ravel()))
        path = MplPath(roi_config) 
        mask_flat = path.contains_points(grid_points)
        mask_2d = mask_flat.reshape(sub_array.shape)
        
        mask_da = xr.DataArray(mask_2d, dims=sub_array.dims, coords=sub_array.coords)
        cropped_da = sub_array.where(mask_da)
        is_roi_applied = True
    else:
        print("Warning (crop_to_roi): Invalid ROI configuration. Using full data extent.")
        return data_array, False

    if cropped_da.size == 0: print("Warning (crop_to_roi): ROI resulted in an empty data slice.")
    elif not cropped_da.notnull().any(): print("Warning (crop_to_roi): ROI resulted in all NaN values.")
    
    return cropped_da, is_roi_applied

def create_permanent_water_mask(elevation_da: xr.DataArray, threshold: float) -> xr.DataArray:
    if elevation_da.size == 0:
        return xr.DataArray(np.empty(elevation_da.shape, dtype=bool), 
                            coords=elevation_da.coords, dims=elevation_da.dims, 
                            name="permanent_water_mask")
    return (elevation_da < threshold).fillna(False)