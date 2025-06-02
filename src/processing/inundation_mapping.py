# src/processing/inundation_mapping.py
import xarray as xr
import numpy as np
import scipy.ndimage
import geopandas as gpd
from rasterio.features import rasterize as rio_features_rasterize
import os
from typing import Dict, Any, Tuple, Optional, List

def compute_connected_inundation(
    elevation_da: xr.DataArray,
    water_level: float,
    permanent_water_mask: xr.DataArray,
    depression_depth_for_isolated_source: float = 0.1
) -> xr.DataArray:
    if not isinstance(elevation_da, xr.DataArray):
        raise TypeError("Input 'elevation_da' must be an xarray.DataArray.")
    if not isinstance(permanent_water_mask, xr.DataArray):
        raise TypeError("Input 'permanent_water_mask' must be an xarray.DataArray.")
    if not isinstance(water_level, (int, float)):
        raise TypeError("Input 'water_level' must be a number.")
    if not isinstance(depression_depth_for_isolated_source, (int, float)):
        raise TypeError("Input 'depression_depth_for_isolated_source' must be a number.")

    if elevation_da.size == 0:
        return xr.DataArray(np.full(elevation_da.shape, np.nan, dtype=np.float32),
                            coords=elevation_da.coords, dims=elevation_da.dims,
                            name='connected_inundation')

    elevation_np = elevation_da.data
    potential_inundation_np = (elevation_np < water_level) & ~np.isnan(elevation_np)

    if not np.any(potential_inundation_np):
        return xr.zeros_like(elevation_da, dtype=np.int8).where(elevation_da.notnull())

    if permanent_water_mask.shape != elevation_da.shape or \
       not all(c in permanent_water_mask.coords and permanent_water_mask.coords[c].equals(elevation_da.coords[c]) for c in elevation_da.dims):
        print("Warning (compute_connected_inundation): Aligning permanent_water_mask with elevation_da.")
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
            source_labels_cond1 = np.unique(labels_in_global_pwm)
            source_labels_list.append(source_labels_cond1)

    min_elevs_per_label = scipy.ndimage.minimum(
        elevation_np,
        labels=labels_np,
        index=np.arange(1, num_features + 1)
    )
    deep_enough_mask = min_elevs_per_label < (water_level - depression_depth_for_isolated_source)
    source_labels_cond2 = np.arange(1, num_features + 1)[deep_enough_mask]
    if source_labels_cond2.size > 0:
        source_labels_list.append(source_labels_cond2)

    if not source_labels_list:
        final_source_labels = np.array([], dtype=int)
    else:
        final_source_labels = np.unique(np.concatenate(source_labels_list))

    connected_inundation_np = np.isin(labels_np, final_source_labels)

    return xr.DataArray(connected_inundation_np,
                        coords=elevation_da.coords,
                        dims=elevation_da.dims,
                        name='connected_inundation') \
             .astype(np.int8).where(elevation_da.notnull())

def apply_land_mask_from_shapefile(
    inundation_da: xr.DataArray,
    land_shapefile_path: str,
    lon_coord: str = "lon",
    lat_coord: str = "lat"
) -> xr.DataArray:
    if not os.path.exists(land_shapefile_path):
        print(f"Warning (apply_land_mask_from_shapefile): Shapefile non trouvé à {land_shapefile_path}. Masquage ignoré.")
        return inundation_da

    try:
        if inundation_da.rio.crs is None:
            print(f"Info (apply_land_mask_from_shapefile): CRS non défini pour inundation_da. Assignation EPSG:4326 (WGS84).")
            inundation_da = inundation_da.rio.set_spatial_dims(x_dim=lon_coord, y_dim=lat_coord, inplace=False)
            inundation_da = inundation_da.rio.write_crs("EPSG:4326", inplace=False)

        print(f"Info (apply_land_mask_from_shapefile): Chargement du shapefile: {land_shapefile_path}")
        land_gdf = gpd.read_file(land_shapefile_path)

        if inundation_da.rio.crs != land_gdf.crs:
            print(f"Info (apply_land_mask_from_shapefile): Réprojection du shapefile de {land_gdf.crs} vers {inundation_da.rio.crs}.")
            land_gdf = land_gdf.to_crs(inundation_da.rio.crs)

        print("Info (apply_land_mask_from_shapefile): Rasterisation des polygones terrestres...")
        transform = inundation_da.rio.transform()
        
        land_mask_np = rio_features_rasterize(
            shapes=land_gdf.geometry,
            out_shape=inundation_da.shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )
        
        land_bool_mask_da = xr.DataArray(
            land_mask_np.astype(bool),
            coords=inundation_da.coords,
            dims=inundation_da.dims,
            name="land_mask_from_shapes"
        )

        print("Info (apply_land_mask_from_shapefile): Application du masque.")
        masked_inundation_da = inundation_da.where(~land_bool_mask_da)

        if not masked_inundation_da.notnull().any():
            print("Warning (apply_land_mask_from_shapefile): Masquage a rendu toutes les valeurs NaN.")
        else:
            original_valid = inundation_da.notnull().sum().item()
            masked_valid = masked_inundation_da.notnull().sum().item()
            print(f"Info (apply_land_mask_from_shapefile): Pixels valides avant: {original_valid}, après: {masked_valid}")

        return masked_inundation_da

    except Exception as e:
        print(f"Erreur (apply_land_mask_from_shapefile): {e}. Retour de la carte originale.")
        import traceback
        traceback.print_exc()
        return inundation_da