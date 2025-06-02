# src2/config.py
import os
import json
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

GLOBAL_CONFIG: Dict[str, Any] = {
    "base_data_path": os.path.join(PROJECT_ROOT, "Donnees", "data", "L2_LR"),
    "zone_info_filepath": os.path.join(PROJECT_ROOT, "Donnees", "data", 'zone_info.json'),
    # MODIFICATION ICI :
    "mnt_filepath": os.path.join(PROJECT_ROOT, "Donnees", "litto3D-Aquitaine", "nouvelle_aquitaine-full", "NA_WGS84.nc"), # Supprimer "data" ici
    "mnt_alt_var_name": "Band1",
    "mnt_lon_coord": "lon",
    "mnt_lat_coord": "lat",
    "permanent_water_definition": {
        "method": "mnt_min_plus_offset",
        "offset": 0.1,
    },
    "depression_depth_for_isolated_source": 0.1,
    "france_land_shapefile": os.path.join(PROJECT_ROOT, "Donnees", "France", "clipped_land_polygons_France.shp"),
    "results_base_path": os.path.join(PROJECT_ROOT, "results"),
    "max_plot_dim_inundation": 1000,
    "max_plot_dim_swot": 2000
}

# ... (le reste du fichier config.py reste identique) ...
def load_zone_configurations(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier d'information de zone non trouvé : {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

def create_roi_polygon_from_extent(extent: Dict[str, List[float]]) -> List[Tuple[float, float]]:
    min_lon, max_lon = extent['lon']
    min_lat, max_lat = extent['lat']
    return [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat), (min_lon, min_lat)]

def get_mnt_roi_bbox_from_polygon(roi_polygon: List[Tuple[float, float]]) -> Dict[str, float]:
    poly_lons = [p[0] for p in roi_polygon]
    poly_lats = [p[1] for p in roi_polygon]
    return {'min_lon': min(poly_lons), 'max_lon': max(poly_lons), 
            'min_lat': min(poly_lats), 'max_lat': max(poly_lats)}