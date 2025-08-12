# src/analysis.py
import os
import traceback
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from typing import Optional
import logging

from . import data_loader
from .processing import Litto3D_processing, swot_processing
from . import plotting 
from . import config

def _interpolate_water_level(config: dict, swot_time, ui_queue, report_queue, task_name, pid):
    """Interpole le niveau d'eau à partir du marégraphe."""
    ui_queue.put((pid, "status", f"{task_name} | Interpolation marée..."))
    ui_queue.put((pid, "log", f"Début interpolation niveau d'eau pour SWOT time: {swot_time}"))
    
    tide_file = config.get("tide_gauge_filepath") 
    if not tide_file:
        msg = "Chemin vers le fichier marégraphique non fourni dans la configuration."
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        return None, None
    
    ui_queue.put((pid, "log", f"Lecture fichier marégraphe: {tide_file}"))
    tide_df = data_loader.parse_tide_gauge_data(tide_file) 
    if tide_df is None:
        msg = f"Fichier marégraphique illisible, vide ou non trouvé à l'emplacement: {tide_file}"
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        return None, None

    ui_queue.put((pid, "log", f"Données marégraphe chargées: {len(tide_df)} points"))
    ui_queue.put((pid, "log", f"Colonnes disponibles: {list(tide_df.columns)}"))
    
    if 'DateTime' not in tide_df.columns:
        msg = f"Colonne 'DateTime' manquante dans le fichier marégraphe. Colonnes trouvées: {list(tide_df.columns)}"
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        return None, None
    
    if 'Value' not in tide_df.columns:
        msg = f"Colonne 'Value' manquante dans le fichier marégraphe. Colonnes trouvées: {list(tide_df.columns)}"
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        return None, None

    tide_df = tide_df.sort_values(by='DateTime')
    ui_queue.put((pid, "log", f"Données triées par DateTime"))
    
    min_time = tide_df['DateTime'].min()
    max_time = tide_df['DateTime'].max()
    min_value = tide_df['Value'].min()
    max_value = tide_df['Value'].max()
    ui_queue.put((pid, "log", f"Plage temporelle marégraphe: {min_time} à {max_time}"))
    ui_queue.put((pid, "log", f"Plage valeurs marégraphe: {min_value:.3f} à {max_value:.3f} m"))
    
    swot_ts = pd.Timestamp(swot_time)
    ui_queue.put((pid, "log", f"SWOT timestamp converti: {swot_ts}"))
    
    time_diff_start = (swot_ts - min_time).total_seconds() / 3600
    time_diff_end = (max_time - swot_ts).total_seconds() / 3600
    ui_queue.put((pid, "log", f"Écart SWOT vs début marégraphe: {time_diff_start:.2f} heures"))
    ui_queue.put((pid, "log", f"Écart fin marégraphe vs SWOT: {time_diff_end:.2f} heures"))
    
    if not (min_time <= swot_ts <= max_time): 
        msg = f"Heure SWOT ({swot_ts}) hors de la plage du marégraphe ({min_time} à {max_time})."
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        return None, None

    ui_queue.put((pid, "log", f"SWOT time dans la plage marégraphe ✓"))
    
    before_swot = tide_df[tide_df['DateTime'] <= swot_ts]
    after_swot = tide_df[tide_df['DateTime'] >= swot_ts]
    
    if not before_swot.empty:
        closest_before = before_swot.iloc[-1]
        time_diff_before = (swot_ts - closest_before['DateTime']).total_seconds() / 60  # en minutes
        ui_queue.put((pid, "log", f"Point précédent: {closest_before['DateTime']} (valeur: {closest_before['Value']:.3f} m, écart: {time_diff_before:.1f} min)"))
    
    if not after_swot.empty:
        closest_after = after_swot.iloc[0]
        time_diff_after = (closest_after['DateTime'] - swot_ts).total_seconds() / 60  # en minutes
        ui_queue.put((pid, "log", f"Point suivant: {closest_after['DateTime']} (valeur: {closest_after['Value']:.3f} m, écart: {time_diff_after:.1f} min)"))

    tide_indexed = tide_df.set_index('DateTime')
    new_index = tide_indexed.index.union([swot_ts])
    ui_queue.put((pid, "log", f"Index étendu avec SWOT time, taille: {len(new_index)}"))
    
    reindexed_df = tide_indexed.reindex(new_index)
    ui_queue.put((pid, "log", f"Données réindexées, valeurs NaN à interpoler: {reindexed_df['Value'].isna().sum()}"))
    
    interpolated_series = reindexed_df['Value'].interpolate(method='time')
    interp_value = interpolated_series.get(swot_ts)
    
    ui_queue.put((pid, "log", f"Valeur interpolée brute: {interp_value}"))

    if pd.notna(interp_value):
        water_level = float(interp_value)
        water_source = "Marégraphe"
        ui_queue.put((pid, "log", f"Interpolation réussie - Niveau d'eau final: {water_level:.3f} m"))
        ui_queue.put((pid, "log", f"Source: {water_source}"))
        return water_level, water_source
    else:
        msg = "Impossible d'interpoler le niveau d'eau à l'heure SWOT. Données manquantes dans le marégraphe."
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        ui_queue.put((pid, "log", f"Échec interpolation: valeur NaN après interpolation"))
        return None, None

def _create_and_save_netcdf(zone_id, cycle, pass_id, config, swot_data, 
                           elevation_roi, inundation_map, water_level, water_source,
                           swot_time, time_fallback, unsmoothed_file, ui_queue, report_queue, task_name, pid):
    """Crée et sauvegarde le fichier NetCDF."""
    ui_queue.put((pid, "status", f"{task_name} | Préparation NetCDF..."))
    pass_str, cycle_str = str(pass_id).zfill(3), str(cycle).zfill(3)
    results_dir = os.path.join(config.get("results_base_path", "results"), "data")
    os.makedirs(results_dir, exist_ok=True)
    netcdf_name = f"combined_analysis_zone_{zone_id}_pass_{pass_str}_cycle_{cycle_str}.nc"
    netcdf_path = os.path.join(results_dir, netcdf_name)

    output_data = xr.Dataset(attrs={
        'description': f"Processed data for SWOT analysis in zone {zone_id}, Pass {pass_str}, Cycle {cycle_str}.",
        'original_swot_file': os.path.basename(unsmoothed_file),
        'mnt_file_used': os.path.basename(config.get("mnt_filepath", "N/A")),
        'tide_gauge_file_used': os.path.basename(config.get("tide_gauge_filepath", "N/A")),
        'processing_date': datetime.now().isoformat(),
        'tide_height': f"{water_level:.3f} m",
        'water_level_source_for_inundation': water_source,
        'swot_time_median': swot_time.isoformat() + (' (FALLBACK)' if time_fallback else '')
    })
    
    downsample = config.get("downsampling", config.get("rasterization_downsampling_factor", 1))
    target_grid = elevation_roi
    if downsample > 1:
        ui_queue.put((pid, "log", f"Sous-échantillonnage (x{downsample})..."))
        original_crs = elevation_roi.rio.crs
        target_grid = elevation_roi.coarsen({config["mnt_lon"]: downsample, config["mnt_lat"]: downsample}, boundary="trim").mean()
        if original_crs is not None:
            target_grid.rio.write_crs(original_crs, inplace=True)

    rasterized_swot = swot_processing.rasterize_swot_data(swot_data, target_grid, config)
    for var_name, data_array in rasterized_swot.items():
        output_data[var_name] = data_array

    output_data['dem_roi'] = target_grid.rename('dem_roi')
    if inundation_map is not None and inundation_map.notnull().any():
        inund_crs = inundation_map.rio.crs
        inund_downsampled = inundation_map.coarsen({config["mnt_lon"]: downsample, config["mnt_lat"]: downsample}, boundary="trim").max()
        if inund_crs is not None:
            inund_downsampled.rio.write_crs(inund_crs, inplace=True)
        output_data['inundation_mask'] = inund_downsampled.rename('inundation_mask')

    if output_data.data_vars:
        try:
            ui_queue.put((pid, "status", f"{task_name} | Sauvegarde NetCDF..."))
            encoding = {var: {'zlib': True, 'complevel': 5} for var in output_data.data_vars}
            output_data.to_netcdf(netcdf_path, engine="netcdf4", encoding=encoding)
            ui_queue.put((pid, "log", f"NetCDF OK: {os.path.basename(netcdf_path)}"))
            return netcdf_path
        except Exception as e:
            tb_str = traceback.format_exc()
            msg = f"Erreur lors de la sauvegarde du NetCDF '{netcdf_name}': {e}\n\n{tb_str}"
            report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
            logging.error(msg)
            return None
    else:
        msg = "Aucune donnée n'a été générée pour la rasterisation. Le fichier NetCDF de sortie serait vide."
        report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': msg})
        return None

def process_and_save_zone_data(zone_id: str, zone_data: dict, config: dict, cycle: str, pass_id: str, ui_queue, report_queue, task_name) -> Optional[str]:
    """
    Traite les données et envoie des messages de log et de statut via les queues.
    """
    pid = os.getpid()
    ui_queue.put((pid, "status", f"Démarrage: {task_name}"))
    ui_queue.put((pid, "log", f"PID: {pid} | Début"))

    # 1. Chargement et traitement SWOT
    swot_data, unsmoothed_file = swot_processing.load_and_process_swot_data(config, cycle, pass_id, zone_data, ui_queue, report_queue, task_name, pid)
    if swot_data is None:
        ui_queue.put((pid, "final", f"[bold yellow]! SWOT manquant: {task_name}[/bold yellow]"))
        return None

    # 2. Orientation et temps SWOT
    swot_data, time_info = swot_processing.process_swot_orientation_and_time(swot_data, ui_queue, report_queue, task_name, pid)
    if swot_data is None:
        ui_queue.put((pid, "final", f"[bold red]✗ Donnée SSH manquante: {task_name}[/bold red]"))
        return None
    swot_time, time_fallback = time_info

    # 3. Interpolation niveau d'eau
    water_result = _interpolate_water_level(config, swot_time, ui_queue, report_queue, task_name, pid)
    if water_result is None or water_result[0] is None:
        ui_queue.put((pid, "final", f"[bold red]✗ Erreur marée: {task_name}[/bold red]"))
        return None
    water_level, water_source = water_result

    # 4. Chargement MNT
    elevation_roi = Litto3D_processing.load_and_process_dem(config, ui_queue, report_queue, task_name, pid)
    if elevation_roi is None:
        ui_queue.put((pid, "final", f"[bold red]✗ Erreur MNT: {task_name}[/bold red]"))
        return None

    # 5. Calcul inondation
    inundation_map = Litto3D_processing.compute_inundation_map(elevation_roi, water_level, config, ui_queue, pid, task_name)

    # 6. Sauvegarde NetCDF
    netcdf_path = _create_and_save_netcdf(zone_id, cycle, pass_id, config, swot_data, 
                                         elevation_roi, inundation_map, water_level, water_source,
                                         swot_time, time_fallback, unsmoothed_file, ui_queue, report_queue, task_name, pid)
    
    if netcdf_path is None:
        ui_queue.put((pid, "final", f"[bold red]✗ Échec sauvegarde: {task_name}[/bold red]"))
        return None

    return netcdf_path