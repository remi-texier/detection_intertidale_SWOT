# src/analysis.py
import os
import numpy as np
import pandas as pd # Pour pd.to_datetime
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

from . import data_loader
from .processing import swot_processing, dem_processing, inundation_mapping
from . import plotting
from . import config as app_config # Pour accéder à GLOBAL_CONFIG et fonctions de config

def run_zone_analysis_pipeline(zone_id: str, zone_data: dict, current_config: dict):
    print(f"\n--- Début de l'analyse pour la Zone: {zone_id} ({zone_data.get('name', 'N/A')}) ---")

    # 1. Chargement des données SWOT
    first_cycle = str(zone_data["cycles"][0]).zfill(3)
    first_pass_id = str(zone_data["pass_id"][0]).zfill(3)

    expert_fp = data_loader.find_swot_files(current_config["base_data_path"], first_cycle, first_pass_id, "Expert")
    unsmoothed_fp = data_loader.find_swot_files(current_config["base_data_path"], first_cycle, first_pass_id, "Unsmoothed")

    if not unsmoothed_fp: # expert_fp est optionnel pour la correction, unsmoothed est essentiel
        print(f"Fichier SWOT Unsmoothed manquant pour la zone {zone_id}. Passage à la zone suivante.")
        return

    expert_ds_full = data_loader.read_swot_datafile(expert_fp, is_expert=True) if expert_fp else None
    unsmoothed_ds_full_groups = data_loader.read_swot_datafile(unsmoothed_fp, is_expert=False)

    if not unsmoothed_ds_full_groups or not any(isinstance(ds, xr.Dataset) for ds in unsmoothed_ds_full_groups.values()):
        print(f"Erreur: Aucune donnée Unsmoothed SWOT chargée pour la zone {zone_id}. Arrêt de cette zone.")
        return
    
    swot_file_info = data_loader.extract_swot_filename_info(unsmoothed_fp)
    
    # 2. Traitement SWOT
    allowed_data_groups = zone_data.get("data_group", [])
    unsmoothed_ds_filtered_groups = {
        gn: ds for gn, ds in unsmoothed_ds_full_groups.items() 
        if not allowed_data_groups or gn in allowed_data_groups
    }
    if not unsmoothed_ds_filtered_groups:
        print(f"Aucun groupe de données Unsmoothed correspondant à {allowed_data_groups} pour la zone {zone_id}. Arrêt.")
        return

    unsmoothed_cleaned_groups = swot_processing.clean_swot_dataset_variables(unsmoothed_ds_filtered_groups)
    if not any(isinstance(ds, xr.Dataset) and (ds.data_vars or ds.coords) for ds in unsmoothed_cleaned_groups.values()):
        print(f"Données Unsmoothed SWOT vides après nettoyage pour la zone {zone_id}. Arrêt."); return

    analysis_roi_swot_poly = current_config["analysis_roi_polygon"] # Le polygone de la config
    unsmoothed_roi_swot = swot_processing.apply_roi_to_swot_data_groups(unsmoothed_cleaned_groups, analysis_roi_swot_poly)
    
    expert_roi_swot = None
    if expert_ds_full is not None:
        expert_ds_full_cleaned = swot_processing.clean_swot_dataset_variables(expert_ds_full)
        expert_roi_swot = swot_processing.apply_roi_to_swot_dataset(expert_ds_full_cleaned, analysis_roi_swot_poly)

    unsmoothed_roi_valid_swot = {
        gn: ds for gn, ds in unsmoothed_roi_swot.items() 
        if isinstance(ds, xr.Dataset) and all(s_val > 0 for dim, s_val in ds.sizes.items() if dim in ['num_lines', 'num_pixels'])
    }
    if not unsmoothed_roi_valid_swot:
        print(f"Aucune donnée Unsmoothed SWOT après ROI pour la zone {zone_id}. Arrêt."); return
    
    expert_roi_swot_is_valid = (expert_roi_swot is not None and isinstance(expert_roi_swot, xr.Dataset) and \
                                all(s_val > 0 for dim, s_val in expert_roi_swot.sizes.items() if dim in ['num_lines', 'num_pixels']))

    unsmoothed_final_swot = swot_processing.apply_ssh_correction(unsmoothed_roi_valid_swot, expert_roi_swot if expert_roi_swot_is_valid else None)

    display_swot_group_name = next((
        g for g in (allowed_data_groups if allowed_data_groups else unsmoothed_final_swot.keys())
        if g in unsmoothed_final_swot and 
           isinstance(unsmoothed_final_swot[g], xr.Dataset) and 
           unsmoothed_final_swot[g].sizes.get('num_lines',0) > 0
    ), None)
    if not display_swot_group_name and unsmoothed_final_swot:
         display_swot_group_name = next((
            g for g, ds in unsmoothed_final_swot.items() 
            if isinstance(ds, xr.Dataset) and ds.sizes.get('num_lines',0) > 0
        ), None)

    if not display_swot_group_name:
        print(f"Aucun groupe SWOT valide avec des données après ROI et correction pour {zone_id}. Arrêt."); return
    
    ds_swot_display = unsmoothed_final_swot[display_swot_group_name]

    if "ssh_karin_2_corrected" not in ds_swot_display:
        print(f"Erreur: 'ssh_karin_2_corrected' non trouvée dans '{display_swot_group_name}' pour {zone_id}. Arrêt."); return
        
    mean_ssh_swot = float(ds_swot_display["ssh_karin_2_corrected"].mean(skipna=True).item())
    print(f"Niveau d'eau moyen (SWOT SSH corrigée sur ROI) pour zone {zone_id}: {mean_ssh_swot:.2f} m")
    
    swot_time_dataarray = ds_swot_display["time"]
    swot_datetime_for_ldl = datetime.now() # Fallback
    if swot_time_dataarray.size > 0:
        valid_times_np = swot_time_dataarray.values.flatten()
        valid_times_np = valid_times_np[~np.isnat(valid_times_np)]
        if valid_times_np.size > 0:
            median_timestamp_ns = np.median(valid_times_np.astype(np.int64))
            swot_datetime_for_ldl = pd.to_datetime(median_timestamp_ns, unit='ns')
    print(f"Heure de passage SWOT (médiane sur ROI) pour zone {zone_id}: {swot_datetime_for_ldl.strftime('%Y-%m-%d %H:%M:%S')}")

    # 3. Chargement et traitement du MNT
    mnt_roi_bbox = app_config.get_mnt_roi_bbox_from_polygon(analysis_roi_swot_poly)
    
    full_elevation_da = data_loader.load_elevation_data(current_config["mnt_filepath"], current_config["mnt_alt_var_name"])
    elevation_da_roi = None
    is_roi_applied_to_dem = False
    if full_elevation_da is not None:
        # Pour le MNT, on croppe d'abord au BBox du polygone pour efficacité
        elevation_da_bbox_crop, _ = dem_processing.crop_to_roi(
            full_elevation_da, mnt_roi_bbox, 
            current_config["mnt_lon_coord"], current_config["mnt_lat_coord"]
        )
        # Ensuite, si le ROI est un polygone, on applique le masque polygonal
        if isinstance(analysis_roi_swot_poly, list):
             elevation_da_roi, is_roi_applied_to_dem = dem_processing.crop_to_roi(
                elevation_da_bbox_crop, analysis_roi_swot_poly,
                current_config["mnt_lon_coord"], current_config["mnt_lat_coord"]
            )
        else: # Si le ROI était déjà un BBox (dict), on utilise le crop BBox
            elevation_da_roi = elevation_da_bbox_crop
            is_roi_applied_to_dem = True


    # 4. Calcul de la ligne d'eau et masquage
    inundation_map = None
    if elevation_da_roi is not None and elevation_da_roi.size > 0 and elevation_da_roi.notnull().any():
        perm_water_config = current_config["permanent_water_definition"]
        permanent_water_thresh = None
        if perm_water_config["method"] == "mnt_min_plus_offset":
            min_alt_mnt = float(elevation_da_roi.min(skipna=True).item())
            permanent_water_thresh = min_alt_mnt + perm_water_config["offset"]
        # ... (autres méthodes pour permanent_water_thresh) ...
        else:
            permanent_water_thresh = 0.0 # Fallback
        
        perm_water_mask = dem_processing.create_permanent_water_mask(elevation_da_roi, permanent_water_thresh)
        
        inundation_map_raw = inundation_mapping.compute_connected_inundation(
            elevation_da_roi, mean_ssh_swot, perm_water_mask,
            current_config.get("depression_depth_for_isolated_source", 0.1)
        )

        if inundation_map_raw is not None and inundation_map_raw.size > 0 and inundation_map_raw.notnull().any():
            if "france_land_shapefile" in current_config:
                print(f"Info zone {zone_id}: Application du masque terrestre au MNT d'inondation.")
                inundation_map = inundation_mapping.apply_land_mask_from_shapefile(
                    inundation_map_raw,
                    current_config["france_land_shapefile"],
                    lon_coord=current_config["mnt_lon_coord"],
                    lat_coord=current_config["mnt_lat_coord"]
                )
            else:
                inundation_map = inundation_map_raw # Pas de masque terrestre à appliquer

        if inundation_map is not None and not (inundation_map.size > 0 and inundation_map.notnull().any()):
             print(f"Carte d'inondation vide ou tout NaN pour zone {zone_id} après masquage éventuel.")
             inundation_map = None # S'assurer qu'elle est None si vide
    else:
        print(f"MNT non chargé ou vide après ROI pour zone {zone_id}, calcul de la ligne d'eau impossible.")

    # 5. Visualisation Combinée
    plot_elements = []
    if "sig0_karin_2" in ds_swot_display and ds_swot_display["sig0_karin_2"].size > 0:
        plot_elements.append("sig0")
    if "ssh_karin_2_corrected" in ds_swot_display and ds_swot_display["ssh_karin_2_corrected"].size > 0 :
        plot_elements.append("ssh")
    if isinstance(inundation_map, xr.DataArray) and inundation_map.size > 0 and inundation_map.notnull().any():
        plot_elements.append("inundation")
    
    num_plots = len(plot_elements)
    if num_plots == 0:
        print(f"Aucune donnée à visualiser pour la zone {zone_id}.")
        return

    fig_combined, axes_combined = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6.5), squeeze=False)
    current_ax_idx = 0

    if "sig0" in plot_elements:
        ax = axes_combined[0, current_ax_idx]
        lons_sig0 = ds_swot_display.longitude.data if 'longitude' in ds_swot_display else None
        lats_sig0 = ds_swot_display.latitude.data if 'latitude' in ds_swot_display else None
        data_sig0 = ds_swot_display["sig0_karin_2"].data
        plotting.create_swot_data_visualization_on_ax(
            ax, data_sig0, lons_sig0, lats_sig0, 'gray', "sig0 (dB)",
            f"SWOT sig0 - Grp: {display_swot_group_name}\nZone: {zone_id}"
        )
        current_ax_idx += 1

    if "ssh" in plot_elements:
        ax = axes_combined[0, current_ax_idx]
        lons_ssh = ds_swot_display.longitude.data if 'longitude' in ds_swot_display else None
        lats_ssh = ds_swot_display.latitude.data if 'latitude' in ds_swot_display else None
        data_ssh = ds_swot_display["ssh_karin_2_corrected"].data
        plotting.create_swot_data_visualization_on_ax(
            ax, data_ssh, lons_ssh, lats_ssh, 'viridis', "SSH Corrigée (m)",
            f"SWOT SSH Corrigée - Grp: {display_swot_group_name}\nZone: {zone_id}"
        )
        current_ax_idx += 1
        
    if "inundation" in plot_elements:
        ax = axes_combined[0, current_ax_idx]
        plotting.plot_inundation_on_ax(
            ax, inundation_map, mean_ssh_swot,
            title_suffix=f"(Zone: {zone_id})",
            max_plot_dim=current_config.get("max_plot_dim_inundation", 1000)
        )
    elif num_plots > current_ax_idx : # Si un slot était prévu mais pas de données
        ax = axes_combined[0, current_ax_idx]
        ax.text(0.5,0.5, "Ligne d'eau\nnon calculée\nou vide", ha='center',va='center',transform=ax.transAxes)
        ax.set_title(f"Ligne d'eau (Zone: {zone_id})")
        ax.grid(True, linestyle=':', alpha=0.5); ax.set_xticklabels([]); ax.set_yticklabels([])

    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.5)
    pass_str = swot_file_info.get('pass_id', 'N/A') if swot_file_info else 'N/A'
    cycle_str = swot_file_info.get('cycle', 'N/A') if swot_file_info else 'N/A'
    fig_combined.suptitle(f"Analyse Combinée - SWOT Passe {pass_str} Cycle {cycle_str} (Zone: {zone_id} - {zone_data.get('name', 'N/A')})",
                          fontsize=16, y=1.03 if num_plots > 1 else 1.05)
    
    # Sauvegarde de la figure
    fig_dir = os.path.join(current_config.get("results_base_path", "results"), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_filename = f"combined_analysis_zone_{zone_id}_pass_{pass_str}_cycle_{cycle_str}.png"
    fig_path = os.path.join(fig_dir, fig_filename)
    try:
        plt.savefig(fig_path)
        print(f"Figure sauvegardée : {fig_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de la figure {fig_path}: {e}")
    plt.show() # Ou plt.close(fig_combined) si exécution en batch sans affichage interactif

    print(f"--- Fin de l'analyse pour la Zone: {zone_id} ---")