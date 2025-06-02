# src/main.py
import sys
import os 
from . import config as app_config
from . import analysis

def main():
    print("--- Début de l'Analyse Combinée SWOT et Ligne d'Eau pour Multiples Zones ---")

    try:
        global_cfg = app_config.GLOBAL_CONFIG
        all_zones_info = app_config.load_zone_configurations(global_cfg["zone_info_filepath"])
    except FileNotFoundError as e:
        print(f"Erreur de configuration: {e}")
        sys.exit(1)
    except Exception as e_cfg:
        print(f"Erreur inattendue lors du chargement de la configuration: {e_cfg}")
        sys.exit(1)


    if not all_zones_info:
        print("Aucune zone définie dans le fichier JSON. Arrêt.")
        sys.exit(1)

    for zone_id, zone_data in all_zones_info.items():
        current_config_for_zone = global_cfg.copy()
        current_config_for_zone["target_zone_id"] = zone_id

        if "extent" not in zone_data or \
           "lon" not in zone_data["extent"] or \
           "lat" not in zone_data["extent"] or \
           len(zone_data["extent"]["lon"]) != 2 or \
           len(zone_data["extent"]["lat"]) != 2:
            print(f"Extent mal défini pour la zone {zone_id}. Passage à la zone suivante.")
            continue
        
        current_config_for_zone["analysis_roi_polygon"] = app_config.create_roi_polygon_from_extent(zone_data["extent"])
        
        try:
            analysis.run_zone_analysis_pipeline(zone_id, zone_data, current_config_for_zone)
        except Exception as e_analysis:
            print(f"ERREUR LORS DE L'ANALYSE DE LA ZONE {zone_id}: {e_analysis}")
            import traceback
            traceback.print_exc()
            print(f"Passage à la zone suivante suite à l'erreur pour la zone {zone_id}.")
            continue


    print("\n--- Fin de toutes les analyses ---")

if __name__ == "__main__":
    main()