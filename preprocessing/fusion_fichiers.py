import os
import re
from collections import defaultdict

def extract_zone(filename): # Extrait la zone de 'ZONE_ANNEE.txt'
    match = re.match(r"(\d+)_(\d{4})\.txt$", filename)
    return match.group(1) if match else None

def merge_files_by_zone(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True) # Crée output_dir si besoin
    files_by_zone = defaultdict(list)

    for filename in os.listdir(input_dir):
        zone = extract_zone(filename)
        if zone: files_by_zone[zone].append(os.path.join(input_dir, filename))

    if not files_by_zone:
        print(f"Aucun fichier valide (format ZONE_ANNEE.txt) trouvé dans {input_dir}")
        return

    for zone, file_paths in files_by_zone.items():
        file_paths.sort() # Tri important pour que le "premier" fichier (en-têtes) soit constant
        output_filepath = os.path.join(output_dir, f"zone_{zone}_fusionne.txt")
        print(f"Traitement Zone {zone} -> {os.path.basename(output_filepath)}")
        first_file_in_zone_processed = False # Pour gestion spéciale des en-têtes du premier fichier

        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for filepath in file_paths:
                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        if not first_file_in_zone_processed: # Logique pour le premier fichier (en-têtes + données)
                            data_header_line_written = False # Pour écrire l'en-tête de données une seule fois
                            for line in infile:
                                if line.startswith("#"): outfile.write(line) # Commentaires
                                elif not line.strip() and not data_header_line_written: outfile.write(line) # Ligne vide dans l'en-tête
                                elif not data_header_line_written: # En-tête de données
                                    outfile.write(line)
                                    data_header_line_written = True
                                else: outfile.write(line) # Données
                            first_file_in_zone_processed = True
                        else: # Logique pour les fichiers suivants (données uniquement)
                            data_header_skipped = False # Pour sauter l'en-tête de données de ce fichier
                            for line in infile:
                                if line.startswith("#") or not line.strip(): continue # Ignorer commentaires et lignes vides
                                if not data_header_skipped: # Sauter l'en-tête de données de ce fichier
                                    data_header_skipped = True
                                    continue
                                outfile.write(line) # Données
                except Exception as e: # Gestion d'erreur minimale par fichier
                    print(f"  Avertissement sur '{os.path.basename(filepath)}': {e} (ignoré)")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Configuration des chemins (ajuster si vos dossiers ne sont pas à côté du script)
    input_directory = os.path.join(script_dir, "maregraphie_WGS84")
    output_directory = os.path.join(script_dir, "maregraphie_WGS84_fusion")

    if not os.path.isdir(input_directory): # Vérification critique
        print(f"Erreur : Répertoire d'entrée '{input_directory}' introuvable.")
    else:
        print(f"Fusion depuis '{input_directory}' vers '{output_directory}'...")
        merge_files_by_zone(input_directory, output_directory)
        print("Fusion terminée.")