# src/plotting.py
import logging
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import matplotlib.lines as mlines

log = logging.getLogger("rich_app")

def plot_inundation_on_ax(ax: plt.Axes, inundation_da: xr.DataArray, water_level: float, title_suffix: str = "", max_plot_dim: int = 1000):
    if not isinstance(inundation_da, xr.DataArray) or not inundation_da.size or not inundation_da.notnull().any().item():
        ax.text(0.5, 0.5, "Données d'inondation\\nnon disponibles ou vides", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize='small')
        ax.set_title(f"Ligne d'eau (Niveau: {water_level:.2f}m) {title_suffix}")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('on') 
        ax.grid(True, linestyle=':', alpha=0.5)
        return

    plot_da = inundation_da
    original_shape = inundation_da.shape

    if inundation_da.ndim == 2:
        dim1_name, dim2_name = inundation_da.dims[0], inundation_da.dims[1]
        factor1 = max(1, int(np.ceil(original_shape[0] / max_plot_dim)))
        factor2 = max(1, int(np.ceil(original_shape[1] / max_plot_dim)))

        if factor1 > 1 or factor2 > 1:
            coarsen_dims = {}
            if factor1 > 1: coarsen_dims[dim1_name] = factor1
            if factor2 > 1: coarsen_dims[dim2_name] = factor2
            if coarsen_dims:
                plot_da = inundation_da.coarsen(**coarsen_dims, boundary="trim").max()
            if not plot_da.size or not plot_da.notnull().any().item():
                ax.text(0.5, 0.5, "Données d'inondation\\ndeviennent vides après sous-échantillonnage", 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=ax.transAxes, fontsize='small')
                ax.set_title(f"Ligne d'eau (Niveau: {water_level:.2f}m) {title_suffix} (Données vides)")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.axis('on')
                ax.grid(True, linestyle=':', alpha=0.5)
                return
    
    cmap_inundation = mcolors.ListedColormap(['#d2b48c', '#76A0C9', '#ffa500'])
    cmap_inundation.set_bad('darkgray', alpha=0.5)

    # Les bornes sont importantes pour que les couleurs correspondent aux classes
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap_inundation.N)

    plot_da.plot.imshow(ax=ax, cmap=cmap_inundation, norm=norm,
                              add_colorbar=False, interpolation='nearest')

    ax.set_title(f"Masque d'Inondation (Niveau: {water_level:.2f}m) {title_suffix}")
    ax.set_xlabel("Longitude (°)") 
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    handles = [
        plt.Rectangle((0, 0), 1, 1, color='#d2b48c'), # Terre
        plt.Rectangle((0, 0), 1, 1, color='#76A0C9'), # Eau
        plt.Rectangle((0, 0), 1, 1, color='#ffa500'), # Incertain
        plt.Rectangle((0, 0), 1, 1, color='darkgray', alpha=0.5)
    ]
    labels = [
        "Terre (Classe 0)",
        "Eau (Classe 1)",
        "Incertain (Classe 2)",
        "Données absentes"
    ]
    ax.legend(handles=handles, labels=labels, title="Légende", loc='upper right', fontsize='x-small', framealpha=0.9)
    
def create_plots_from_netcdf(netcdf_filepath: str, config: dict, ui_queue, report_queue, task_name):
    """
    Crée une figure de visualisation et envoie le statut final via les queues.
    """
    pid = os.getpid()

    if not os.path.exists(netcdf_filepath):
        msg = f"Le fichier NetCDF '{os.path.basename(netcdf_filepath)}' devait exister pour créer la figure, mais il est manquant."
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        ui_queue.put((pid, "final", f"[bold red]✗ Fichier NetCDF manquant: {task_name}[/bold red]"))
        return

    ui_queue.put((pid, "status", f"{task_name} | Création figure..."))
    
    try:
        ds = xr.open_dataset(netcdf_filepath)
    except Exception as e:
        msg = f"Erreur à l'ouverture du NetCDF '{os.path.basename(netcdf_filepath)}' pour plotting : {e}"
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        ui_queue.put((pid, "final", f"[bold red]✗ Erreur lecture NetCDF: {task_name}[/bold red]"))
        return

    try:
        desc = ds.attrs.get('description', '')
        zone_id = desc.split('zone ')[-1].split(' ')[0] if 'zone ' in desc else 'N/A'
        pass_id = desc.split('Pass ')[-1].split(',')[0] if 'Pass ' in desc else 'N/A'
        cycle = desc.split('Cycle ')[-1].split('.')[0] if 'Cycle ' in desc else 'N/A'
        water_level_str = ds.attrs.get('tide_height', 'N/A')
        
        plot_elements = [var for var in ["swot_sig0", "swot_ssh", "inundation_mask"] if var in ds.data_vars]
        if not plot_elements:
            msg = "Le NetCDF ne contient aucune variable attendue pour le plotting (swot_sig0, swot_ssh, inundation_mask)."
            report_queue.put({'task_name': task_name, 'level': 'WARNING', 'message': msg})
            ui_queue.put((pid, "final", f"[bold yellow]! Données à plotter absentes: {task_name}[/bold yellow]"))
            return

        fig, axes = plt.subplots(1, len(plot_elements), figsize=(7 * len(plot_elements), 6.5), squeeze=False)
        plot_map = {"swot_sig0": ("SWOT sig0", 'gray', 'sig0 (dB)'), "swot_ssh": ("SWOT SSH", 'viridis', 'SSH (m)')}

        for i, element in enumerate(plot_elements):
            ax = axes[0, i]
            if element in plot_map:
                title_part, cmap, cbar_label = plot_map[element]
                ds[element].plot.imshow(ax=ax, cmap=cmap, add_colorbar=True, cbar_kwargs={'label': cbar_label})
                ax.set_title(f"{title_part}\nZone: {zone_id}, C{cycle} P{pass_id}")
                ax.set_aspect('equal', adjustable='box')
            elif element == "inundation_mask":
                water_level_val = float(water_level_str.split(' ')[0])
                title_suffix = f"(Zone: {zone_id} C{cycle} P{pass_id})"
                plot_inundation_on_ax(ax, ds['inundation_mask'], water_level=water_level_val, title_suffix=title_suffix, max_plot_dim=1000)


        plt.tight_layout(pad=2.0)
        base_title = f"Analyse Combinée - SWOT Passe {pass_id} Cycle {cycle} (Zone: {zone_id})"
        fig.suptitle(base_title, fontsize=16, y=1.03)

        fig_dir = os.path.join(config.get("results_base_path", "results"), "figures")
        os.makedirs(fig_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(netcdf_filepath))[0]
        fig_filename = f"{base_filename}.png"
        fig_path = os.path.join(fig_dir, fig_filename)
    
        plt.savefig(fig_path, dpi=150)
        ui_queue.put((pid, "log", f"Figure OK: {os.path.basename(fig_path)}"))
        ui_queue.put((pid, "final", f"[bold green]✓ Terminé: {task_name}[/bold green]"))

    except Exception as e:
        msg = f"Erreur inattendue lors de la création ou sauvegarde de la figure: {e}"
        report_queue.put({'task_name': task_name, 'level': 'ERROR', 'message': msg})
        ui_queue.put((pid, "final", f"[bold red]✗ Échec figure: {task_name}[/bold red]"))
    finally:
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        ds.close()