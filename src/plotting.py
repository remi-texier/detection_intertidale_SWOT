# src/plotting.py
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

def create_single_inundation_plot( # Renommé pour clarté
    inundation_da: xr.DataArray,
    elev_fname_base: str,
    current_time: datetime,
    water_level: float,
    roi_info_flag: bool,
    max_plot_dim: int = 2000
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:

    if not inundation_da.size or not inundation_da.notnull().any().item():
        print(f"Info (create_single_inundation_plot): Pas de données valides pour {current_time}.")
        return None, None

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
                 print(f"Warning (create_single_inundation_plot): Downsampling a vidé les données pour {current_time}.")
                 return None, None
    elif inundation_da.ndim > 0 : 
        print(f"Warning (create_single_inundation_plot): inundation_da non 2D (dims: {inundation_da.dims}).")

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = mcolors.ListedColormap(['#FFFFFF', "#76A0C9"])
    cmap.set_bad('darkgray', alpha=0.5)

    plot_da.plot.imshow(ax=ax, cmap=cmap, vmin=0, vmax=1, add_colorbar=False, interpolation='nearest')

    if plot_da.isin([0]).any().item() and plot_da.isin([1]).any().item():
        try:
            plot_da.plot.contour(ax=ax, levels=[0.5], colors=['#00008B'], linewidths=0.5)
        except Exception as e:
            print(f"Warning (create_single_inundation_plot): Contour plot a échoué pour {current_time}. Error: {e}")

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    roi_str = " (ROI Applied)" if roi_info_flag else ""
    title = (f"Ligne d'Eau ({elev_fname_base}{roi_str})\n"
             f"Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | Niveau d'eau: {water_level:.2f} m")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.5)

    handles = [
        plt.Rectangle((0,0),1,1, color='#76A0C9'), plt.Rectangle((0,0),1,1, color='#FFFFFF', ec='gray'),
        mlines.Line2D([],[], color='#00008B', linewidth=0.5), plt.Rectangle((0,0),1,1, color='darkgray', alpha=0.5)
    ]
    labels = ["Eau (connectée)", "Sec / Eau (non connectée)", f"Ligne d'eau ({water_level:.2f}m)", "Données absentes / Hors ROI"]
    ax.legend(handles=handles, labels=labels, title="Légende", loc='upper right', framealpha=0.9)
    
    return fig, ax

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
    
    cmap_inundation = mcolors.ListedColormap(['#FFFFFF', "#76A0C9"])
    cmap_inundation.set_bad('darkgray', alpha=0.5)

    plot_da.plot.imshow(ax=ax, cmap=cmap_inundation, vmin=0, vmax=1, 
                              add_colorbar=False, interpolation='nearest')

    if plot_da.isin([0]).any().item() and plot_da.isin([1]).any().item():
        try:
            plot_da.plot.contour(ax=ax, levels=[0.5], colors=['#00008B'], linewidths=0.7)
        except Exception:
            pass 

    ax.set_title(f"Ligne d'eau (Niveau: {water_level:.2f}m) {title_suffix}")
    ax.set_xlabel("Longitude (°)") 
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    handles = [
        plt.Rectangle((0, 0), 1, 1, color='#76A0C9'),
        plt.Rectangle((0, 0), 1, 1, color='#FFFFFF', ec='gray'),
        mlines.Line2D([], [], color='#00008B', linewidth=0.7),
        plt.Rectangle((0, 0), 1, 1, color='darkgray', alpha=0.5)
    ]
    labels = [
        "Eau (connectée)",
        "Sec / Eau (non connectée)",
        f"Ligne d'eau ({water_level:.2f}m)",
        "Données absentes"
    ]
    ax.legend(handles=handles, labels=labels, title="Légende", loc='upper right', fontsize='x-small', framealpha=0.7)

def create_swot_data_visualization_on_ax(ax: plt.Axes, data_to_plot: xr.DataArray, 
                                         lon_coords: Optional[np.ndarray], lat_coords: Optional[np.ndarray],
                                         cmap_str: str, cbar_label: str, title: str):
    if lon_coords is not None and lat_coords is not None and \
       lon_coords.shape == data_to_plot.shape and lat_coords.shape == data_to_plot.shape:
        im = ax.pcolormesh(lon_coords, lat_coords, data_to_plot, cmap=cmap_str, shading='auto')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else: # Fallback to imshow if coords are not suitable for pcolormesh
        origin_val = 'lower' if data_to_plot.ndim == 2 and data_to_plot.shape[0] > 1 else 'upper'
        im = ax.imshow(data_to_plot, cmap=cmap_str, aspect='auto', origin=origin_val)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    ax.set_title(title)
    plt.gcf().colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04) # Use plt.gcf() to get current figure
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.axis('on')