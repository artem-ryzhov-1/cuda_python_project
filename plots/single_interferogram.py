########################################
# plots/single_interferogram.py
########################################

import os
import sys
import platform
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

time1 = time.time()

# --- Imports ---
import _setup_paths
from app.python.config import SimRunGridMode
from app.python.simulation import run_simulation

# remove rho_avg_out.bin if it exists
(_setup_paths.CUDA_OUTPUT / "rho_avg_out.bin").unlink(missing_ok=True)
# remove rho_dynamics_single_mode_out.bin if it exists
(_setup_paths.CUDA_OUTPUT / "rho_dynamics_single_mode_out.bin").unlink(missing_ok=True)

#from main_v4_1_win_function import run_simulation


# environment = "Windows" or "Linux" or "Google_Colab" or "WSL2"
# Step 1: Get the general system type
environment = platform.system()

# Step 2: Further refine for Google Colab or WSL2
if environment == 'Linux':
    # Check if we're in Google Colab (check for google.colab module)
    if 'google.colab' in sys.modules:
        environment = 'Google_Colab'
    # Check if it's WSL2 (WSL2 includes 'microsoft' in release string)
    elif 'microsoft' in platform.release().lower():
        environment = 'WSL2'

print(f"Environment detected: {environment}")



##########################################

# possible value of platform_type: 'colab_linux', 'local_windows', 'local_linux', 'local_wsl2', 'local_macos'

# Determine the platform type (local or colab)
if 'COLAB_GPU' in os.environ:  # Check if running in Google Colab
    platform_type = 'colab_linux'
else:
    if sys.platform.startswith('linux'):
        # Check for WSL2
        # Check if running under WSL (by looking for /mnt/c/)
        if Path('/mnt/c/').exists():
            platform_type = 'local_wsl2'
        else:
            platform_type = 'local_linux'
    elif sys.platform.startswith('win'):
        platform_type = 'local_windows'
    elif sys.platform == 'darwin':
        platform_type = 'local_macos'
    else:
        raise RuntimeError("Unsupported platform")

print(f"Platform type: {platform_type}")
        
#########################################################

qs_flag = False # "True", "False"

E_C=0.195
hbar=6.582119569e-16
nu=21

delta_C=0.0005
GammaL0=500.0
GammaR0=100.0
Gamma_eg0=20.0
Gamma_phi0=3.6*5

sigma_eps=(Gamma_phi0*1e9*np.sqrt(2)*hbar/E_C)

N_points_target=100000

time2 = time.time()

if not qs_flag:
    simr = SimRunGridMode(
        delta_C=delta_C,
        GammaL0=GammaL0,
        GammaR0=GammaR0,
        Gamma_eg0=Gamma_eg0,
        Gamma_phi0=Gamma_phi0,
        nu=nu,
        E_C=E_C,
        
        eps0_min=-0.006,
        eps0_max=0.006,
        A_min=0.0,
        A_max=0.008,
        N_points_target=N_points_target,
        N_steps_period=1000,
        N_periods=10,
        N_periods_avg=1,
        
        quasi_static_ensemble_dephasing_flag=False,
        sigma_eps=None,
        N_samples_noise=None,
        
        rho00_init=0.25,
        rho11_init=0.25,
        rho22_init=0.25,
        rho33_init=0.25,
        
        platform_type = platform_type,
        repo_path=_setup_paths.PROJECT_ROOT
    )

elif qs_flag:
    simr = SimRunGridMode(
        delta_C=delta_C,
        GammaL0=GammaL0,
        GammaR0=GammaR0,
        Gamma_eg0=Gamma_eg0,
        Gamma_phi0=None,
        nu=nu,
        E_C=E_C,
        
        eps0_min=-0.006,
        eps0_max=0.006,
        A_min=0.0,
        A_max=0.008,
        N_points_target=N_points_target,
        N_steps_period=1000,
        N_periods=10,
        N_periods_avg=1,
        
        quasi_static_ensemble_dephasing_flag=True,
        sigma_eps=sigma_eps,
        N_samples_noise=100,
        
        rho00_init=0.25,
        rho11_init=0.25,
        rho22_init=0.25,
        rho33_init=0.25,
        
        platform_type = platform_type,
        repo_path=_setup_paths.PROJECT_ROOT
    )


time3 = time.time()

eps0_grid, A_grid, rho_avg_cdc_3d, returncode = run_simulation(simr)

time4 = time.time()

print(f"time2 - time1: {(time2 - time1):.3f} s")
print(f"time3 - time2: {(time3 - time2):.3f} s")
print(f"time4 - time3: {(time4 - time3):.3f} s")
print(f"time4 - time1: {(time4 - time1):.3f} s")



# plotting

from matplotlib.colors import LinearSegmentedColormap

# Custom colormap - brighter, more saturated colors
def make_physics_palette():

    colors = ['#5A0000', '#8B0000', '#B22222', '#C41E3A', '#DC143C',
              '#FF0000', '#FF4500', '#FF6347', '#FF7F00', '#FFA500',
              '#FFB300', '#FFC800', '#FFD700', '#FFE600', '#FFF000',
              '#FFFF00']

    cmap = LinearSegmentedColormap.from_list('physics', colors, N=256)
    return cmap

PHYSICS_CMAP = make_physics_palette()

def plot_2d_colormap(rho_avg_cdc_3d, level):
    """
    Plot a 2D colormap of rho_avg_cdc_3d at the specified level.

    Args:
        rho_avg_cdc_3d (ndarray): 3D array of rho values
        level (str): level of the array to plot, one of "00", "01", "10", "11", "C", "dC"
    """
    # Get the level number from the level string
    level_num = {
        "00": 0,
        "01": 1,
        "10": 2,
        "11": 3,
        "C": 4,
        "dC": 5
    }.get(level, None)
    if level_num is None:
        raise ValueError("Invalid level parameter")

    # Set the appropriate range for the Z-axis based on the level parameter
    zlim = None
    if level_num in [0, 1, 2, 3]:
        zlim = (0, 1)
    elif level_num == 4:
        zlim = (-18.2, 18.2)
    elif level_num == 5:
        zlim = (-0.01, 0.003)

    # Plot the 2D colormap
    fig, ax = plt.subplots(figsize=(12, 8))
    mesh = ax.pcolormesh(eps0_grid, A_grid, rho_avg_cdc_3d[level_num], cmap=PHYSICS_CMAP, vmin=zlim[0], vmax=zlim[1])
    fig.colorbar(mesh, ax=ax, label=r'$\Delta \varphi$', pad=0.01, shrink=0.8)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\varepsilon_0$')
    ax.set_ylabel(r'$A$')
    #plt.title('2D Colormap of rho_avg_cdc_3d[{}]'.format(level))
    fig.tight_layout(pad=0.1)
    plt.show()




def plot_4_subplots(rho_avg_cdc_3d):
    """
    Plots a 2x2 grid of colormaps for levels 00, 01, 10, 11.
    All subplots share a common colorbar.

    Args:
        rho_avg_cdc_3d (ndarray): 3D array of rho values.
    """
    levels = ["00", "01", "10", "11"]
    level_nums = [0, 1, 2, 3]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    #fig.suptitle('Interferogram for different levels', fontsize=16)

    # Flatten the axes array for easy iteration
    axs = axs.flatten()

    zlim = (0, 1)
    
    # Use a single colormap instance for all subplots
    cmap = cc.cm['fire']
    
    # Create a single pcolormesh object to be used for the colorbar
    mesh = None

    for i, (level, level_num) in enumerate(zip(levels, level_nums)):
        ax = axs[i]
        mesh = ax.pcolormesh(eps0_grid, A_grid, rho_avg_cdc_3d[level_num], cmap=cmap, vmin=zlim[0], vmax=zlim[1])
        ax.set_aspect('equal')
        #ax.set_title(f'Level {level}')
        if i >= 2: # bottom row
            ax.set_xlabel(r'$\varepsilon_0$')
        if i % 2 == 0: # left column
            ax.set_ylabel(r'$A$')

    # Adjust layout to bring subplots closer
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Add a single colorbar for the entire figure.
    # The `fig.colorbar` function with the `ax` parameter pointing to all subplots
    # ensures that only one colorbar is drawn for the entire figure.
    fig.colorbar(mesh, ax=axs.tolist(), orientation='vertical', fraction=0.046, pad=0.01)
    
    plt.show()


plot_2d_colormap(rho_avg_cdc_3d, "dC")

plot_4_subplots(rho_avg_cdc_3d)



