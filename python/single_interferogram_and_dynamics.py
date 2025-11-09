########################################
# python/single_interferogram_and_dynamics.py
########################################

import os
import sys
import platform
import time

time1 = time.time()

from config import SimRunGridSingleMode
from simulation import run_simulation
from pathlib import Path

repo_path = Path(__file__).resolve().parent.parent

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




time2 = time.time()

'''
simr = SimRunGridMode(
    delta_C=0.00011608757555650906,
    GammaL0=50.0,
    GammaR0=12.0,
    Gamma_eg0=0.8,
    Gamma_phi0=3.6,
    nu=21,
    
    eps0_min=-0.006,
    eps0_max=0.006,
    A_min=0.0,
    A_max=0.01,
    N_points_target=1000000,
    N_steps_period=1000,
    N_periods=10,
    N_periods_avg=1,
    
    quasi_static_ensemble_dephasing_flag=False,
    sigma_eps=None,
    N_samples_noise=None,
    
    platform_type = platform_type,
    repo_path=repo_path
)

'''

simr = SimRunGridSingleMode(
    delta_C=0.00011608757555650906,
    GammaL0=50.0,
    GammaR0=12.0,
    Gamma_eg0=0.8,
    Gamma_phi0=None,
    nu=21,
    
    eps0_min=-0.006,
    eps0_max=0.006,
    A_min=0.0,
    A_max=0.01,
    eps0_target_singlepoint=0.001,
    A_target_singlepoint=0.003,
    N_points_target=50000,
    N_steps_period=1000,
    N_periods=10,
    N_periods_avg=1,
    
    quasi_static_ensemble_dephasing_flag=True,
    sigma_eps=0.0001,
    N_samples_noise=10,
    
    platform_type = platform_type,
    repo_path=repo_path
)

time3 = time.time()

eps0_grid, A_grid, rho_avg_cdc_3d, time_dynamics, eps_dynamics, rho_dynamics, rho_avg, returncode = run_simulation(simr)

time4 = time.time()

print(f"time2 - time1: {(time2 - time1):.3f} s")
print(f"time3 - time2: {(time3 - time2):.3f} s")
print(f"time4 - time3: {(time4 - time3):.3f} s")
print(f"time4 - time1: {(time4 - time1):.3f} s")








