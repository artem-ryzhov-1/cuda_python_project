########################################
# plots/single_dynamics.py
########################################

import numpy as np
import os
import sys
import platform
import time
from pathlib import Path

time1 = time.time()

# --- Imports ---
import _setup_paths
from app.python.config import SimRunSingleMode
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

parameters_from_experiment = False # "True", "False"
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


regime = "single" # "single" "double" "multi" "incoherent"

if regime == "single":
    eps0_target_singlepoint=0.0
    A_target_singlepoint=0.008
elif regime == "double":
    eps0_target_singlepoint=0.0025
    A_target_singlepoint=0.0055
elif regime == "multi":
    eps0_target_singlepoint=0.00043
    A_target_singlepoint=0.0023
elif regime == "incoherent":
    eps0_target_singlepoint=0.0047
    A_target_singlepoint=0.0026


time2 = time.time()


# experimental paper parameters
if parameters_from_experiment:
    simr = SimRunSingleMode(
        delta_C=delta_C,
        GammaL0=50.0,
        GammaR0=12.5,
        Gamma_eg0=0.8,
        Gamma_phi0=3.6,
        nu=21,
        E_C=0.195,
        
        eps0_target_singlepoint=eps0_target_singlepoint,
        A_target_singlepoint=A_target_singlepoint,
        N_steps_period=1000,
        N_periods=5,
        N_periods_avg=1,
        
        quasi_static_ensemble_dephasing_flag=False,
        sigma_eps=None,
        N_samples_noise=None,
        
        rho00_init=0.0,
        rho11_init=1.0,
        rho22_init=0.0,
        rho33_init=0.0,
        
        platform_type = platform_type,
        repo_path=_setup_paths.PROJECT_ROOT
    )


elif (not parameters_from_experiment and not qs_flag):
    # fit parameters Lindblad dephasing
    
    simr = SimRunSingleMode(
        delta_C=delta_C,
        GammaL0=GammaL0,
        GammaR0=GammaR0,
        Gamma_eg0=Gamma_eg0,
        Gamma_phi0=Gamma_phi0,
        nu=nu,
        E_C=E_C,
        
        eps0_target_singlepoint=eps0_target_singlepoint,
        A_target_singlepoint=A_target_singlepoint,
        N_steps_period=1000,
        N_periods=3,
        N_periods_avg=1,
        
        quasi_static_ensemble_dephasing_flag=False,
        sigma_eps=None,
        N_samples_noise=None,
        
        rho00_init=0.0,
        rho11_init=1.0,
        rho22_init=0.0,
        rho33_init=0.0,
        
        platform_type = platform_type,
        repo_path=_setup_paths.PROJECT_ROOT
    )


elif (not parameters_from_experiment and qs_flag):
        
    simr = SimRunSingleMode(
        delta_C=delta_C,
        GammaL0=GammaL0,
        GammaR0=GammaR0,
        Gamma_eg0=Gamma_eg0,
        Gamma_phi0=Gamma_phi0,
        nu=nu,
        E_C=E_C,
        
        eps0_target_singlepoint=eps0_target_singlepoint,
        A_target_singlepoint=A_target_singlepoint,
        N_steps_period=1000,
        N_periods=3,
        N_periods_avg=1,
        
        quasi_static_ensemble_dephasing_flag=True,
        sigma_eps=sigma_eps,
        N_samples_noise=1000,
        
        rho00_init=0.0,
        rho11_init=1.0,
        rho22_init=0.0,
        rho33_init=0.0,
        
        platform_type = platform_type,
        repo_path=_setup_paths.PROJECT_ROOT
    )

else:
    raise RuntimeError("Unsupported regime")



time3 = time.time()

time_dynamics, eps_dynamics, rho_dynamics, rho_avg, returncode = run_simulation(simr)

time4 = time.time()

print(f"time2 - time1: {(time2 - time1):.3f} s")
print(f"time3 - time2: {(time3 - time2):.3f} s")
print(f"time4 - time3: {(time4 - time3):.3f} s")
print(f"time4 - time1: {(time4 - time1):.3f} s")






import matplotlib.pyplot as plt
import numpy as np

# Skip first element
time = time_dynamics[1:]
eps = eps_dynamics[1:]
rho = rho_dynamics[1:, :]

# Parameters

a=0.1
m=10
B = ((a + 2) * (m + 1)) / (a * (m - 1))
sqrt_term = np.sqrt(1 + m**2 * (B**2 - 1) * delta_C**2)
eps_L = (B + sqrt_term) / (m * (B**2 - 1))
eps_R = (B - sqrt_term) / (m * (B**2 - 1))

print("epsilon_L=",eps_L)
print("epsilon_R=",eps_R)

# Create figure with gridspec for relative heights
fig, (ax_upper, ax_lower) = plt.subplots(
    2, 1,
    figsize=(10, 6),
    gridspec_kw={'height_ratios': [2, 1]},
    sharex=True
)

# Upper subplot: populations
colors_upper = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
labels_upper = [r'$P_{00}$', r'$P_{01}$', r'$P_{10}$', r'$P_{11}$']

for i in range(4):
    ax_upper.plot(time, rho[:, i], label=labels_upper[i], color=colors_upper[i], linewidth=1.5)

ax_upper.set_ylabel(r'Populations', fontsize=12)
ax_upper.legend(#frameon=False,
                fontsize=10)
ax_upper.tick_params(axis='both', which='major', labelsize=10)
ax_upper.margins(x=0)

# Lower subplot: eps_dynamics
color_lower = plt.rcParams['axes.prop_cycle'].by_key()['color'][4]  # 5th color
ax_lower.plot(time, eps, color=color_lower, label=r'$\varepsilon / E_C$', linewidth=1.5)

ax_lower.set_ylabel(r'$\varepsilon / E_C$', fontsize=12)
ax_lower.set_xlabel(r'time [ps]', fontsize=12)
ax_lower.tick_params(axis='both', which='major', labelsize=10)

# Dashed lines
ax_lower.axhline(0, color='gray', linestyle='--', linewidth=1)
ax_lower.axhline(eps_L, color='red', linestyle='--', linewidth=1, label=r'$\pm \varepsilon_L$')
ax_lower.axhline(-eps_L, color='red', linestyle='--', linewidth=1)
ax_lower.axhline(eps_R, color='blue', linestyle='--', linewidth=1, label=r'$\pm \varepsilon_R$')
ax_lower.axhline(-eps_R, color='blue', linestyle='--', linewidth=1)

ax_lower.legend(#frameon=False,
                fontsize=10)
ax_lower.margins(x=0)

plt.tight_layout()
plt.show()












