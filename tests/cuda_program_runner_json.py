########################################
# tests/cuda_program_runner_json.py
########################################

# ===============================================
# RUN PRECOMPILED CUDA PROGRAM WITH JSON CONFIG
# ===============================================

import os
import json
import subprocess
from pathlib import Path
import sys

# -------------------------------
# 1. PROJECT ROOT
# -------------------------------

# The repo root is two levels up from the tests folder
repo_path = Path(__file__).resolve().parents[1]
print("Repo path:", repo_path)

# Paths for CUDA input/output
cuda_dir = repo_path / "cuda"
input_dir = cuda_dir / "input"
output_dir = cuda_dir / "output"

# Ensure input/output folders exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# remove rho_avg_out.bin if it exists
(output_dir / "rho_avg_out.bin").unlink(missing_ok=True)
# remove rho_dynamics_single_mode_out.bin if it exists
(output_dir / "rho_dynamics_single_mode_out.bin").unlink(missing_ok=True)

# -------------------------------
# 2. CREATE JSON CONFIG
# -------------------------------


config = {
  "grid_single_mode": "grid_single",
  "ouput_option": "bin_file",
  "unrolled_option": "unrolled",
  "ram_shared_mmap_name": "MySimSharedMemory",
  "single_mode_log_option": True,
  "threads_per_traj_opt": "one_thread_per_traj",
  "eps0_target_singlepoint": 0.001,
  "A_target_singlepoint": 0.003,
  "eps0_min": -0.006,
  "eps0_max": 0.006,
  "A_min": 0.0,
  "A_max": 0.01,
  "N_points_eps0_range": 245,
  "N_points_A_range": 204,
  "N_steps_period": 1000,
  "N_periods": 10,
  "N_periods_avg": 1,
  "N_samples_noise": None,
  "delta_C": 0.00011608757555650906,
  "nu": 21.0,
  "rho00_init": 0.0,
  "rho11_init": 1.0,
  "rho22_init": 0.0,
  "rho33_init": 0.0,
  "path_output_csv": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_avg_out.csv",
  "path_output_bin_file_gridmode": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_avg_out.bin",
  "path_output_bin_file_singlemode": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_dynamics_single_mode_out.bin",
  "path_dynamics_single_mode_output_csv": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_dynamics_single_mode_out.csv",
  "path_dynamics_single_mode_output_log_csv": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_dynamics_single_mode_log_out.csv",
  "path_dynamics_single_mode_output_log_hdf5": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_dynamics_single_mode_log_out.bin",
  "GammaL0": 50.0,
  "GammaR0": 12.0,
  "Gamma_eg0": 0.8,
  "omega_c": 0.0015731484686413405,
  "Gamma_phi0": 3.6,
  "quasi_static_ensemble_dephasing_opt": "false",
  "sigma_eps": None,
  "version": "1.0"
}

'''
config = {
  "grid_single_mode": "grid_single",
  "ouput_option": "bin_file",
  "unrolled_option": "unrolled",
  "ram_shared_mmap_name": "MySimSharedMemory",
  "single_mode_log_option": True,
  "threads_per_traj_opt": "one_thread_per_traj",
  "eps0_target_singlepoint": 0.001,
  "A_target_singlepoint": 0.003,
  "eps0_min": -0.006,
  "eps0_max": 0.006,
  "A_min": 0.0,
  "A_max": 0.01,
  "N_points_eps0_range": 245,
  "N_points_A_range": 204,
  "N_steps_period": 1000,
  "N_periods": 10,
  "N_periods_avg": 1,
  "N_samples_noise": 10,
  "delta_C": 0.00011608757555650906,
  "nu": 21.0,
  "rho00_init": 0.25,
  "rho11_init": 0.25,
  "rho22_init": 0.25,
  "rho33_init": 0.25,
  "path_output_csv": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_avg_out.csv",
  "path_output_bin_file_gridmode": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_avg_out.bin",
  "path_output_bin_file_singlemode": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_dynamics_single_mode_out.bin",
  "path_dynamics_single_mode_output_csv": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_dynamics_single_mode_out.csv",
  "path_dynamics_single_mode_output_log_csv": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_dynamics_single_mode_log_out.csv",
  "path_dynamics_single_mode_output_log_hdf5": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_dynamics_single_mode_log_out.bin",
  "GammaL0": 50.0,
  "GammaR0": 12.0,
  "Gamma_eg0": 0.8,
  "omega_c": 0.0015731484686413405,
  "Gamma_phi0": None,
  "quasi_static_ensemble_dephasing_opt": "sequential",
  "sigma_eps": 0.0001,
  "version": "1.0"
}
'''



'''
config = {
  "grid_single_mode": "single",
  "ouput_option": "bin_file",
  "unrolled_option": "unrolled",
  "ram_shared_mmap_name": "MySimSharedMemory",
  "single_mode_log_option": True,
  "threads_per_traj_opt": "one_thread_per_traj",
  "eps0_target_singlepoint": 0.001,
  "A_target_singlepoint": 0.003,
  "eps0_min": None,
  "eps0_max": None,
  "A_min": None,
  "A_max": None,
  "N_points_eps0_range": None,
  "N_points_A_range": None,
  "N_steps_period": 1000,
  "N_periods": 10,
  "N_periods_avg": 1,
  "N_samples_noise": None,
  "delta_C": 0.00011608757555650906,
  "nu": 21.0,
  "rho00_init": 0.25,
  "rho11_init": 0.25,
  "rho22_init": 0.25,
  "rho33_init": 0.25,
  "path_output_csv": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_avg_out.csv",
  "path_output_bin_file_gridmode": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_avg_out.bin",
  "path_output_bin_file_singlemode": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_dynamics_single_mode_out.bin",
  "path_dynamics_single_mode_output_csv": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_dynamics_single_mode_out.csv",
  "path_dynamics_single_mode_output_log_csv": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_dynamics_single_mode_log_out.csv",
  "path_dynamics_single_mode_output_log_hdf5": "C:\\Users\\E-Store\\cuda_python_project\\cuda\\output\\rho_dynamics_single_mode_log_out.bin",
  "GammaL0": 420.0,
  "GammaR0": 68.0,
  "Gamma_eg0": 10.0,
  "omega_c": 0.0015731484686413405,
  "Gamma_phi0": 3.6,
  "quasi_static_ensemble_dephasing_opt": "false",
  "sigma_eps": None,
  "version": "2.0"
}
'''

# Save JSON
config_path = input_dir / "run_config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print("Config written to:", config_path)

# -------------------------------
# 3. RUN THE PROGRAM
# -------------------------------

# Determine binary path depending on platform
if sys.platform.startswith("win"):
    binary_path = cuda_dir / "bin" / "lindblad_gpu.exe"
else:
    binary_path = cuda_dir / "bin" / "lindblad_gpu"

if not binary_path.exists():
    raise FileNotFoundError(f"CUDA binary not found at {binary_path}")

# Change working directory to repo root
os.chdir(repo_path)
print("cwd:", os.getcwd())

cmd = f"{binary_path} {config_path}"
print("Executing:", cmd)

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

print("\n=== STDOUT ===")
print(result.stdout)
print("\n=== STDERR ===")
print(result.stderr)
print("\n=== Program executed ===")
