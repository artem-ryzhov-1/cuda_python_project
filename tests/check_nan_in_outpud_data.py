########################################
# tests/check_nan_in_outpud_data.py
########################################

import numpy as np
from pathlib import Path
import sys

# --- Path to project root ---
project_root = Path(__file__).resolve().parent.parent

# --- Make python/ importable ---
sys.path.insert(0, str(project_root))

# --- Imports ---
from python.file_io import (
    read_bin_file_singlemode,
    read_bin_file_gridmode_and_calculate_deriv
)

# --- Output path ---
output_dir = project_root / "cuda" / "output"



time_dynamics, eps_dynamics, rho_dynamics, rho_avg = read_bin_file_singlemode(output_dir / "rho_dynamics_single_mode_out.bin", 1.0)

eps0_grid, A_grid, result = read_bin_file_gridmode_and_calculate_deriv(output_dir / "rho_avg_out.bin", 1.0)



def has_nan_or_none(arr):
    # Check NaN safely
    try:
        has_nan = np.isnan(arr).any()
    except Exception:
        # fallback for object arrays
        has_nan = any(isinstance(x, float) and np.isnan(x) 
                      for x in arr.flatten())
    
    # Check None safely
    if arr.dtype == object:
        has_none = any(x is None for x in arr.flatten())
    else:
        has_none = False  # numeric arrays cannot contain None
    
    return has_nan or has_none



any_nan_or_none = False

for array in [time_dynamics, eps_dynamics, rho_dynamics, rho_avg,
              eps0_grid, A_grid, result]:
    
    any_nan_or_none = any_nan_or_none or has_nan_or_none(array)

print("NaN or None found in output data:", any_nan_or_none)