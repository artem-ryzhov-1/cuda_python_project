import numpy as np
from pathlib import Path


output_dir = Path(r"C:\Users\E-Store\cuda_python_project\cuda\output")


time_dynamics, eps_dynamics, rho_dynamics, rho_avg = read_bin_file_singlemode(output_dir / "rho_dynamics_single_mode_out.bin")


eps0_grid, A_grid, result = read_bin_file_gridmode_and_calculate_deriv(output_dir / "rho_avg_out.bin")



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

