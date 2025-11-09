########################################
# python/file_io.py
########################################

import numpy as np

def read_bin_file_gridmode_and_calculate_deriv(path_output_bin_file):
    
    with open(path_output_bin_file, 'rb') as f:
        header = np.frombuffer(f.read(8), dtype=np.int32)
        n_A, n_eps0 = header[0], header[1]
        
        rho_avg_3d = np.fromfile(f, dtype=np.float32, count=4*n_A*n_eps0).reshape(4, n_eps0, n_A)
        
        A_grid = np.fromfile(f, dtype=np.float32, count=n_A)
        eps0_grid = np.fromfile(f, dtype=np.float32, count=n_eps0)
    
    
    # Preallocate final result
    result = np.empty((6, n_A, n_eps0), dtype=np.float32)
    
    # Transpose all at once (more cache-efficient)
    result[0:4] = rho_avg_3d.transpose(0, 2, 1)
    
    # Calculate C directly in output array (avoid temporary)
    np.subtract(result[1], result[2], out=result[4])  # p01 - p10
    result[4] += 18.0 * (result[0] - result[3])  # + 18*(p00 - p11)
    
    # Calculate dC
    deps0 = eps0_grid[1] - eps0_grid[0]
    result[5] = np.gradient(result[4], deps0, axis=1)
    
    return eps0_grid, A_grid, result



def read_bin_file_singlemode(path_output_bin_file):
    with open(path_output_bin_file, 'rb') as f:
        # Read header (N_steps_total)
        N_steps_total = np.frombuffer(f.read(4), dtype=np.int32)[0]

        # Read rho_avg (16 floats)
        rho_avg = np.fromfile(f, dtype=np.float32, count=16)

        # Read rho_dynamics (N_steps_total * 4 floats)
        rho_dynamics = np.fromfile(f, dtype=np.float32, count=N_steps_total * 4)
        rho_dynamics = rho_dynamics.reshape(N_steps_total, 4)

        # Read time_dynamics (N_steps_total floats)
        time_dynamics = np.fromfile(f, dtype=np.float32, count=N_steps_total)

        # Read eps_dynamics (N_steps_total floats)
        eps_dynamics = np.fromfile(f, dtype=np.float32, count=N_steps_total)

    return time_dynamics, eps_dynamics, rho_dynamics, rho_avg




