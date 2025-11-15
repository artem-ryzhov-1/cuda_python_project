########################################
# python/cuda_runner.py
########################################

from pathlib import Path
import json

import time
import subprocess
import threading
import signal

from config import SimulationConfig


def run_gpu_lindblad_program(c: SimulationConfig):
    
    """
    Runs the CUDA Lindblad executable with parameters from SimulationConfig `c`.
    
    Uses JSON config file instead of command-line arguments.
    Streams output live to console, supports graceful termination via ENTER or q.
    
    Raises subprocess.CalledProcessError on non-zero exit code, including full output.
    
    Returns:
        int: subprocess return code.
    """
        
    if c.platform_type == 'local_windows':
        import msvcrt
    
    print("\n==============================================")
    
    # Create configuration dictionary
    config = {
        "version": "1.0",  # For future compatibility
        
        # Mode and output options
        "grid_single_mode": c.grid_single_mode,
        "avg_periods_ouput_option": c.avg_periods_ouput_option,
        "ouput_option": c.ouput_option,
        "unrolled_option": c.unrolled_option,
        "ram_shared_mmap_name": c.ram_shared_mmap_name,
        
        # Parameter ranges
        "eps0_min": c.eps0_min,
        "eps0_max": c.eps0_max,
        "A_min": c.A_min,
        "A_max": c.A_max,
        "N_points_eps0_range": c.N_points_eps0_range,
        "N_points_A_range": c.N_points_A_range,
        
        # Simulation parameters
        "N_steps_period": c.N_steps_period,
        "N_periods": c.N_periods,
        "N_periods_avg": c.N_periods_avg,
        "N_samples_noise": c.N_samples_noise,
        
        # Physical parameters
        "alpha": c.alpha,
        "nu": c.nu,
        "eps0_target_singlepoint": c.eps0_target_singlepoint,
        "A_target_singlepoint": c.A_target_singlepoint,
        
        # Initial density matrix
        "rho00_init": c.rho00_init,
        "rho11_init": c.rho11_init,
        "rho22_init": c.rho22_init,
        "rho33_init": c.rho33_init,
        
        # Output paths
        "path_output_csv": str(c.path_output_csv),
        "path_output_bin_file_gridmode": str(c.path_output_bin_file_gridmode),
        "path_output_bin_file_singlemode": str(c.path_output_bin_file_singlemode),
        "path_dynamics_single_mode_output_csv": str(c.path_dynamics_single_mode_output_csv),
        
        # Delta parameters
        "delta_C": c.delta_C,
        "delta_L": c.delta_L,
        "delta_R": c.delta_R,
        
        # Coupling parameters
        "g_en": c.g_en,
        "g_phi": c.g_phi,
        "gL_en": c.gL_en,
        "gL_phi": c.gL_phi,
        "gR_en": c.gR_en,
        "gR_phi": c.gR_phi,
        
        # Reservoir parameters
        "GammaL0": c.GammaL0,
        "GammaR0": c.GammaR0,
        "muL": c.muL,
        "muR": c.muR,
        "T_K": c.T_K,
        
        # Phonon parameters
        "Gamma_eg0": c.Gamma_eg0,
        "omega_c": c.omega_c,
        
        # Dephasing parameters
        "Gamma_phi0": c.Gamma_phi0,
        "sigma_eps": c.sigma_eps,
        
        # Logging options
        "single_mode_log_option": c.single_mode_log_option,
        "path_dynamics_single_mode_output_log_csv": str(c.path_dynamics_single_mode_output_log_csv),
        "path_dynamics_single_mode_output_log_hdf5": str(c.path_dynamics_single_mode_output_log_hdf5),
        
        # Performance options
        "threads_per_traj_opt": c.threads_per_traj_opt,
        "quasi_static_ensemble_dephasing_flag": c.quasi_static_ensemble_dephasing_flag
    }
    
    # Write config to JSON file
    config_path = Path(c.cuda_cwd) / "run_config.json"
    print(f"Writing configuration to: {config_path}")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved with {len(config)} parameters")
    print("==============================================")
    
    # Build argument list (now just program path + config file)
    args = [
        str(c.cuda_program_path),
        str(config_path)
    ]
    
    print("Args being passed to CUDA program:", args)
    print(f"Total arguments: {len(args)}")
    print("==============================================")
    print("CUDA program run started. Console output:")
    print()
    
  
    # Launch CUDA program with output capture
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=c.cuda_cwd,
        bufsize=1,  # line-buffered
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if c.platform_type == 'local_windows' else 0  # Allow sending CTRL_C_EVENT
    )
    
    output_lines = []

    
    # previous
    def output_reader():
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            print(line, end='', flush=True)
            output_lines.append(line)



    
    
    output_thread = threading.Thread(target=output_reader)
    output_thread.daemon = True
    output_thread.start()
        
    if c.platform_type in ['local_windows', 'local_linux', 'local_wsl2']:  
        # Optional: handle user input for stopping
        try:
            print("CUDA program running. Press 'q' to stop...")
            while True:
                if proc.poll() is not None:
                    break
                
                if c.platform_type == 'local_windows':
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        #print(f"Key pressed: {key}")  # Debug print
                        if key.lower() == b'q':
                            print("\n'q' pressed. Stopping CUDA program. Sending CTRL_C_EVENT...")
                            proc.send_signal(signal.CTRL_C_EVENT)
                            break
                else:
                    key = input()  # Linux/WSL2/Colab
                    if key.lower() == 'q':
                        print("\n'q' pressed. Stopping CUDA program. Sending SIGINT...")
                        proc.send_signal(signal.SIGINT)
                        break
    
                time.sleep(0.001)  # sleep to reduce CPU hogging
                
                
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received in Python. Stopping CUDA program. Sending CTRL_C_EVENT...")
            if c.platform_type == 'local_windows':
                proc.send_signal(signal.CTRL_C_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
    
    elif c.platform_type == "colab_linux":
        print("CUDA program running (non-interactive mode)...")

    # Finalize
    proc.wait()
    output_thread.join()
    proc.stdout.close()  
    
    
    # Raise an exception if the program failed
    if proc.returncode == 0:
        print("CUDA program run finished successfully. Console output closed.")
    elif proc.returncode == 0xC000013A:  # CTRL+C
        print("CUDA program was interrupted by CTRL+C or signal. Exiting gracefully.")
    else:
        print(f"CUDA program exited with error code {proc.returncode}")
        full_output = "".join(output_lines)
        raise subprocess.CalledProcessError(proc.returncode, args, output=full_output)

    
    print("==============================================")
    print()
    
    return proc.returncode








