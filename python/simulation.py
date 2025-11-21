########################################
# python/simulation.py
########################################

import os
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from pathlib import Path
import math
import platform
import sys


from config import SimulationConfig, SimRunGridMode, SimRunSingleMode, SimRunGridSingleMode
from cuda_runner import run_gpu_lindblad_program
from file_io import read_bin_file_gridmode_and_calculate_deriv, read_bin_file_singlemode


def compute_grid(N_points_target, eps0_min, eps0_max, A_min, A_max):
    eps0_range = eps0_max - eps0_min
    A_range = A_max - A_min

    if A_range == 0 or eps0_range == 0:
        raise ValueError("Axis ranges must be non-zero.")

    aspect_ratio = eps0_range / A_range

    N_A = round(math.sqrt(N_points_target / aspect_ratio))
    N_eps0 = round(N_A * aspect_ratio)

    # Recompute total just to be sure
    #total_points = N_A * N_eps0

    return N_eps0, N_A


def run_simulation(simr):
    
    if isinstance(simr, SimRunGridMode):
        print("grid mode")
        grid_single_mode = "grid"
        
        eps0_target_singlepoint = None
        A_target_singlepoint    = None
        
        eps0_min =  simr.eps0_min
        eps0_max =  simr.eps0_max
        A_min =     simr.A_min
        A_max =     simr.A_max
        
        N_points_eps0_range, N_points_A_range = compute_grid(N_points_target=simr.N_points_target,
                                                             eps0_min=eps0_min,
                                                             eps0_max=eps0_max,
                                                             A_min=A_min,
                                                             A_max=A_max)
        
    elif isinstance(simr, SimRunSingleMode):
        print("single mode")
        grid_single_mode = "single"
        
        eps0_target_singlepoint = simr.eps0_target_singlepoint
        A_target_singlepoint    = simr.A_target_singlepoint
        
        eps0_min =  None
        eps0_max =  None
        A_min =     None
        A_max =     None
        
        N_points_eps0_range = None
        N_points_A_range    = None
    
    elif isinstance(simr, SimRunGridSingleMode):
        print("grid-single mode")
        grid_single_mode = "grid_single"
        
        eps0_target_singlepoint = simr.eps0_target_singlepoint
        A_target_singlepoint    = simr.A_target_singlepoint
        
        eps0_min =  simr.eps0_min
        eps0_max =  simr.eps0_max
        A_min =     simr.A_min
        A_max =     simr.A_max
        
        N_points_eps0_range, N_points_A_range = compute_grid(N_points_target=simr.N_points_target,
                                                             eps0_min=eps0_min,
                                                             eps0_max=eps0_max,
                                                             A_min=A_min,
                                                             A_max=A_max)
    
    
    
    
    
    ram_shared_mmap_name = "MySimSharedMemory"  # can also generate dynamically if needed
    run_cuda_program_option = True
    
    unrolled_option = "unrolled" # "as_arrays" or "unrolled"
    
    ouput_option = "bin_file" # "ssd_csv" or "ram" or "bin_file"
    single_mode_log_option = False # boolean: "True" or "False"
    
    
    # "one_thread_per_traj" or "thread_group_in_warp_per_traj_shuffle" or "thread_group_in_warp_per_traj_shmem"
    threads_per_traj_opt = "one_thread_per_traj" 
    #threads_per_traj_opt = "thread_group_in_warp_per_traj_shuffle" 
    #threads_per_traj_opt = "thread_group_in_warp_per_traj_shmem" 
    
    

    

    
    #N_points_target = simr.N_points_target
    N_steps_period = simr.N_steps_period
    N_periods =      simr.N_periods
    N_periods_avg =  simr.N_periods_avg
    
    #N_points_target = 2500*10
    #N_steps_period = 10_000*1
    #N_periods =      5
    
    
    #delta = 0.00011608757555650906
    delta_C = simr.delta_C
    
    nu    = simr.nu
    
    
    #rho00_init = 0
    #rho11_init = 1
    #rho22_init = 0
    #rho33_init = 0
    
    rho00_init = 0.25
    rho11_init = 0.25
    rho22_init = 0.25
    rho33_init = 0.25
    
    
    # ===== Physics params =====
    
    
    GammaL0  = simr.GammaL0
    GammaR0  = simr.GammaR0
    #muL  = 0
    #muR  = 0
    #T_K  = 0
    
    Gamma_eg0 = simr.Gamma_eg0
    omega_c = 0.0015731484686413405
    
    Gamma_phi0 = simr.Gamma_phi0
    sigma_eps  = simr.sigma_eps    


    if simr.quasi_static_ensemble_dephasing_flag:
        quasi_static_ensemble_dephasing_opt = "sequential"
        #quasi_static_ensemble_dephasing_opt = "parallel"
    else:
        quasi_static_ensemble_dephasing_opt = "false"
    
    # paths
    
    
    #if system == "Windows":

    # Define the base output path
    #path_output = Path(r"S:/Physics/2025_DQD/cuda/programs/CUDA/set_of_interferograms/v4.1/output")
    #path_output = Path("R:/output")  # The path can still use forward slashes or backslashes on Windows
    

    
    repo_path  = simr.repo_path
    output_dir = repo_path / "cuda" / "output"
    cuda_cwd   = repo_path / "cuda" / "bin"
    
    if simr.platform_type == "local_windows":
        cuda_program_path = cuda_cwd / "lindblad_gpu.exe"
    elif simr.platform_type in ["colab_linux", "local_linux", "local_wsl2"]:
        cuda_program_path = cuda_cwd / "lindblad_gpu"
    
    ###################################
    
    #cuda_cwd = Path(r"C:\Users\E-Store\Documents\projects\repos\lindblad_cuda3\x64\Release")
    #cuda_program_path = cuda_cwd / "lindblad_cuda3.exe"
    #output_dir        = cuda_cwd / "output"
    
    ###################################
    
    
    print("cuda_program_path = ", cuda_program_path)
    print("output_dir = ", output_dir)
    print("cuda_cwd = ", cuda_cwd)
    
    # Combine the base path with the filenames
    
    path_output_bin_file_gridmode         = output_dir / "rho_avg_out.bin"
    path_output_bin_file_singlemode       = output_dir / "rho_dynamics_single_mode_out.bin"
        
    path_output_csv                           = output_dir / "rho_avg_out.csv"
    path_dynamics_single_mode_output_csv      = output_dir / "rho_dynamics_single_mode_out.csv"
    path_dynamics_single_mode_output_log_csv  = output_dir / "rho_dynamics_single_mode_log_out.csv"
    path_dynamics_single_mode_output_log_hdf5 = output_dir / "rho_dynamics_single_mode_log_out.h5"
    
    
    #path_dynamics_grid_mode_output_csv_after_ram  = output_dir / "rho_avg_out_after_ram.csv"
    #path_dynamics_grid_mode_output_hdf5_after_ram = output_dir / "rho_avg_out_after_ram.h5"
    
    
    #if simr.save_option == "save_hdf5":
    #    path_dynamics_grid_mode_output_hdf5_after_ram = simr.get_filepath()
    #elif simr.save_option=="onthefly":
    #    path_dynamics_grid_mode_output_hdf5_after_ram = None
    
    path_dynamics_grid_mode_output_hdf5_after_ram = None
    
    
    if simr.platform_type in ['colab_linux', 'local_linux', 'local_wsl2', 'local_macos']:
        # Check if executable
        if not os.access(cuda_program_path, os.X_OK):
            print("lindblad_gpu program is not executable. Changing permission...")
            os.chmod(cuda_program_path, 0o755)  # rwxr-xr-x
    
    ##########
    
    tr_pho = rho00_init + rho11_init + rho22_init + rho33_init
    
    rho00_init /= tr_pho;
    rho11_init /= tr_pho;
    rho22_init /= tr_pho;
    rho33_init /= tr_pho;
    

    
    
    
    
    
    
    ################################
    #checks
    
        
    if grid_single_mode != "grid" and grid_single_mode != "single" and grid_single_mode != "grid_single":
        raise ValueError("variable grid_single_mode should be grid or single or grid_single")
    
    if not(ouput_option == "bin_file"):
        raise ValueError("Other ouput_option not implemented yet")
    
    ################################################################
    # pass all needed variables into func:
    
    #########################################################    
    
    
    
    
    
    
    
    cfg = SimulationConfig(
        grid_single_mode=grid_single_mode,
        ouput_option=ouput_option,
        unrolled_option=unrolled_option,
        single_mode_log_option=single_mode_log_option,
        ram_shared_mmap_name=ram_shared_mmap_name,
        threads_per_traj_opt=threads_per_traj_opt,
        eps0_target_singlepoint=eps0_target_singlepoint,
        A_target_singlepoint=A_target_singlepoint,
        eps0_min=eps0_min,
        eps0_max=eps0_max,
        A_min=A_min,
        A_max=A_max,
        N_points_eps0_range=N_points_eps0_range,
        N_points_A_range=N_points_A_range,
        N_steps_period=N_steps_period,
        N_periods=N_periods,
        N_periods_avg=N_periods_avg,
        delta_C=delta_C,
        nu=nu,
        rho00_init=rho00_init,
        rho11_init=rho11_init,
        rho22_init=rho22_init,
        rho33_init=rho33_init,
        cuda_cwd=cuda_cwd,
        cuda_program_path=cuda_program_path,
        path_output_csv=path_output_csv,
        path_output_bin_file_gridmode=path_output_bin_file_gridmode,
        path_output_bin_file_singlemode=path_output_bin_file_singlemode,
        path_dynamics_grid_mode_output_hdf5_after_ram=path_dynamics_grid_mode_output_hdf5_after_ram,
        path_dynamics_single_mode_output_csv=path_dynamics_single_mode_output_csv,
        path_dynamics_single_mode_output_log_csv=path_dynamics_single_mode_output_log_csv,
        path_dynamics_single_mode_output_log_hdf5=path_dynamics_single_mode_output_log_hdf5,
        platform_type=simr.platform_type,
        
        GammaL0=GammaL0,
        GammaR0=GammaR0,
        #muL=muL,
        #muR=muR,
        #T_K=T_K,
        
        Gamma_eg0=Gamma_eg0,
        omega_c = omega_c,
        
        Gamma_phi0=Gamma_phi0,
        
        quasi_static_ensemble_dephasing_opt=quasi_static_ensemble_dephasing_opt,
        sigma_eps=sigma_eps,
        N_samples_noise=simr.N_samples_noise
    )
    
    ######################################################
    
    
    
    
    #print("N_points_target =", N_points_target)
    #print("N_points =", N_points_eps0_range*N_points_A_range)
    #print("N_points_eps0_range =", N_points_eps0_range)
    #print("N_points_A_range =", N_points_A_range)
    
    
    
    
    # do not run many times, because the log file is 0.5 GB
    
    if run_cuda_program_option == True:
        
        returncode = run_gpu_lindblad_program(cfg)
        
        if grid_single_mode == "grid":
                 
            eps0_grid, A_grid, rho_avg_cdc_3d = read_bin_file_gridmode_and_calculate_deriv(path_output_bin_file_gridmode)
        
            return eps0_grid, A_grid, rho_avg_cdc_3d, returncode
        
        
        elif grid_single_mode == "single":
        
            time_dynamics, eps_dynamics, rho_dynamics, rho_avg = read_bin_file_singlemode(path_output_bin_file_singlemode)
            
            return time_dynamics, eps_dynamics, rho_dynamics, rho_avg, returncode
        
        elif grid_single_mode == "grid_single":
            
            eps0_grid, A_grid, rho_avg_cdc_3d = read_bin_file_gridmode_and_calculate_deriv(path_output_bin_file_gridmode)
            time_dynamics, eps_dynamics, rho_dynamics, rho_avg = read_bin_file_singlemode(path_output_bin_file_singlemode)
            
            return eps0_grid, A_grid, rho_avg_cdc_3d, time_dynamics, eps_dynamics, rho_dynamics, rho_avg, returncode

    else:
        return








