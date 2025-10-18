
from dataclasses import dataclass
from typing import List, Tuple, Any
#import numpy.typing as npt
from pathlib import Path
from numpy import nan, isnan

@dataclass
class SimulationConfig:
    grid_or_single_mode: str
    avg_periods_ouput_option: str
    ouput_option: str
    unrolled_option: str
    ram_shared_mmap_name: str
    single_mode_log_option: bool
    threads_per_traj_opt: str

    eps0_target_singlepoint: float
    A_target_singlepoint: float
    eps0_min: float
    eps0_max: float
    A_min: float
    A_max: float

    N_points_eps0_range: int
    N_points_A_range: int
    N_steps_period: int
    N_periods: int
    N_periods_avg: int
    N_samples_noise: int

    delta_C: float
    delta_L: float
    delta_R: float

    alpha: float
    nu: float

    rho00_init: float
    rho11_init: float
    rho22_init: float
    rho33_init: float

    gamma: float
    g_en: float
    g_phi: float
    gL_en: float
    gL_phi: float
    gR_en: float
    gR_phi: float

    a: float
    m: float
    
    cuda_cwd: str
    cuda_program_path: str
    path_output_csv: str
    path_output_bin_file: str
    path_dynamics_grid_mode_output_hdf5_after_ram: str
    path_dynamics_single_mode_output_csv: str
    path_dynamics_single_mode_output_log_csv: str
    path_dynamics_single_mode_output_log_hdf5: str
    
    platform_type: str
    
    GammaL0:  float
    GammaR0:  float
    muL:      float
    muR:      float
    T_K:      float
    Gamma_eg0:  float
    omega_c:    float
    Gamma_phi0: float
    quasi_static_ensemble_dephasing_flag: bool
    sigma_eps: float
    




class SimRunGridMode:

    def __init__(self, *, delta_C, GammaL0, GammaR0, Gamma_eg0, Gamma_phi0,
                 eps0_min, eps0_max, A_min, A_max,
                 N_points_target, N_steps_period, N_periods, N_periods_avg,
                 quasi_static_ensemble_dephasing_flag, sigma_eps, N_samples_noise,
                 platform_type, repo_path):
        
        for name, val in [("delta_C", delta_C),
                          ("GammaL0", GammaL0),
                          ("GammaR0", GammaR0),
                          ("Gamma_eg0", Gamma_eg0),
                          ("eps0_min", eps0_min),
                          ("eps0_max", eps0_max),
                          ("A_min", A_min),
                          ("A_max", A_max)]:
            #if not isinstance(val, (float, np.floating, int, np.integer)):
            if not isinstance(val, (float, int)):
                raise TypeError(f"{name} must be a float, got {val} ({type(val)})")
        
        for name, val in [("Gamma_phi0", Gamma_phi0),
                          ("sigma_eps", sigma_eps)]:
            if not (isinstance(val, float) or val is None):
                raise TypeError(f"{name} must be an float, or None, got {val} ({type(val)})")        
        
        for name, val in [("N_steps_period", N_steps_period),
                          ("N_periods", N_periods),
                          ("N_periods_avg", N_periods_avg),
                          ("N_points_target", N_points_target)]:
            #if not isinstance(val, (int, np.integer)):
            if not isinstance(val, int):
                raise TypeError(f"{name} must be an integer, got {val} ({type(val)})")
        
        if not (isinstance(N_samples_noise, int) or N_samples_noise is None):
            raise TypeError(f"N_samples_noise must be an integer, or None, got {N_samples_noise} ({type(N_samples_noise)})")        
                
        if not isinstance(quasi_static_ensemble_dephasing_flag, bool):
            raise TypeError(f"quasi_static_ensemble_dephasing_flag must be a bool, got {quasi_static_ensemble_dephasing_flag} ({type(quasi_static_ensemble_dephasing_flag)})")        
    
        if not isinstance(platform_type, str):
            raise TypeError(f"platform_type must be a str, got {platform_type} ({type(platform_type)})")        
        
        if not isinstance(repo_path, Path):
            raise TypeError(f"repo_path must be an object pathlib.Path, got {repo_path} ({type(repo_path)})")
        
        # Store individually as attributes
        self.delta_C    = float(delta_C)
        self.GammaL0    = float(GammaL0)
        self.GammaR0    = float(GammaR0)
        self.Gamma_eg0  = float(Gamma_eg0)
        self.Gamma_phi0 = float(Gamma_phi0)
        self.sigma_eps  = float(sigma_eps) if sigma_eps is not None else None
        
        self.eps0_min = float(eps0_min)
        self.eps0_max = float(eps0_max)
        self.A_min    = float(A_min)
        self.A_max    = float(A_max)
        
        self.N_points_target = int(N_points_target)
        self.N_steps_period  = int(N_steps_period)
        self.N_periods       = int(N_periods)
        self.N_periods_avg   = int(N_periods_avg)
        self.N_samples_noise = int(N_samples_noise) if N_samples_noise is not None else None
        
        self.quasi_static_ensemble_dephasing_flag = quasi_static_ensemble_dephasing_flag
        self.platform_type = platform_type
        self.repo_path     = repo_path

        




class SimRunSingleMode:

    def __init__(self, *, delta_C, GammaL0, GammaR0, Gamma_eg0, Gamma_phi0,
                 eps0_target_singlepoint, A_target_singlepoint,
                 N_steps_period, N_periods, N_periods_avg,
                 quasi_static_ensemble_dephasing_flag, sigma_eps, N_samples_noise,
                 platform_type, repo_path):
        
        for name, val in [("delta_C", delta_C),
                          ("GammaL0", GammaL0),
                          ("GammaR0", GammaR0),
                          ("Gamma_eg0", Gamma_eg0),
                          ("eps0_target_singlepoint", eps0_target_singlepoint),
                          ("A_target_singlepoint", A_target_singlepoint)]:
            #if not isinstance(val, (float, np.floating, int, np.integer)):
            if not isinstance(val, (float, int)):
                raise TypeError(f"{name} must be a float, got {val} ({type(val)})")
        
        for name, val in [("Gamma_phi0", Gamma_phi0),
                          ("sigma_eps", sigma_eps)]:
            if not (isinstance(val, float) or val is None):
                raise TypeError(f"{name} must be an float, or None, got {val} ({type(val)})")        
        
        for name, val in [("N_steps_period", N_steps_period),
                          ("N_periods", N_periods),
                          ("N_periods_avg", N_periods_avg)]:
            #if not isinstance(val, (int, np.integer)):
            if not isinstance(val, int):
                raise TypeError(f"{name} must be an integer, got {val} ({type(val)})")
        
        if not (isinstance(N_samples_noise, int) or N_samples_noise is None):
            raise TypeError(f"N_samples_noise must be an integer, or None, got {N_samples_noise} ({type(N_samples_noise)})")        
        
        if not isinstance(quasi_static_ensemble_dephasing_flag, bool):
            raise TypeError(f"quasi_static_ensemble_dephasing_flag must be a bool, got {quasi_static_ensemble_dephasing_flag} ({type(quasi_static_ensemble_dephasing_flag)})")        
        
        if not isinstance(platform_type, str):
            raise TypeError(f"platform_type must be a str, got {platform_type} ({type(platform_type)})")        
        
        if not isinstance(repo_path, Path):
            raise TypeError(f"repo_path must be an object pathlib.Path, got {repo_path} ({type(repo_path)})")
        
        # Store individually as attributes
        self.delta_C    = float(delta_C)
        self.GammaL0    = float(GammaL0)
        self.GammaR0    = float(GammaR0)
        self.Gamma_eg0  = float(Gamma_eg0)
        self.Gamma_phi0 = float(Gamma_phi0)
        self.sigma_eps  = float(sigma_eps) if sigma_eps is not None else None
        
        self.eps0_target_singlepoint = float(eps0_target_singlepoint)
        self.A_target_singlepoint = float(A_target_singlepoint)
        
        self.N_steps_period  = int(N_steps_period)
        self.N_periods       = int(N_periods)
        self.N_periods_avg   = int(N_periods_avg)
        self.N_samples_noise = int(N_samples_noise) if N_samples_noise is not None else None
        
        self.quasi_static_ensemble_dephasing_flag = quasi_static_ensemble_dephasing_flag
        self.platform_type = platform_type
        self.repo_path     = repo_path







