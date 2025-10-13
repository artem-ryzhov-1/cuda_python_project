
from dataclasses import dataclass
from typing import List, Tuple, Any
import numpy as np
#import numpy.typing as npt

@dataclass
class SimulationConfig:
    single_point_mode_flag: str
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
    
    cuda_program_path: str
    path_output_csv: str
    path_output_bin_file: str
    path_dynamics_grid_mode_output_hdf5_after_ram: str
    path_dynamics_single_mode_output_csv: str
    path_dynamics_single_mode_output_log_csv: str
    path_dynamics_single_mode_output_log_hdf5: str
    
    environment: str
    
    GammaL0:  float
    GammaR0:  float
    muL:      float
    muR:      float
    T_K:      float
    Gamma_eg0:  float
    omega_c:    float
    Gamma_phi0: float
    






class SimRun:

    def __init__(self, *, delta_C, GammaL0, GammaR0, Gamma_eg0, Gamma_phi0,
                 eps0_min, eps0_max, A_min, A_max,
                 N_points_target, N_steps_period, N_periods, N_periods_avg):
        
        for name, val in [("N_steps_period", N_steps_period),
                          ("N_periods", N_periods),
                          ("N_periods_avg", N_periods_avg),
                          ("N_points_target", N_points_target)]:
            if not (isinstance(val, np.int32) or isinstance(val, int)):
                raise TypeError(f"{name} must be an integer, got {val} ({type(val)})")
        
        # Store individually as attributes
        self.delta_C = float(delta_C)
        self.GammaL0 = float(GammaL0)
        self.GammaR0 = float(GammaR0)
        self.Gamma_eg0 = float(Gamma_eg0)
        self.Gamma_phi0 = float(Gamma_phi0)
        
        self.eps0_min = float(eps0_min)
        self.eps0_max = float(eps0_max)
        self.A_min = float(A_min)
        self.A_max = float(A_max)
        
        self.N_points_target = int(N_points_target)
        self.N_steps_period = int(N_steps_period)
        self.N_periods = int(N_periods)
        self.N_periods_avg = int(N_periods_avg)
        











