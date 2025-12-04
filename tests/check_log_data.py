########################################
# checks/check_log_data.py
########################################

import numpy as np
import pandas as pd
import mpmath
from mpmath import mpf
from dataclasses import dataclass, field
from typing import Tuple


import h5py
import os
import warnings
import sympy as sp
from pathlib import Path
import sys

import matplotlib.pyplot as plt


#np.set_printoptions(linewidth=175)


#import _setup_paths
#from tests.log_reader import read_log_binary


PROJECT_ROOT = Path(r"C:\Users\E-Store\cuda_python_project")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CUDA_INPUT = PROJECT_ROOT / "app" / "cuda" / "input"
CUDA_OUTPUT = PROJECT_ROOT / "app" / "cuda" / "output"
CUDA_BIN = PROJECT_ROOT / "app" / "cuda" / "bin"

from tests.log_reader import read_log_binary
from app.python.config import SimRunSingleMode
from app.python.simulation import run_simulation

path_dynamics_single_mode_output_log_bin = CUDA_OUTPUT / "rho_dynamics_single_mode_log_out.bin"



###############################

prec = 100

# Set precision (bits)(default 53 ~ double precision)
mpmath.mp.dps = int(prec * 0.30103)  # Convert bits to decimal digits (approx)


one = mpmath.mpf('1')
zero = mpmath.mpf('0')
half = mpmath.mpf('0.5')
minus_i = mpmath.mpc('0', '-1')  # 0 - 1j


# Helper to convert safely from any numeric type to high-precision mpf
def mp(x):
    return mpmath.mpf(str(x))

def mp_safe(x):
    """Safely convert NumPy scalar or regular numeric to mpmath.mpf"""
    if isinstance(x, (np.floating, np.integer)):
        return mpmath.mpf(float(x))
    return mpmath.mpf(str(x))



###############################






@dataclass
class DynamicsParamsMpf:
    """
    Parameters for lindblad_single_trajectory_adb kernel launch.
    Direct mapping to kernel arguments.
    """
    
    
    # Hamiltonian parameters
    delta_C_prime: mpf  # (in E_C)
    epsilon0_prime: mpf  # (in E_C)
    A_prime: mpf  # (in E_C)
    nu_phys: mpf  # (in GHz)
    E_C_phys: mpf # (in eV)
    
    m: mpf
    a: mpf
    
    # Ralxation parameters (in GHz):
    Gamma_L0_phys: mpf
    Gamma_R0_phys: mpf
    Gamma_eg0_phys: mpf
    Gamma_phi0_phys: mpf
    
    N_steps_per_period: int
    N_periods: int
    
    rho_init: Tuple[mpf, mpf, mpf, mpf]  # Initial state (rho00, rho11, rho22, rho33)
    
    # ---- Derived parameter (computed once) ----
    one_div_m: mpf = field(init=False)
    B: mpf = field(init=False)
    omega_phys: mpf = field(init=False)
    dt: mpf = field(init=False)
    
    epsilon_L: mpf = field(init=False)
    epsilon_R: mpf = field(init=False)
    
    HBAR_div_E_C: mpf = field(init=False)
    
    HBAR: mpf = mpf('6.582119569e-16')
    

    def __post_init__(self):
        
        self.one_div_m = 1 / self.m
        
        # B = (1 / (a * (m - 1))) * (a + 2) * (m + 1)
        self.B = (1 / (self.a * (self.m - 1))) * (self.a + 2) * (self.m + 1)    

        # omega = 2 * pi * nu
        self.omega_phys = 2 * mpmath.pi * self.nu_phys * mpf('1e9')
        
        
        self.HBAR_div_E_C = self.HBAR / self.E_C_phys
        
        T_phys = one / (self.nu_phys * mpf('1e9'));  # seconds
        T_dimless = T_phys / self.HBAR_div_E_C;
        self.dt_prime = T_dimless / self.N_steps_per_period


        self.omega_prime = self.HBAR_div_E_C * self.omega_phys
        self.Gamma_L0_prime = self.HBAR_div_E_C * self.Gamma_L0_phys * mpf('1e9')
        self.Gamma_R0_prime = self.HBAR_div_E_C * self.Gamma_R0_phys * mpf('1e9')
        self.Gamma_eg0_prime = self.HBAR_div_E_C * self.Gamma_eg0_phys * mpf('1e9')
        self.Gamma_phi0_prime = self.HBAR_div_E_C * self.Gamma_phi0_phys * mpf('1e9')
        
        # ---- Compute epsilon_L and epsilon_R ----
        sqrt_term = np.sqrt(1 + self.m**2 * (self.B**2 - 1) * self.delta_C_prime**2)
        denom = self.m * (self.B**2 - 1)
        
        self.epsilon_L = (self.B + sqrt_term) / denom
        self.epsilon_R = (self.B - sqrt_term) / denom



params = DynamicsParamsMpf(
    # Hamiltonian parameters
    delta_C_prime=mpf('0.0005'),
    epsilon0_prime=mpf('0.001'),
    A_prime=mpf('0.006'),
    nu_phys=mpf('21'),
    E_C_phys=mpf('0.2'),
    
    m=mpf('10'),
    a=mpf('0.1'),
    
    N_steps_per_period=500,
    N_periods=1,
    
    # Relaxation parameters:
    Gamma_L0_phys=mpf('50'),
    Gamma_R0_phys=mpf('12'),
    Gamma_eg0_phys=mpf('0.8'),
    Gamma_phi0_phys=mpf('3.6'),
    
    rho_init=(mpf('0.0'), mpf('1.0'), mpf('0.0'), mpf('0.0'))
)

###############################


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


simr_without_log = SimRunSingleMode(
    delta_C=float(params.delta_C_prime),
    GammaL0=float(params.Gamma_L0_phys),
    GammaR0=float(params.Gamma_R0_phys),
    Gamma_eg0=float(params.Gamma_eg0_phys),
    Gamma_phi0=float(params.Gamma_phi0_phys),
    nu=float(params.nu_phys),
    E_C=float(params.E_C_phys),
    
    eps0_target_singlepoint=float(params.epsilon0_prime),
    A_target_singlepoint=float(params.A_prime),
    N_steps_period=params.N_steps_per_period,
    N_periods=params.N_periods,
    N_periods_avg=1,
    
    quasi_static_ensemble_dephasing_flag=False,
    sigma_eps=None,
    N_samples_noise=None,
    
    rho00_init=float(params.rho_init[0]),
    rho11_init=float(params.rho_init[1]),
    rho22_init=float(params.rho_init[2]),
    rho33_init=float(params.rho_init[3]),
    
    platform_type = platform_type,
    repo_path=PROJECT_ROOT
)


simr_with_log = SimRunSingleMode(
    delta_C=float(params.delta_C_prime),
    GammaL0=float(params.Gamma_L0_phys),
    GammaR0=float(params.Gamma_R0_phys),
    Gamma_eg0=float(params.Gamma_eg0_phys),
    Gamma_phi0=float(params.Gamma_phi0_phys),
    nu=float(params.nu_phys),
    E_C=float(params.E_C_phys),
    
    eps0_target_singlepoint=float(params.epsilon0_prime),
    A_target_singlepoint=float(params.A_prime),
    N_steps_period=params.N_steps_per_period,
    N_periods=params.N_periods,
    N_periods_avg=1,
    
    quasi_static_ensemble_dephasing_flag=False,
    sigma_eps=None,
    N_samples_noise=None,
    
    rho00_init=float(params.rho_init[0]),
    rho11_init=float(params.rho_init[1]),
    rho22_init=float(params.rho_init[2]),
    rho33_init=float(params.rho_init[3]),
    
    platform_type = platform_type,
    repo_path=PROJECT_ROOT,
    single_mode_log_option=True
)


# remove rho_avg_out.bin if it exists
(CUDA_OUTPUT / "rho_avg_out.bin").unlink(missing_ok=True)
# remove rho_dynamics_single_mode_out.bin if it exists
(CUDA_OUTPUT / "rho_dynamics_single_mode_out.bin").unlink(missing_ok=True)
# remove rho_dynamics_single_mode_log_out.bin if it exists
path_dynamics_single_mode_output_log_bin.unlink(missing_ok=True)

time_dynamics_nolog, eps_dynamics_nolog, rho_dynamics_nolog, _, returncode_nolog = run_simulation(simr_without_log)

# remove rho_avg_out.bin if it exists
(CUDA_OUTPUT / "rho_avg_out.bin").unlink(missing_ok=True)
# remove rho_dynamics_single_mode_out.bin if it exists
(CUDA_OUTPUT / "rho_dynamics_single_mode_out.bin").unlink(missing_ok=True)
# remove rho_dynamics_single_mode_log_out.bin if it exists
path_dynamics_single_mode_output_log_bin.unlink(missing_ok=True)

_,   _,   _,   _, returncode_log   = run_simulation(simr_with_log)
loaded_data = read_log_binary(path_dynamics_single_mode_output_log_bin)

# remove rho_avg_out.bin if it exists
(CUDA_OUTPUT / "rho_avg_out.bin").unlink(missing_ok=True)
# remove rho_dynamics_single_mode_out.bin if it exists
(CUDA_OUTPUT / "rho_dynamics_single_mode_out.bin").unlink(missing_ok=True)
# remove rho_dynamics_single_mode_log_out.bin if it exists
path_dynamics_single_mode_output_log_bin.unlink(missing_ok=True)

assert returncode_nolog==0 or returncode_log==0, f"returncode_nolog={returncode_nolog}, returncode_log={returncode_log}"


###############################




'''
attibutes = loaded_data['attributes']

one_div_m   = mp(attibutes['one_div_m'])
B           = mp(attibutes['B'])
pi_alpha    = mp(attibutes['pi_alpha'])
pi_alpha_delta_C = mp(attibutes['pi_alpha_delta_C'])
delta_C     = mp(attibutes['delta_C'])
Gamma_L0    = mp(attibutes['Gamma_L0'])
Gamma_R0    = mp(attibutes['Gamma_R0'])
Gamma_eg0   = mp(attibutes['Gamma_eg0'])
Gamma_eg0_norm = mp(attibutes['Gamma_eg0_norm'])
beta        = mp(attibutes['beta'])
Gamma_phi0  = mp(attibutes['Gamma_phi0'])
epsilon_L   = mp(attibutes['epsilon_L'])
epsilon_R   = mp(attibutes['epsilon_R'])
'''


# Access the 'logs' dataset
df_np = loaded_data['data']



print(df_np.dtype)





equal_eps_dynamics = np.array_equal(df_np[df_np["substep_num"] == 0]['eps_t_substep'], eps_dynamics_nolog)


rho_dynamics_subset = df_np[df_np["substep_num"] == 3][
    ["rho_in_0", "rho_in_1", "rho_in_2", "rho_in_3"]
]

rho_dynamics_subset_2d = np.column_stack([
    rho_dynamics_subset['rho_in_0'],
    rho_dynamics_subset['rho_in_1'],
    rho_dynamics_subset['rho_in_2'],
    rho_dynamics_subset['rho_in_3'],
])


are_close_rho_dynamics = np.allclose(rho_dynamics_subset_2d, rho_dynamics_nolog, atol=1e-5)


abs_diff = np.abs(rho_dynamics_subset_2d - rho_dynamics_nolog)
max_diff = np.max(abs_diff)
print("\nMaximum absolute difference:", max_diff)
indices = np.argwhere(abs_diff == max_diff)
for idx in indices:
    i, j = idx
    print(f"Index {i},{j} -> dataset: {rho_dynamics_nolog[i,j]}, check dataset: {rho_dynamics_subset_2d[i,j]}")



# Small epsilon to avoid division by zero
eps = 1e-12

# Compute elementwise relative difference
rel_diff = np.abs(rho_dynamics_subset_2d - rho_dynamics_nolog) / (np.abs(rho_dynamics_nolog) + eps)

# Find the maximum relative difference
max_rel_diff = np.max(rel_diff)
print("\nMaximum relative difference:", max_rel_diff)

# Find indices of the maximum
indices = np.argwhere(rel_diff == max_rel_diff)

# Print the corresponding elements from both arrays
for idx in indices:
    i, j = idx
    print(f"Index {i},{j} -> dataset: {rho_dynamics_nolog[i,j]}, check dataset: {rho_dynamics_subset_2d[i,j]}")



# Convert structured array to pandas DataFrame
df_pd = pd.DataFrame(df_np)




#print(df_pd.dtypes.to_string())

'''
selected_columns = [
    #'t_idx_step',
    #'substep_num',
    't_idx_substep',
    #'t_step',
    't_substep',
    'eps_t_substep',
    #'interval_dissipator',
    #'gp_sqr',
    #'gm_sqr',
    #'gp_gm',
    #'Gamma_lprm',
    #'Gamma_lmrp',
    #'Gamma_eg',
    'debug_eps_t_substep',
    'debug_delta_C',
    'debug_radical',
    'debig_radical_div_delta_C',
    'debug_Gamma_eg0_norm',
    'debug_beta',
    'debug_Gamma_eg_loc'
]

df_pd_selected_cols = df_pd[selected_columns]


df_pd_selected_cols_chatgpt = df_pd_selected_cols.drop(['t_idx_substep', 't_substep'], axis=1)
'''







##############################################################
##############################################################
##############################################################


'''
np.set_printoptions(precision=1,    # number of digits
                    suppress=False, # allow scientific notation
                    formatter={'complexfloat_kind': lambda x: f"{x:.3e}"})
'''


def mpmath_to_numpy(mat):
    nrows, ncols = mat.rows, mat.cols
    arr = np.zeros((nrows, ncols), dtype=np.complex128)
    for i in range(nrows):
        for j in range(ncols):
            x = mat[i,j]
            # Convert string or mpc to complex
            if isinstance(x, str):
                arr[i,j] = complex(float(x))
            else:  # mpc
                arr[i,j] = complex(float(x.real), float(x.imag))
    return arr

def print_split_complex_matrix(A, eps=1e-12, precision=1, space=2):
    """
    Pretty-print a complex matrix with:
      - Zeros printed in the imaginary column
      - Real parts in left column, imaginary parts in right column
      - Both real and imaginary parts in scientific notation
      - Columns aligned
      - Commas aligned
      - Closing brackets aligned
      - Space between real and imaginary parts
      - precision: number of digits after decimal in scientific notation
    """
    fmt_str = f".{precision}e"  # force scientific notation

    A = np.array(A, dtype=np.complex128)
    nrows, ncols = A.shape

    real_strs = np.empty((nrows, ncols), dtype=object)
    imag_strs = np.empty((nrows, ncols), dtype=object)

    for i in range(nrows):
        for j in range(ncols):
            r, im = A[i,j].real, A[i,j].imag
            if abs(r) < eps and abs(im) < eps:
                # Pure zero: leave real empty, put 0 in imag
                real_strs[i,j] = ""
                imag_strs[i,j] = "0"
            else:
                real_strs[i,j] = f"{r:{fmt_str}}" if abs(r) >= eps else ""
                imag_strs[i,j] = f"{im:+{fmt_str}}j" if abs(im) >= eps else ""

    # Compute max widths
    real_widths = [max(len(real_strs[i,j]) for i in range(nrows)) for j in range(ncols)]
    imag_widths = [max(len(imag_strs[i,j]) for i in range(nrows)) for j in range(ncols)]

    # Build row strings
    row_strings = []
    for i in range(nrows):
        items = []
        for j in range(ncols):
            r = real_strs[i,j].rjust(real_widths[j])
            im = imag_strs[i,j].rjust(imag_widths[j])
            if r and im:
                items.append(r + " " * space + im)
            else:
                # if only imag exists (pure zero), print imag only
                items.append(im)
        row_strings.append(", ".join(items))

    max_len = max(len(rs) for rs in row_strings)

    # Print
    print("[")
    for rs in row_strings:
        print("  [" + rs.ljust(max_len) + "],")
    print("]")




def cuda_matrix_from_row_mpmath(row, base_name, encoding):
    """
    Constructs a 4x4 complex mpmath matrix from dataframe row columns named like '<base_name>_<i>'.

    Expected column order:
        0–3:   real parts of diagonal elements (<base_name>_0 ... _3)
        4–15:  real/imag parts of upper triangle elements in order:
                (0,1) r,i; (0,2) r,i; (0,3) r,i; (1,2) r,i; (1,3) r,i; (2,3) r,i

    Parameters:
    - row: dict-like object (e.g. pandas.Series)
    - base_name: str, e.g. 'rho' or 'U'

    Returns:
    - M: 4x4 mpmath.matrix (complex Hermitian if input is)
    """
    
    if encoding == 'hermitian_0_15':
    
        # Extract diagonal (real only)
        r00 = mp_safe(row[f'{base_name}_0'])
        r11 = mp_safe(row[f'{base_name}_1'])
        r22 = mp_safe(row[f'{base_name}_2'])
        r33 = mp_safe(row[f'{base_name}_3'])
    
        # Extract upper-triangle real/imag parts
        r01, i01 = mp_safe(row[f'{base_name}_4']), mp_safe(row[f'{base_name}_5'])
        r02, i02 = mp_safe(row[f'{base_name}_6']), mp_safe(row[f'{base_name}_7'])
        r03, i03 = mp_safe(row[f'{base_name}_8']), mp_safe(row[f'{base_name}_9'])
        r12, i12 = mp_safe(row[f'{base_name}_10']), mp_safe(row[f'{base_name}_11'])
        r13, i13 = mp_safe(row[f'{base_name}_12']), mp_safe(row[f'{base_name}_13'])
        r23, i23 = mp_safe(row[f'{base_name}_14']), mp_safe(row[f'{base_name}_15'])
    
        # Create 4x4 matrix
        M = mpmath.matrix(4, 4)
    
        # Diagonal
        M[0, 0] = mpmath.mpc(r00, 0)
        M[1, 1] = mpmath.mpc(r11, 0)
        M[2, 2] = mpmath.mpc(r22, 0)
        M[3, 3] = mpmath.mpc(r33, 0)
    
        # Upper triangle
        M[0, 1] = mpmath.mpc(r01, i01)
        M[0, 2] = mpmath.mpc(r02, i02)
        M[0, 3] = mpmath.mpc(r03, i03)
        M[1, 2] = mpmath.mpc(r12, i12)
        M[1, 3] = mpmath.mpc(r13, i13)
        M[2, 3] = mpmath.mpc(r23, i23)
    
        # Hermitian conjugate (lower triangle)
        M[1, 0] = mpmath.conj(M[0, 1])
        M[2, 0] = mpmath.conj(M[0, 2])
        M[3, 0] = mpmath.conj(M[0, 3])
        M[2, 1] = mpmath.conj(M[1, 2])
        M[3, 1] = mpmath.conj(M[1, 3])
        M[3, 2] = mpmath.conj(M[2, 3])
    
        return M
    
    elif encoding == 'hermitian_r00_i23':
    
        # Extract diagonal (real only)
        r00 = mp_safe(row[f'{base_name}_r00'])
        r11 = mp_safe(row[f'{base_name}_r11'])
        r22 = mp_safe(row[f'{base_name}_r22'])
        r33 = mp_safe(row[f'{base_name}_r33'])
    
        # Extract upper-triangle real/imag parts
        r01, i01 = mp_safe(row[f'{base_name}_r01']), mp_safe(row[f'{base_name}_i01'])
        r02, i02 = mp_safe(row[f'{base_name}_r02']), mp_safe(row[f'{base_name}_i02'])
        r03, i03 = mp_safe(row[f'{base_name}_r03']), mp_safe(row[f'{base_name}_i03'])
        r12, i12 = mp_safe(row[f'{base_name}_r12']), mp_safe(row[f'{base_name}_i12'])
        r13, i13 = mp_safe(row[f'{base_name}_r13']), mp_safe(row[f'{base_name}_i13'])
        r23, i23 = mp_safe(row[f'{base_name}_r23']), mp_safe(row[f'{base_name}_i23'])
    
        # Create 4x4 matrix
        M = mpmath.matrix(4, 4)
    
        # Diagonal
        M[0, 0] = mpmath.mpc(r00, 0)
        M[1, 1] = mpmath.mpc(r11, 0)
        M[2, 2] = mpmath.mpc(r22, 0)
        M[3, 3] = mpmath.mpc(r33, 0)
    
        # Upper triangle
        M[0, 1] = mpmath.mpc(r01, i01)
        M[0, 2] = mpmath.mpc(r02, i02)
        M[0, 3] = mpmath.mpc(r03, i03)
        M[1, 2] = mpmath.mpc(r12, i12)
        M[1, 3] = mpmath.mpc(r13, i13)
        M[2, 3] = mpmath.mpc(r23, i23)
    
        # Hermitian conjugate (lower triangle)
        M[1, 0] = mpmath.conj(M[0, 1])
        M[2, 0] = mpmath.conj(M[0, 2])
        M[3, 0] = mpmath.conj(M[0, 3])
        M[2, 1] = mpmath.conj(M[1, 2])
        M[3, 1] = mpmath.conj(M[1, 3])
        M[3, 2] = mpmath.conj(M[2, 3])
    
        return M
    
    elif encoding == 'regular_real_matrix':
        # 16 values correspond to a 4x4 real matrix in row-major order
        M = mpmath.matrix(4, 4)
        for i in range(4):
            for j in range(4):
                idx = 4 * i + j
                M[i, j] = mp_safe(row[f'{base_name}_{idx}'])
        return M
    
    else:
        return




def get_H_mpmath(epsilon,
                 params=params):
    """
    Returns the 4x4 Hamiltonian H_diab as an mpmath.matrix using arbitrary precision.

    Parameters:
    - epsilon: a numeric value (will be converted to mpmath.mpf)
    - B, pi_alpha, pi_alpha_delta_C, one_div_m: parameters as mpmath.mpf or convertible to it

    Returns:
    - H: 4x4 mpmath.matrix with arbitrary-precision floats
    """

    # Ensure epsilon is arbitrary-precision float
    #epsilon = mpmath.mpf(epsilon)

    # Initialize 4x4 matrix with zeros
    H = mpmath.matrix(4, 4)

    # Fill in diagonal and off-diagonal elements
    H[0, 0] = -half * (-params.B * epsilon - params.one_div_m)
    H[1, 1] = -half * epsilon
    H[1, 2] = -half * params.delta_C_prime
    H[2, 1] = -half * params.delta_C_prime
    H[2, 2] = +half * epsilon
    H[3, 3] = -half * (params.B * epsilon - params.one_div_m)

    return H



def compute_commutator_math_mpmath(H, rho):
    """
    Compute the commutator [H, rho] = H * rho - rho * H using mpmath matrices.

    Parameters:
    - rho: 4x4 mpmath.matrix (complex)
    - H: 4x4 mpmath.matrix (complex)

    Returns:
    - commutator: 4x4 mpmath.matrix (complex)
    """
    return H * rho - rho * H








def get_U_by_interval_check(interval_num, gamma_plus, gamma_minus):
    """
    Returns the 4x4 mpmath.matrix U for the given interval number,
    with gamma_plus and gamma_minus as mpmath.mpf (or compatible).

    Parameters:
    - interval_num: int from 1 to 6
    - gamma_plus: mpmath.mpf
    - gamma_minus: mpmath.mpf

    Returns:
    - U: 4x4 mpmath.matrix with mpmath.mpf entries
    """
    #zero = mpmath.mpf('0')
    #one = mpmath.mpf('1')

    if interval_num == 1:
        U = mpmath.matrix([
            [one,       zero,        zero,           zero],
            [zero,      gamma_plus,  -gamma_minus,   zero],
            [zero,      gamma_minus, gamma_plus,     zero],
            [zero,      zero,        zero,           one ]
        ])
    elif interval_num == 2:
        U = mpmath.matrix([
            [zero,          one,    zero,           zero],
            [gamma_plus,    zero,   -gamma_minus,   zero],
            [gamma_minus,   zero,   gamma_plus,     zero],
            [zero,          zero,   zero,           one ]
        ])
    elif interval_num == 3:
        U = mpmath.matrix([
            [zero,          zero,           one,    zero],
            [gamma_plus,    -gamma_minus,   zero,   zero],
            [gamma_minus,   gamma_plus,     zero,   zero],
            [zero,          zero,           zero,   one ]
        ])
    elif interval_num == 4:
        U = mpmath.matrix([
            [zero,          zero,           zero,   one],
            [gamma_plus,    -gamma_minus,   zero,   zero],
            [gamma_minus,   gamma_plus,     zero,   zero],
            [zero,          zero,           one,    zero]
        ])
    elif interval_num == 5:
        U = mpmath.matrix([
            [zero,          zero,   zero,           one ],
            [gamma_plus,    zero,   -gamma_minus,   zero],
            [gamma_minus,   zero,   gamma_plus,     zero],
            [zero,          one,    zero,           zero]
        ])
    elif interval_num == 6:
        U = mpmath.matrix([
            [zero,  zero,           zero,           one],
            [zero,  gamma_plus,     -gamma_minus,   zero],
            [zero,  gamma_minus,    gamma_plus,     zero],
            [one,   zero,           zero,           zero]
        ])
    else:
        raise ValueError("Invalid interval number. Please enter a number between 1 and 6.")
    
    return U




def compute_Gamma_i_to_f_check(U, i, f):
    """
    Computes Gamma_{i->f} = Gamma_L0 * W_L + Gamma_R0 * W_R with mpmath.

    Parameters:
    - U: mpmath.matrix (4 x N) with complex entries
    - i: int, initial state index
    - f: int, final state index
    - Gamma_L0, Gamma_R0: mpmath floats or complexes

    Returns:
    - Gamma: mpmath.mpc or mpf (depending on Gamma_L0/R0)
    """

    U_0f = U[0, f]  # mpmath.mpc
    U_1f = U[1, f]
    U_2f = U[2, f]
    # U_3f = U[3, f]  # unused

    # U_0i = U[0, i]  # unused
    U_1i = U[1, i]
    U_2i = U[2, i]
    U_3i = U[3, i]

    # M^{(L)}_{i->f} = U_0f * U_2i + U_1f * U_3i
    M_L = U_0f * U_2i + U_1f * U_3i

    # M^{(R)}_{i->f} = U_0f * U_1i + U_2f * U_3i
    M_R = U_0f * U_1i + U_2f * U_3i

    # W^{(L)}_{i->f} = M_L * M_L
    W_L = M_L * M_L

    # W^{(R)}_{i->f} = M_R * M_R
    W_R = M_R * M_R

    Gamma = params.Gamma_L0_prime * W_L + params.Gamma_R0_prime * W_R

    return Gamma




def compute_Gammas_by_interval_check(interval_num, U):
    """
    Compute Gamma values for the given interval using mpmath.

    Parameters:
    - interval_num: int (1 to 6)
    - U: mpmath.matrix (4 x N complex matrix)

    Returns:
    Tuple of six Gamma values (Gamma_10, Gamma_20, Gamma_30, Gamma_21, Gamma_31, Gamma_32),
    each mpmath.mpf or mpmath.mpc
    """
    #zero = mpmath.mpf('0')
    
    if interval_num == 1:
        # Interval 1: U_1
        Gamma_10 = compute_Gamma_i_to_f_check(U, 1, 0);   # Delta N = -1
        Gamma_20 = compute_Gamma_i_to_f_check(U, 2, 0);   # Delta N = -1
        Gamma_30 = zero;                                   # Delta N = -2 -> forbidden
        Gamma_21 = zero;                                   # Delta N =  0 -> forbidden
        Gamma_31 = compute_Gamma_i_to_f_check(U, 3, 1);   # Delta N = -1
        Gamma_32 = compute_Gamma_i_to_f_check(U, 3, 2);   # Delta N = -1
    elif interval_num == 2:
        # Interval 2: U_2
        Gamma_10 = compute_Gamma_i_to_f_check(U, 0, 1);   # Delta N = +1 -> reverse
        Gamma_20 = zero;                                   # Delta N = +2 -> forbidden
        Gamma_30 = compute_Gamma_i_to_f_check(U, 3, 0);   # Delta N = -1
        Gamma_21 = compute_Gamma_i_to_f_check(U, 2, 1);   # Delta N = -1
        Gamma_31 = zero;                                   # Delta N = +2 -> forbidden
        Gamma_32 = compute_Gamma_i_to_f_check(U, 3, 2);   # Delta N = -1
    elif interval_num == 3:
        # Interval 3: U_3
        Gamma_10 = zero;                                   # Delta N = 0
        Gamma_20 = compute_Gamma_i_to_f_check(U, 0, 2);   # Delta N = +1 -> reverse
        Gamma_30 = compute_Gamma_i_to_f_check(U, 3, 0);   # Delta N = -1
        Gamma_21 = compute_Gamma_i_to_f_check(U, 1, 2);   # Delta N = +1 -> reverse
        Gamma_31 = compute_Gamma_i_to_f_check(U, 3, 1);   # Delta N = -1
        Gamma_32 = zero;                                   # Delta N = 0
    elif interval_num == 4:
        # Interval 4: U_4
        Gamma_10 = zero;                                   # Delta N = 0
        Gamma_20 = compute_Gamma_i_to_f_check(U, 2, 0);   # Delta N = -1
        Gamma_30 = compute_Gamma_i_to_f_check(U, 0, 3);   # Delta N = +1 -> reverse
        Gamma_21 = compute_Gamma_i_to_f_check(U, 2, 1);   # Delta N = -1
        Gamma_31 = compute_Gamma_i_to_f_check(U, 1, 3);   # Delta N = +1 -> reverse
        Gamma_32 = zero;                                   # Delta N = 0
    elif interval_num == 5:
        # Interval 5: U_5
        Gamma_10 = compute_Gamma_i_to_f_check(U, 1, 0);   # Delta N = -1
        Gamma_20 = zero;                                   # Delta N = +2
        Gamma_30 = compute_Gamma_i_to_f_check(U, 0, 3);   # Delta N = +1 -> reverse
        Gamma_21 = compute_Gamma_i_to_f_check(U, 1, 2);   # Delta N = +1 -> reverse
        Gamma_31 = zero;                                   # Delta N = +2
        Gamma_32 = compute_Gamma_i_to_f_check(U, 2, 3);   # Delta N = +1 -> reverse
    elif interval_num == 6:
        # Interval 6: U_6
        Gamma_10 = compute_Gamma_i_to_f_check(U, 0, 1);   # Delta N = +1 -> reverse
        Gamma_20 = compute_Gamma_i_to_f_check(U, 0, 2);   # Delta N = +1 -> reverse
        Gamma_30 = zero;                                   # Delta N = -2
        Gamma_21 = zero;                                   # Delta N = 0
        Gamma_31 = compute_Gamma_i_to_f_check(U, 1, 3);   # Delta N = +1 -> reverse
        Gamma_32 = compute_Gamma_i_to_f_check(U, 2, 3);   # Delta N = +1 -> reverse
    else:
        raise ValueError("Invalid interval number. Please enter a number between 1 and 6.")
    
    return Gamma_10, Gamma_20, Gamma_30, Gamma_21, Gamma_31, Gamma_32



def L_adb_en_check(i, j, Gamma):
    """
    Constructs a 4x4 Lindblad operator L with a single non-zero element at (i, j),
    whose value is sqrt(Gamma), using mpmath for arbitrary precision.

    Parameters:
    - i, j: integer indices
    - Gamma: mpmath.mpf (real positive number)

    Returns:
    - L: 4x4 mpmath matrix with complex entries
    """
    L = mpmath.zeros(4, 4)
    L[i, j] = mpmath.sqrt(Gamma)
    return L




def compute_dissipator_mpmath(rho, L_list):
    """
    Compute the dissipator D[rho] = sum_i (L_i * rho * L_i† - 0.5 * {L_i† * L_i, rho})
    using mpmath for arbitrary-precision arithmetic.

    Parameters:
    - rho: 4x4 mpmath matrix (density matrix)
    - L_array: list of 4x4 mpmath matrices (Lindblad operators)

    Returns:
    - dissipator: 4x4 mpmath matrix
    """
    dissipator = mpmath.matrix(4, 4)
    #half = mpmath.mpf('0.5')

    for L in L_list:
        L_dag = L.transpose().conjugate()
        dissipator += L*rho*L_dag - half*(L_dag*L*rho + rho*L_dag*L)

    return dissipator



def is_hermitian_mpmath(matrix, tol=1e-28):
    """
    Checks whether a given mpmath matrix is Hermitian within a numerical tolerance.

    Parameters:
    - matrix: mpmath.matrix (must be square)
    - tol: float, the numerical tolerance (default: 1e-30)

    Returns:
    - True if the matrix is Hermitian within the given tolerance, False otherwise.
    """

    for i in range(4):
        for j in range(4):
            diff = matrix[i, j] - matrix[j, i].conjugate()
            if abs(diff) > tol:
                return False
    return True




def relative_difference_single_mpmath(val1, val2, d=mpmath.mpf('1e-30')):
    """
    Calculate the relative difference between two scalar mpmath values.
    
    Parameters:
    - val1: First value (mpmath.mpf or mpc)
    - val2: Second value (mpmath.mpf or mpc)
    - epsilon: Small positive value to avoid division by zero (default 1e-30)
    
    Returns:
    - relative_diff: The relative difference as mpmath.mpf
    """

    numerator = mpmath.fabs(val1 - val2)
    denom_candidate = mpmath.fabs(val1) + mpmath.fabs(val2)
    denominator = denom_candidate if denom_candidate > d else d
    return numerator / denominator



def relative_matrix_difference(A, B):
    """
    Compute the elementwise relative difference between two 4x4 mpmath matrices.

    The difference is defined as:
        D = (A - B) / scale
    where scale = min(max(|A|), max(|B|))

    Parameters
    ----------
    A, B : mpmath.matrix
        Two 4x4 matrices to compare (real or complex).

    Returns
    -------
    D : mpmath.matrix
        The elementwise relative difference matrix.
    scale : mpmath.mpf
        The scale factor used for normalization.
    max_diff : mpmath.mpf
        The maximum absolute element of D (overall relative difference).
    """

    if A.rows != 4 or A.cols != 4 or B.rows != 4 or B.cols != 4:
        raise ValueError("Both matrices must be 4x4.")

    # Compute max absolute element in each
    maxA = mpmath.mpf(0)
    maxB = mpmath.mpf(0)

    for i in range(4):
        for j in range(4):
            maxA = max(maxA, abs(A[i, j]))
            maxB = max(maxB, abs(B[i, j]))

    # Use smaller of the two maxima
    scale = max(maxA, maxB)

    if scale == 0:
        # If both are zero matrices
        return mpmath.zeros(4, 4), mpmath.mpf(0), mpmath.mpf(0)

    # Compute relative difference matrix
    diff_abs = (A - B)
    diff_rel = (A - B) / scale

    # Find largest element in D
    max_abs_diff = mpmath.mpf(0)
    max_rel_diff = mpmath.mpf(0)
    for i in range(4):
        for j in range(4):
            max_abs_diff = max(max_abs_diff, abs(diff_abs[i, j]))
            max_rel_diff = max(max_rel_diff, abs(diff_rel[i, j]))

    return diff_abs, diff_rel, max_abs_diff, max_rel_diff, scale







#########################################################

































































########################################################################################


def process_row(row, t_idx_row_check, params, minus_i, one, half, zero):

    metrics = {
        'rel_diff_t_step': None,
        'rel_diff_t_substep': None,
        'rel_diff_eps_t_substep': None,

        'rel_diff_gp_sqr': None,
        'rel_diff_gm_sqr': None,
        'rel_diff_gp_gm': None,
        'rel_diff_Gamma_lprm': None,
        'rel_diff_Gamma_lmrp': None,
        'rel_diff_Gamma_phi': None,

        'max_rel_diff_commutator': None,
        'max_rel_diff_dissipator_dl_db': None,
        'max_rel_diff_dissipator_phi_db': None,
        'max_rel_diff_dissipator_eg_db': None,
        'max_rel_diff_rhs': None,
        
        'max_abs_UTU_1': None,
        'max_abs_UUT_1': None,
        'abs_det_U_1': None,
        'H_adb_non_diag_sum': None,
        'U_analyt_numeric_max_diff': None,
        'H_adb_analyt_numeric_check_max_diff': None,
        'all_checks_passed': True
    }


    # =========================================================================
    # Index and time validation
    # =========================================================================
    t_idx_substep_check = t_idx_row_check
    is_t_idx_substep_correct = row['t_idx_substep'] == t_idx_substep_check
    if not is_t_idx_substep_correct:
        metrics['all_checks_passed'] = False

    substep_num_check = t_idx_substep_check % 4
    is_substep_num_correct = row['substep_num'] == substep_num_check
    if not is_substep_num_correct:
        metrics['all_checks_passed'] = False
    
    t_idx_step_check = t_idx_substep_check // 4
    is_t_idx_step_correct = row['t_idx_step'] == t_idx_step_check
    if not is_t_idx_step_correct:
        metrics['all_checks_passed'] = False

    # Validate t_step
    t_step_check = t_idx_step_check * params.dt_prime
    rel_diff_t_step = relative_difference_single_mpmath(row['t_step'], t_step_check)
    metrics['rel_diff_t_step'] = rel_diff_t_step
    
    # Validate t_substep based on substep_num
    if substep_num_check == 0:
        t_substep_check = t_step_check
    elif substep_num_check == 1:
        t_substep_check = t_step_check + half * params.dt_prime
    elif substep_num_check == 2:
        t_substep_check = t_step_check + half * params.dt_prime
    elif substep_num_check == 3:
        t_substep_check = t_step_check + params.dt_prime
    
    rel_diff_t_substep = relative_difference_single_mpmath(row['t_substep'], t_substep_check)
    metrics['rel_diff_t_substep'] = rel_diff_t_substep

    
    # =========================================================================
    # Epsilon validation
    # =========================================================================

    eps_t_substep_check = params.epsilon0_prime + params.A_prime * mpmath.cos(params.omega_prime * t_substep_check)
    
    rel_diff_eps_t_substep = relative_difference_single_mpmath(row['eps_t_substep'], eps_t_substep_check)
    metrics['rel_diff_eps_t_substep'] = rel_diff_eps_t_substep

    
    # =========================================================================
    # Gamma coefficients validation
    # =========================================================================
    denominator_check = mpmath.sqrt(params.delta_C_prime**2 + eps_t_substep_check**2)
    
    gamma_plus_check = mpmath.sqrt(half * (one + eps_t_substep_check / denominator_check))
    gamma_minus_check = mpmath.sqrt(half * (one - eps_t_substep_check / denominator_check))
    
    gp_sqr_check = gamma_plus_check**2
    gm_sqr_check = gamma_minus_check**2
    gp_gm_check = gamma_plus_check * gamma_minus_check
    
    rel_diff_gp_sqr = relative_difference_single_mpmath(mp(row['gp_sqr']), gp_sqr_check)
    rel_diff_gm_sqr = relative_difference_single_mpmath(mp(row['gm_sqr']), gm_sqr_check)
    rel_diff_gp_gm = relative_difference_single_mpmath(mp(row['gp_gm']), gp_gm_check)
    
    metrics['rel_diff_gp_sqr'] = rel_diff_gp_sqr
    metrics['rel_diff_gm_sqr'] = rel_diff_gm_sqr
    metrics['rel_diff_gp_gm'] = rel_diff_gp_gm
    
    # Gamma_lprm and Gamma_lmrp validation
    Gamma_lprm_check = params.Gamma_L0_prime * gp_sqr_check + params.Gamma_R0_prime * gm_sqr_check
    Gamma_lmrp_check = params.Gamma_L0_prime * gm_sqr_check + params.Gamma_R0_prime * gp_sqr_check
    
    rel_diff_Gamma_lprm = relative_difference_single_mpmath(row['Gamma_lprm'], Gamma_lprm_check)
    rel_diff_Gamma_lmrp = relative_difference_single_mpmath(row['Gamma_lmrp'], Gamma_lmrp_check)
    
    metrics['rel_diff_Gamma_lprm'] = rel_diff_Gamma_lprm
    metrics['rel_diff_Gamma_lmrp'] = rel_diff_Gamma_lmrp
    
    
    # =========================================================================
    # Density matrix extraction and validation
    # =========================================================================
    rho = cuda_matrix_from_row_mpmath(row, 'rho_in', encoding='hermitian_0_15')
    
    is_hermitian_rho = is_hermitian_mpmath(rho)
    if not is_hermitian_rho:
        metrics['all_checks_passed'] = False
    
    # =========================================================================
    # Commutator validation
    # =========================================================================
    H_check = get_H_mpmath(eps_t_substep_check)
    
    commutator = cuda_matrix_from_row_mpmath(row, 'drho_out_comm', encoding='hermitian_0_15')
    commutator_check = minus_i * compute_commutator_math_mpmath(H_check, rho)
    
    is_hermitian_commutator = is_hermitian_mpmath(commutator)
    is_hermitian_commutator_check = is_hermitian_mpmath(commutator_check)
    if not (is_hermitian_commutator and is_hermitian_commutator_check):
        metrics['all_checks_passed'] = False
    
    _, _, _, max_rel_diff_commutator, _ = relative_matrix_difference(commutator, commutator_check)
    metrics['max_rel_diff_commutator'] = max_rel_diff_commutator
    
    # =========================================================================
    # Interval determination for dissipator
    # =========================================================================
    if eps_t_substep_check < -params.epsilon_L:
        interval_check = 1
    elif eps_t_substep_check < -params.epsilon_R:
        interval_check = 2
    elif eps_t_substep_check < zero:
        interval_check = 3
    elif eps_t_substep_check < params.epsilon_R:
        interval_check = 4
    elif eps_t_substep_check < params.epsilon_L:
        interval_check = 5
    elif eps_t_substep_check > params.epsilon_L:
        interval_check = 6
    else:
        KeyError("eps_t_substep_check is equal to epsilon_L or epsilon_R or 0. Check sepately")
        metrics['all_checks_passed'] = False
        return metrics
    
    is_interval_dissipator_correct = row['interval_dissipator'] == interval_check
    if not is_interval_dissipator_correct:
        metrics['all_checks_passed'] = False
    
    # =========================================================================
    # U matrix validation
    # =========================================================================
    U_analyt_check = get_U_by_interval_check(
        interval_num=interval_check,
        gamma_plus=gamma_plus_check,
        gamma_minus=gamma_minus_check
    )
    
    # Check orthogonality
    UTU = U_analyt_check.transpose() * U_analyt_check
    UUT = U_analyt_check * U_analyt_check.transpose()
    UTU_1 = UTU - mpmath.eye(4)
    UUT_1 = UUT - mpmath.eye(4)
    max_abs_UTU_1 = max(np.abs(UTU_1))
    max_abs_UUT_1 = max(np.abs(UUT_1))
    
    metrics['max_abs_UTU_1'] = max_abs_UTU_1
    metrics['max_abs_UUT_1'] = max_abs_UUT_1
    
    # Check determinant
    det_U = mpmath.det(U_analyt_check)
    abs_det_U_1 = abs(abs(det_U) - 1)
    metrics['abs_det_U_1'] = abs_det_U_1
    
    # Validate diagonalization
    H_adb_analyt_check = U_analyt_check.transpose() * H_check * U_analyt_check
    
    H_adb_non_diag_sum = sum(
        abs(H_adb_analyt_check[i, j])
        for i in range(H_adb_analyt_check.rows)
        for j in range(H_adb_analyt_check.cols)
        if i != j
    )
    metrics['H_adb_non_diag_sum'] = H_adb_non_diag_sum
    
    # Check energy ordering
    is_energies_ascending = (
        H_adb_analyt_check[0, 0] <= H_adb_analyt_check[1, 1] <=
        H_adb_analyt_check[2, 2] <= H_adb_analyt_check[3, 3]
    )
    if not is_energies_ascending:
        metrics['all_checks_passed'] = False
    
    # Compare analytic vs numeric U
    E_numeric_check, U_numeric_check = mpmath.eigsy(H_check)
    U_analyt_numeric_diff_check = U_analyt_check - U_numeric_check
    U_analyt_numeric_max_diff = max(np.abs(U_analyt_numeric_diff_check))
    metrics['U_analyt_numeric_max_diff'] = U_analyt_numeric_max_diff

    # Compare analytic vs numeric H diagonalization
    H_adb_numeric_check = U_numeric_check.transpose() * H_check * U_numeric_check

    H_adb_analyt_numeric_diff_check = H_adb_analyt_check - H_adb_numeric_check
    H_adb_analyt_numeric_check_max_diff = np.max(np.abs(H_adb_analyt_numeric_diff_check))
    metrics['H_adb_analyt_numeric_check_max_diff'] = H_adb_analyt_numeric_check_max_diff
    
    # =========================================================================
    # Dissipator dot-lead validation
    # =========================================================================
    Gamma_10, Gamma_20, Gamma_30, Gamma_21, Gamma_31, Gamma_32 = compute_Gammas_by_interval_check(
        interval_num=interval_check,
        U=U_analyt_check
    )
    
    # Lindblad operators in adiabatic basis
    L_adb_01 = L_adb_en_check(0, 1, Gamma_10)
    L_adb_02 = L_adb_en_check(0, 2, Gamma_20)
    L_adb_03 = L_adb_en_check(0, 3, Gamma_30)
    L_adb_12 = L_adb_en_check(1, 2, Gamma_21)
    L_adb_13 = L_adb_en_check(1, 3, Gamma_31)
    L_adb_23 = L_adb_en_check(2, 3, Gamma_32)
    
    # Transform to diabatic basis
    L_db_01 = U_analyt_check * L_adb_01 * U_analyt_check.transpose()
    L_db_02 = U_analyt_check * L_adb_02 * U_analyt_check.transpose()
    L_db_03 = U_analyt_check * L_adb_03 * U_analyt_check.transpose()
    L_db_12 = U_analyt_check * L_adb_12 * U_analyt_check.transpose()
    L_db_13 = U_analyt_check * L_adb_13 * U_analyt_check.transpose()
    L_db_23 = U_analyt_check * L_adb_23 * U_analyt_check.transpose()
    
    L_db_array = [L_db_01, L_db_02, L_db_03, L_db_12, L_db_13, L_db_23]
    
    dissipator_dl_db_check = compute_dissipator_mpmath(rho, L_db_array)
    dissipator_dl_db = cuda_matrix_from_row_mpmath(row, 'drho_out_D_dl', encoding='hermitian_r00_i23')
    
    is_hermitian_dissipator_dl_db = is_hermitian_mpmath(dissipator_dl_db)
    is_hermitian_dissipator_dl_db_check = is_hermitian_mpmath(dissipator_dl_db_check)
    if not (is_hermitian_dissipator_dl_db and is_hermitian_dissipator_dl_db_check):
        metrics['all_checks_passed'] = False
    
    _, _, _, max_rel_diff_dissipator_dl_db, _ = relative_matrix_difference(dissipator_dl_db, dissipator_dl_db_check)
    metrics['max_rel_diff_dissipator_dl_db'] = max_rel_diff_dissipator_dl_db

    
    # =========================================================================
    # Dissipator PHI (dephasing) validation
    # =========================================================================
    Gamma_phi_check = params.Gamma_phi0_prime * abs(eps_t_substep_check) / mpmath.sqrt(
        params.delta_C_prime**2 + eps_t_substep_check**2
    )
    
    rel_diff_Gamma_phi = relative_difference_single_mpmath(row['Gamma_phi'], Gamma_phi_check)
    metrics['rel_diff_Gamma_phi'] = rel_diff_Gamma_phi
    
    L_phi_adb = mpmath.zeros(4, 4)
    L_phi_adb[1, 1] = -mpmath.sqrt(Gamma_phi_check / mpmath.mpf('2'))
    L_phi_adb[2, 2] = mpmath.sqrt(Gamma_phi_check / mpmath.mpf('2'))
    
    U_analyt_natural_check = get_U_by_interval_check(1, gamma_plus_check, gamma_minus_check)
    
    L_phi_db = U_analyt_natural_check * L_phi_adb * U_analyt_natural_check.transpose()
    
    dissipator_phi_db_check = compute_dissipator_mpmath(rho, [L_phi_db])
    dissipator_phi_db = cuda_matrix_from_row_mpmath(row, 'drho_out_D_phi', encoding='hermitian_r00_i23')
    
    is_hermitian_dissipator_phi_db = is_hermitian_mpmath(dissipator_phi_db)
    is_hermitian_dissipator_phi_db_check = is_hermitian_mpmath(dissipator_phi_db_check)
    if not (is_hermitian_dissipator_phi_db and is_hermitian_dissipator_phi_db_check):
        metrics['all_checks_passed'] = False
    
    _, _, _, max_rel_diff_dissipator_phi_db, _ = relative_matrix_difference(dissipator_phi_db, dissipator_phi_db_check)
    metrics['max_rel_diff_dissipator_phi_db'] = max_rel_diff_dissipator_phi_db
    
    # =========================================================================
    # Dissipator EG validation
    # =========================================================================
    
    #TBD to be derived
    Gamma_eg = row['Gamma_eg']
    
    L_eg_adb = mpmath.zeros(4, 4)
    L_eg_adb[1, 2] = mpmath.sqrt(Gamma_eg)
    
    L_eg_db = U_analyt_natural_check * L_eg_adb * U_analyt_natural_check.transpose()
    
    dissipator_eg_db_check = compute_dissipator_mpmath(rho, [L_eg_db])
    dissipator_eg_db = cuda_matrix_from_row_mpmath(row, 'drho_out_D_eg', encoding='hermitian_r00_i23')
    
    is_hermitian_dissipator_eg_db = is_hermitian_mpmath(dissipator_eg_db)
    is_hermitian_dissipator_eg_db_check = is_hermitian_mpmath(dissipator_eg_db_check)
    if not (is_hermitian_dissipator_eg_db and is_hermitian_dissipator_eg_db_check):
        metrics['all_checks_passed'] = False
    
    _, _, _, max_rel_diff_dissipator_eg_db, _ = relative_matrix_difference(dissipator_eg_db, dissipator_eg_db_check)
    metrics['max_rel_diff_dissipator_eg_db'] = max_rel_diff_dissipator_eg_db
    
    # =========================================================================
    # Total RHS validation
    # =========================================================================
    rhs = cuda_matrix_from_row_mpmath(row, 'drho_out_total', encoding='hermitian_0_15')
    rhs_check = commutator_check + dissipator_dl_db_check + dissipator_eg_db_check + dissipator_phi_db_check
    
    is_hermitian_rhs = is_hermitian_mpmath(rhs)
    is_hermitian_rhs_check = is_hermitian_mpmath(rhs_check)
    if not (is_hermitian_rhs and is_hermitian_rhs_check):
        metrics['all_checks_passed'] = False
    
    _, _, _, max_rel_diff_rhs, _ = relative_matrix_difference(rhs, rhs_check)
    metrics['max_rel_diff_rhs'] = max_rel_diff_rhs
    
    return metrics


# rhs_np = mpmath_to_numpy(rhs)
# print_split_complex_matrix(rhs_np, precision=4)

# rhs_check_np = mpmath_to_numpy(rhs_check)
# print_split_complex_matrix(rhs_check_np, precision=4)

########################################################################################




def process_and_add_calculated_columns_to_np_mpf(df_np, params, minus_i, one, half, zero):
    
    print(f"[PID {os.getpid()}] Starting chunk", flush=True)
    
    # Get the dtype of the first row to derive the types
    first_row = df_np[0]  # The first row of the structured array
    
    # Process the first row to get the calculated columns
    results = process_row(first_row, t_idx_row_check=0, params=params, minus_i=minus_i, one=one, half=half, zero=zero)
    
    # Prepare a list for the new dtype: Start with the existing fields
    new_column_dtypes = [(name, df_np.dtype[name]) for name in df_np.dtype.names]
    existing_fields = set(df_np.dtype.names)  # Keep track of the existing fields to avoid conflicts

    # Add the new calculated columns to the dtype list
    for key, value in results.items():
        # For scalar types (int, float, bool), we add directly to the dtype list
        if key not in existing_fields:
            if isinstance(value, mpf):
                new_column_dtypes.append((key, mpf))
            elif value is None:
                # Handle None values - default to mpf for numerical metrics
                new_column_dtypes.append((key, mpf))
            elif isinstance(value, bool):
                new_column_dtypes.append((key, bool))
            else:
                new_column_dtypes.append((key, type(value)))
            existing_fields.add(key)

    # Create the new dtype from the new column dtypes
    new_dtype = np.dtype(new_column_dtypes)

    # Create a new structured array with the new dtype
    df_np2 = np.empty(df_np.shape, dtype=new_dtype)
    
    # Copy the data from the original array into the new structured array
    for name in df_np.dtype.names:
        df_np2[name] = df_np[name]
    
    num_rows = df_np.shape[0]  # Get the total number of rows in df_np
    
    # Now process each row and add the calculated columns
    for i, row in enumerate(df_np):
        
        # Print the index every 1000th row
        if i % 100 == 0:
            print(f"[PID {os.getpid()}] Processing row index {i}/{num_rows}...", flush=True)
            
        results = process_row(row, t_idx_row_check=i, params=params, minus_i=minus_i, one=one, half=half, zero=zero)
        
        # Assign the calculated results to the new columns
        for key, value in results.items():
            if isinstance(value, np.ndarray) or isinstance(value, list):
                for j in range(len(value)):  # For arrays, assign to individual columns
                    val = value[j]
                    df_np2[key + f'_{j}'][i] = val
            else:
                df_np2[key][i] = value
    
    return df_np2




def print_max_metrics(df_np2):
    """Calculate and print all max metric values in scientific notation."""
    
    metrics_data = {
        'rel_diff_t_step': max(df_np2['rel_diff_t_step']),
        'rel_diff_t_substep': max(df_np2['rel_diff_t_substep']),
        'rel_diff_eps_t_substep': max(df_np2['rel_diff_eps_t_substep']),
        'rel_diff_gp_sqr': max(df_np2['rel_diff_gp_sqr']),
        'rel_diff_gm_sqr': max(df_np2['rel_diff_gm_sqr']),
        'rel_diff_gp_gm': max(df_np2['rel_diff_gp_gm']),
        'rel_diff_Gamma_lprm': max(df_np2['rel_diff_Gamma_lprm']),
        'rel_diff_Gamma_lmrp': max(df_np2['rel_diff_Gamma_lmrp']),
        'rel_diff_Gamma_phi': max(df_np2['rel_diff_Gamma_phi']),
        'max_rel_diff_commutator': max(df_np2['max_rel_diff_commutator']),
        'max_rel_diff_dissipator_dl_db': max(df_np2['max_rel_diff_dissipator_dl_db']),
        'max_rel_diff_dissipator_phi_db': max(df_np2['max_rel_diff_dissipator_phi_db']),
        'max_rel_diff_dissipator_eg_db': max(df_np2['max_rel_diff_dissipator_eg_db']),
        'max_rel_diff_rhs': max(df_np2['max_rel_diff_rhs']),
        'max_abs_UTU_1': max(df_np2['max_abs_UTU_1']),
        'max_abs_UUT_1': max(df_np2['max_abs_UUT_1']),
        'abs_det_U_1': max(df_np2['abs_det_U_1']),
        'H_adb_non_diag_sum': max(df_np2['H_adb_non_diag_sum']),
        'U_analyt_numeric_max_diff': max(df_np2['U_analyt_numeric_max_diff']),
        'H_adb_analyt_numeric_check_max_diff': max(df_np2['H_adb_analyt_numeric_check_max_diff']),
    }
    
    print("=" * 70)
    print("MAXIMUM METRIC VALUES")
    print("=" * 70)
    
    for name, value in metrics_data.items():
        # Convert to float for formatting
        val_float = float(value)
        print(f"{name:<45} : {val_float:.2e}")
    
    print("=" * 70)
    
    return metrics_data





##########################################




def verify_rk4_steps(df_np, tolerance = 1e-6):
    """
    Verify RK4 integration by checking:
    rho(n+1) = rho(n) + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    Where k_i are the RHS matrices at each substep.
    
    Parameters
    ----------
    df_np : np.ndarray
        Structured numpy array with RK4 substeps
    
    Returns
    -------
    results : dict
        Contains verification results for each RK4 step
    """
    import mpmath
    
    # Get timestep from data
    t_step_0 = mp_safe(df_np[0]['t_step'])
    t_step_1 = mp_safe(df_np[4]['t_step'])  # Next step (4 substeps later)
    h = t_step_1 - t_step_0
    
    # RK4 coefficient
    h_over_6 = h / mpmath.mpf('6')
    
    n_rows = len(df_np)
    n_steps = n_rows // 4  # Each RK4 step has 4 substeps
    
    results = {
        'step_indices': [],
        'max_rel_diff': [],
        'max_abs_diff': [],
        'passed': [],
        'failed_steps': []
    }
    
    print(f"Verifying {n_steps} RK4 steps...")
    print(f"Timestep h = {h}")
    
    for step_idx in range(n_steps - 1):  # -1 because we need next step
        base_idx = step_idx * 4
        
        # Get current step data (substep 0)
        row_substep_0 = df_np[base_idx]
        
        # Verify this is substep 0
        if int(row_substep_0['substep_num']) != 0:
            print(f"Warning: Expected substep 0 at index {base_idx}, got {row_substep_0['substep_num']}")
            continue
        
        # Extract rho at current step (substep 0)
        rho_n = cuda_matrix_from_row_mpmath(row_substep_0, 'rho_in', encoding='hermitian_0_15')
        
        if rho_n is None:
            print(f"Warning: Could not extract rho_n at step {step_idx}")
            continue
        
        # Extract k1, k2, k3, k4 from the 4 substeps
        k_matrices = []
        for substep in range(4):
            row = df_np[base_idx + substep]
            
            # Verify substep number
            if int(row['substep_num']) != substep:
                print(f"Warning: Expected substep {substep} at index {base_idx + substep}")
                continue
            
            k = cuda_matrix_from_row_mpmath(row, 'drho_out_total', encoding='hermitian_0_15')
            if k is None:
                print(f"Warning: Could not extract k at substep {substep}")
                break  # Exit inner loop
            k_matrices.append(k)
        
        if len(k_matrices) != 4:
            print(f"Warning: Could not get all 4 k matrices for step {step_idx}")
            continue
        
        k1, k2, k3, k4 = k_matrices
        
        # Compute RK4 update: rho(n+1) = rho(n) + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        rk4_update = k1 + mpmath.mpf('2')*k2 + mpmath.mpf('2')*k3 + k4
        rk4_update = h_over_6 * rk4_update
        
        rho_n_plus_1_computed = rho_n + rk4_update
        
        # Get actual rho at next step (substep 0 of next RK4 step)
        next_step_idx = (step_idx + 1) * 4
        row_next_step = df_np[next_step_idx]
        
        # Verify this is substep 0 of next step
        if int(row_next_step['substep_num']) != 0:
            print(f"Warning: Expected substep 0 at index {next_step_idx}")
            continue
        
        rho_n_plus_1_actual = cuda_matrix_from_row_mpmath(row_next_step, 'rho_in', encoding='hermitian_0_15')
        
        # Check if both are Hermitian
        
        if not (is_hermitian_mpmath(rho_n_plus_1_computed) and is_hermitian_mpmath(rho_n_plus_1_actual)):
            print(f"Warning: Non-Hermitian matrix at step {step_idx}")
            results['passed'].append(False)
            results['failed_steps'].append(step_idx)
            continue
        
        # Compute relative difference
        diff_abs, diff_rel, max_abs_diff, max_rel_diff, scale = relative_matrix_difference(
            rho_n_plus_1_actual, rho_n_plus_1_computed
        )
        
        # Store results
        results['step_indices'].append(step_idx)
        results['max_rel_diff'].append(float(max_rel_diff))
        results['max_abs_diff'].append(float(max_abs_diff))
        
        # Check tolerance (adjust as needed)
        passed = max_rel_diff < tolerance
        results['passed'].append(passed)
        
        if not passed:
            results['failed_steps'].append(step_idx)
            print(f"Step {step_idx}: FAILED - max_rel_diff = {max_rel_diff:.2e}")
        
        # Progress indicator
        if (step_idx + 1) % 1000 == 0:
            print(f"Verified {step_idx + 1}/{n_steps - 1} steps")
    
    # Summary statistics
    n_verified = len(results['passed'])
    n_passed = sum(results['passed'])
    n_failed = n_verified - n_passed
    
    print(f"\n{'='*60}")
    print("RK4 Verification Summary:")
    print(f"{'='*60}")
    print(f"Total steps verified: {n_verified}")
    print(f"Passed: {n_passed} ({100*n_passed/n_verified:.2f}%)")
    print(f"Failed: {n_failed} ({100*n_failed/n_verified:.2f}%)")
    
    if results['max_rel_diff']:
        print(f"Max relative difference: {max(results['max_rel_diff']):.2e}")
        print(f"Min relative difference: {min(results['max_rel_diff']):.2e}")
        print(f"Mean relative difference: {sum(results['max_rel_diff'])/len(results['max_rel_diff']):.2e}")
    
    if results['failed_steps']:
        print(f"\nFirst 10 failed steps: {results['failed_steps'][:10]}")
    
    return results





# Optional: Create a detailed report for failed steps
def analyze_failed_steps(df_np_expanded, failed_steps, max_steps=5):
    """Analyze the first few failed steps in detail"""
    print(f"\nDetailed analysis of first {max_steps} failed steps:")
    print("="*60)
    
    for i, step_idx in enumerate(failed_steps[:max_steps]):
        base_idx = step_idx * 4
        
        print(f"\nStep {step_idx} (indices {base_idx} to {base_idx+3}):")
        
        row_0 = df_np_expanded[base_idx]
        print(f"  t_step: {row_0['t_step']}")
        print(f"  t_substep range: {row_0['t_substep']} to {df_np_expanded[base_idx+3]['t_substep']}")
        
        rho_n = cuda_matrix_from_row_mpmath(row_0, 'rho', encoding='hermitian')
        
        # Check trace of density matrix
        trace_rho = sum(rho_n[i,i] for i in range(4))
        print(f"  Trace(rho_n): {trace_rho}")
        
        # Check if RHS are Hermitian
        for substep in range(4):
            row = df_np_expanded[base_idx + substep]
            rhs = cuda_matrix_from_row_mpmath(row, 'drho_out_total_rhs', encoding='hermitian')
            is_herm = is_hermitian_mpmath(rhs)
            print(f"  RHS substep {substep} Hermitian: {is_herm}")

















font_size = 30
tick_length = 8
tick_width = 2
fig_size = (38, 20)
line_width = 4


def plot(df_np, cols, epsilon_L=params.epsilon_L, epsilon_R=params.epsilon_R):
    # Extract the time substep column (assuming it exists)
    t_substep = df_np['t_substep']
    
    
    
    # Create figure with gridspec to control subplot height ratios
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(30, 18),
        gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True
    )

    
    # Loop through each variable and plot it
    
    for col in cols:
        print(col)
        if col in df_np.dtype.names:  # Ensure the variable exists in the dataset
            ax1.plot(t_substep, df_np[col], label=col, linewidth=line_width)
    
    ax1.set_ylabel('values', fontsize=font_size)
    ax1.set_title('Value', fontsize=font_size)
    ax1.legend(fontsize=font_size)
    ax1.legend(loc='best', fontsize=font_size, bbox_to_anchor=(1.05, 1), title="Variables")
    ax1.tick_params(axis='both', labelsize=font_size)
    ax1.grid(True)
    
    
    
    # Lower subplot: epsilon with reference lines
    ax2.plot(t_substep, df_np["eps_t_substep"], color="black", linewidth=line_width, label="epst")
    ax2.axhline(-epsilon_L, color="red", linewidth=line_width, linestyle="--", label="-epsilon_L")
    ax2.axhline(-epsilon_R, color="blue", linewidth=line_width, linestyle="--", label="-epsilon_R")
    ax2.axhline(0, color="gray", linewidth=line_width, linestyle="-")
    ax2.axhline(epsilon_R, color="blue", linewidth=line_width, linestyle="--", label="epsilon_R")
    ax2.axhline(epsilon_L, color="red", linewidth=line_width, linestyle="--", label="epsilon_L")
    ax2.set_xlabel("t_substep", fontsize=font_size)
    ax2.set_ylabel("Epsilon", fontsize=font_size)
    ax2.grid(True)
    ax2.legend(fontsize=font_size)
    ax2.tick_params(axis='both', labelsize=font_size)
    
    # Show plot
    #plt.tight_layout()
    plt.show()
    





#########################################################
# main




df_np2 = process_and_add_calculated_columns_to_np_mpf(
    df_np, 
    params=params, 
    minus_i=minus_i, 
    one=one, 
    half=half, 
    zero=zero
)



all_checks_all_rows_passed = np.all(df_np2['all_checks_passed'])



if not all_checks_all_rows_passed:
    warnings.warn("Not all boolean checks were passed")
else:
    print("\n\nAll values in 'all_checks_passed' are True for all rows")



# Usage:
max_vals = print_max_metrics(df_np2)

# Run the verification
rk4_results = verify_rk4_steps(df_np)

if rk4_results['failed_steps']:
    analyze_failed_steps(df_np, rk4_results['failed_steps'])







df_pd2 = pd.DataFrame(df_np2)

df_pd2_display_5_digits = df_pd2.map(
    lambda x: mpmath.nstr(x, 5)
    if isinstance(x, mpf)
    else x
)

df_pd2_display_all_digits = df_pd2.map(
    lambda x: mpmath.nstr(x, mpmath.mp.dps)
    if isinstance(x, mpf)
    else x
)

#########################################################




# Filter where U_analyt_numeric_max_diff > 1
filtered_df_np2 = df_np2[df_np2['U_analyt_numeric_max_diff'] > 1]


filtered_df_pd2 = pd.DataFrame(filtered_df_np2)


















##############
##############
##############
# in parallel

'''
from joblib import Parallel, delayed


chunks = np.array_split(df_np, 16)
results = Parallel(n_jobs=16)(
    delayed(add_calculated_columns_to_np_mpf)(chunk) for chunk in chunks
)
df_np2_multiple_threads = np.concatenate(results, axis=0)

#df_np2_multiple_threads == df_np2_single_thread
'''


##############
##############
##############
















# List of variables you want to plot
cols = [
    't_idx_step',
    'substep_num',
    't_idx_substep',
    't_step',
    't_substep',
    'eps_t_substep',
    'interval_dissipator',
    'gp_sqr',
    'gm_sqr',
    'gp_gm',
    'Gamma_10_32',
    'Gamma_20_31',
    'rho_in_0',
    'rho_in_1',
    'rho_in_2',
    'rho_in_3',
    'rho_in_4',
    'rho_in_5',
    'rho_in_6',
    'rho_in_7',
    'rho_in_8',
    'rho_in_9',
    'rho_in_10',
    'rho_in_11',
    'rho_in_12',
    'rho_in_13',
    'rho_in_14',
    'rho_in_15',
    'drho_out_comm_0',
    'drho_out_comm_1',
    'drho_out_comm_2',
    'drho_out_comm_3',
    'drho_out_comm_4',
    'drho_out_comm_5',
    'drho_out_comm_6',
    'drho_out_comm_7',
    'drho_out_comm_8',
    'drho_out_comm_9',
    'drho_out_comm_10',
    'drho_out_comm_11',
    'drho_out_comm_12',
    'drho_out_comm_13',
    'drho_out_comm_14',
    'drho_out_comm_15',
    'drho_out_D_r00',
    'drho_out_D_r11',
    'drho_out_D_r22',
    'drho_out_D_r33',
    'drho_out_D_r01',
    'drho_out_D_i01',
    'drho_out_D_r02',
    'drho_out_D_i02',
    'drho_out_D_r03',
    'drho_out_D_i03',
    'drho_out_D_r12',
    'drho_out_D_i12',
    'drho_out_D_r13',
    'drho_out_D_i13',
    'drho_out_D_r23',
    'drho_out_D_i23',
    'drho_out_total_0',
    'drho_out_total_1',
    'drho_out_total_2',
    'drho_out_total_3',
    'drho_out_total_4',
    'drho_out_total_5',
    'drho_out_total_6',
    'drho_out_total_7',
    'drho_out_total_8',
    'drho_out_total_9',
    'drho_out_total_10',
    'drho_out_total_11',
    'drho_out_total_12',
    'drho_out_total_13',
    'drho_out_total_14',
    'drho_out_total_15',
    'is_hermitian_rho',
    'is_hermitian_comm',
    'is_hermitian_diss',
    'rel_diff_gp_sqr',
    'rel_diff_gm_sqr',
    'rel_diff_gp_gm',
    'rel_diff_comm_0',
    'rel_diff_comm_1',
    'rel_diff_comm_2',
    'rel_diff_comm_3',
    'rel_diff_comm_4',
    'rel_diff_comm_5',
    'rel_diff_comm_6',
    'rel_diff_comm_7',
    'rel_diff_comm_8',
    'rel_diff_comm_9',
    'rel_diff_comm_10',
    'rel_diff_comm_11',
    'rel_diff_comm_12',
    'rel_diff_comm_13',
    'rel_diff_comm_14',
    'rel_diff_comm_15',
    'diss_db_vec_cuda_0',
    'diss_db_vec_cuda_1',
    'diss_db_vec_cuda_2',
    'diss_db_vec_cuda_3',
    'diss_db_vec_cuda_4',
    'diss_db_vec_cuda_5',
    'diss_db_vec_cuda_6',
    'diss_db_vec_cuda_7',
    'diss_db_vec_cuda_8',
    'diss_db_vec_cuda_9',
    'diss_db_vec_cuda_10',
    'diss_db_vec_cuda_11',
    'diss_db_vec_cuda_12',
    'diss_db_vec_cuda_13',
    'diss_db_vec_cuda_14',
    'diss_db_vec_cuda_15',
    'diss_db_vec_pyth_0',
    'diss_db_vec_pyth_1',
    'diss_db_vec_pyth_2',
    'diss_db_vec_pyth_3',
    'diss_db_vec_pyth_4',
    'diss_db_vec_pyth_5',
    'diss_db_vec_pyth_6',
    'diss_db_vec_pyth_7',
    'diss_db_vec_pyth_8',
    'diss_db_vec_pyth_9',
    'diss_db_vec_pyth_10',
    'diss_db_vec_pyth_11',
    'diss_db_vec_pyth_12',
    'diss_db_vec_pyth_13',
    'diss_db_vec_pyth_14',
    'diss_db_vec_pyth_15',
    'rel_diff_dissipator_db_0',
    'rel_diff_dissipator_db_1',
    'rel_diff_dissipator_db_2',
    'rel_diff_dissipator_db_3',
    'rel_diff_dissipator_db_4',
    'rel_diff_dissipator_db_5',
    'rel_diff_dissipator_db_6',
    'rel_diff_dissipator_db_7',
    'rel_diff_dissipator_db_8',
    'rel_diff_dissipator_db_9',
    'rel_diff_dissipator_db_10',
    'rel_diff_dissipator_db_11',
    'rel_diff_dissipator_db_12',
    'rel_diff_dissipator_db_13',
    'rel_diff_dissipator_db_14',
    'rel_diff_dissipator_db_15',
]







'''
#ok
plot(
    df_np2,
    ['rel_diff_gp_sqr',
    'rel_diff_gm_sqr',
    'rel_diff_gp_gm']
)



#ok
plot(
    df_np2,
    ['rel_diff_comm_0', 'rel_diff_comm_1', 'rel_diff_comm_2', 'rel_diff_comm_3', 
    'rel_diff_comm_4', 'rel_diff_comm_5', 'rel_diff_comm_6', 'rel_diff_comm_7', 
    'rel_diff_comm_8', 'rel_diff_comm_9', 'rel_diff_comm_10', 'rel_diff_comm_11', 
    'rel_diff_comm_12', 'rel_diff_comm_13', 'rel_diff_comm_14', 'rel_diff_comm_15']
)





plot(
    df_np2,
    ['rel_diff_dissipator_db_0', 'rel_diff_dissipator_db_1', 'rel_diff_dissipator_db_2', 
    'rel_diff_dissipator_db_3', 'rel_diff_dissipator_db_4', 'rel_diff_dissipator_db_5', 
    'rel_diff_dissipator_db_6', 'rel_diff_dissipator_db_7', 'rel_diff_dissipator_db_8', 
    'rel_diff_dissipator_db_9', 'rel_diff_dissipator_db_10', 'rel_diff_dissipator_db_11', 
    'rel_diff_dissipator_db_12', 'rel_diff_dissipator_db_13', 'rel_diff_dissipator_db_14', 
    'rel_diff_dissipator_db_15']
)



plot(
    df_np2,
    ['diss_db_vec_cuda_0',
    'diss_db_vec_cuda_1',
    'diss_db_vec_cuda_2',
    'diss_db_vec_cuda_3',
    
    'diss_db_vec_pyth_0',
    'diss_db_vec_pyth_1',
    'diss_db_vec_pyth_2',
    'diss_db_vec_pyth_3',
    
    'rel_diff_dissipator_db_0',
    'rel_diff_dissipator_db_1',
    'rel_diff_dissipator_db_2']
)




plot(
    df_np2,
    ['diss_db_vec_cuda_0',
    'diss_db_vec_cuda_1',
    'diss_db_vec_cuda_2',
    'diss_db_vec_cuda_3']
)




plot(
    df_np2,
    ['diss_db_vec_pyth_0',
    'diss_db_vec_pyth_1',
    'diss_db_vec_pyth_2',
    'diss_db_vec_pyth_3']
)




plot(
    df_np2,
    ['rel_diff_dissipator_db_0',
    'rel_diff_dissipator_db_1',
    'rel_diff_dissipator_db_2']
)

'''









########################################################

#row = df_np[31100] 

#row = df_np[8400]

row = df_np[3]


results_row = process_row(row)



results_row['dissipator_db_pyth']


results_row['diss_db_vec_cuda']



results_row['diss_db_vec_pyth']



results_row['rel_diff_dissipator_db']







##################################################
# detailed analysis


#np.set_printoptions(suppress=True, precision=12)

#np.set_printoptions(suppress=False)


# Extract density matrix rho from row
rho = rho_from_row_mpmath(row)

#rho_sp = sp.Matrix(rho)

is_hermitian_rho = is_hermitian_mpmath(rho)
print(is_hermitian_rho)

eps_t_substep = mp(row['eps_t_substep'])

H = get_H_mpmath(eps_t_substep)

#commutator_re_math, commutator_im_math = compute_commutator_separate(rho_re, rho_im, H)
commutator_math = compute_commutator_math_mpmath(rho, H)



commutator_lind = minus_i * commutator_math

is_hermitian_comm = is_hermitian_mpmath(commutator_lind)
print(is_hermitian_comm)

#commutator_lind_sympy = sp.Matrix(commutator_lind)

comm_vec_pyth = rho_to_vec_mpmath(commutator_lind)



comm_vec_cuda = cuda_comm_vec_from_row_mpmath(row)


rel_diff_comm = relative_difference_vec_mpmath(comm_vec_pyth, comm_vec_cuda)



'''
commutator_math = compute_commutator(rho, H)



commutator_diff = commutator_re_math + 1j*commutator_im_math - commutator_math




# Compute Frobenius norm of the difference
commutator_diff_norm = np.linalg.norm(commutator_diff)

# Compute the relative difference (if commutator_math is not all zeroes)
commutator_math_norm = np.linalg.norm(commutator_math)
relative_diff = commutator_diff_norm / commutator_math_norm if commutator_math_norm != 0 else 0

# Print the results
print(f"Frobenius norm of the commutator difference: {commutator_diff_norm}")
print(f"Relative difference: {relative_diff}")
'''


#########################################################




denominator = mpmath.sqrt(delta_C**2 + eps_t_substep**2)

gamma_plus_pyth  = mpmath.sqrt(half * (one + eps_t_substep / denominator))
gamma_minus_pyth = mpmath.sqrt(half * (one - eps_t_substep / denominator))



gp_sqr_cuda = mp(row['gp_sqr'])
gm_sqr_cuda = mp(row['gm_sqr'])
gp_gm_cuda  = mp(row['gp_gm'])


gp_sqr_pyth = gamma_plus_pyth**2
gm_sqr_pyth = gamma_minus_pyth**2
gp_gm_pyth  = gamma_plus_pyth*gamma_minus_pyth



rel_diff_gp_sqr = relative_difference_single_mpmath(gp_sqr_pyth, gp_sqr_cuda)
rel_diff_gm_sqr = relative_difference_single_mpmath(gm_sqr_pyth, gm_sqr_cuda)
rel_diff_gp_gm  = relative_difference_single_mpmath(gp_gm_pyth,  gp_gm_cuda)



interval_dissipator = row['interval_dissipator'] # int, not mpmath



U_analyt_pyth = get_U_by_interval_mpmath(interval_num=interval_dissipator,
                                      gamma_plus=gamma_plus_pyth,
                                      gamma_minus=gamma_minus_pyth)


UTU = U_analyt_pyth.transpose()*U_analyt_pyth

UUT = U_analyt_pyth*U_analyt_pyth.transpose()


H_adb_analyt_pyth = U_analyt_pyth.transpose() * H * U_analyt_pyth



E_numeric_pyth, U_numeric_pyth = mpmath.eigsy(H)

U_numeric_pyth[:, 0] *= -1


H_adb_numeric_pyth = U_numeric_pyth.transpose() * H * U_numeric_pyth


U_diff = U_analyt_pyth - U_numeric_pyth 

H_diff = H_adb_analyt_pyth - H_adb_numeric_pyth 








Gamma_10, Gamma_20, Gamma_30, Gamma_21, Gamma_31, Gamma_32 = compute_Gammas_by_interval_mpmath(interval_num=interval_dissipator, U=U_analyt_pyth)




print('Gamma_10 =', Gamma_10)
print('Gamma_20 =', Gamma_20)
print('Gamma_30 =', Gamma_30)
print('Gamma_21 =', Gamma_21)
print('Gamma_31 =', Gamma_31)
print('Gamma_32 =', Gamma_32)


#Gamma_10_32_cuda = mp(row['Gamma_10_32'])
#Gamma_20_31_cuda = mp(row['Gamma_20_31'])



L_adb_01 = L_adb_en_mpmath(0,1,Gamma_10)
L_adb_02 = L_adb_en_mpmath(0,2,Gamma_20)
L_adb_03 = L_adb_en_mpmath(0,3,Gamma_30)
L_adb_12 = L_adb_en_mpmath(1,2,Gamma_21)
L_adb_13 = L_adb_en_mpmath(1,3,Gamma_31)
L_adb_23 = L_adb_en_mpmath(2,3,Gamma_32)


L_db_01 = U_analyt_pyth * L_adb_01 * U_analyt_pyth.transpose()
L_db_02 = U_analyt_pyth * L_adb_02 * U_analyt_pyth.transpose()
L_db_03 = U_analyt_pyth * L_adb_03 * U_analyt_pyth.transpose()
L_db_12 = U_analyt_pyth * L_adb_12 * U_analyt_pyth.transpose()
L_db_13 = U_analyt_pyth * L_adb_13 * U_analyt_pyth.transpose()
L_db_23 = U_analyt_pyth * L_adb_23 * U_analyt_pyth.transpose()




L_db_array = [L_db_01, L_db_02, L_db_03, L_db_12, L_db_13, L_db_23]

dissipator_db_pyth = compute_dissipator_mpmath(rho, L_db_array)




is_hermitian_diss = is_hermitian_mpmath(dissipator_db_pyth)
print(is_hermitian_diss)

#dissipator_db_sympy = sp.Matrix(dissipator_db_pyth)

diss_db_vec_pyth = rho_to_vec_mpmath(dissipator_db_pyth)
diss_db_vec_cuda = cuda_dissipator_vec_from_row_mpmath(row)

rel_diff_dissipator_db = relative_difference_vec_mpmath(diss_db_vec_pyth, diss_db_vec_cuda)

#########################################################


# Element-wise addition
total_rhs_vec_pyth = [a + b for a, b in zip(comm_vec_pyth, diss_db_vec_pyth)]

total_rhs_vec_cuda = cuda_total_rhs_vec_from_row_mpmath(row)

rel_diff_total_rhs = relative_difference_vec_mpmath(total_rhs_vec_pyth, total_rhs_vec_cuda)














#########################################################
#########################################################
#########################################################
#########################################################
#########################################################

# test simplified code for cuda in the first place

rho_vec = cuda_rho_vec_from_row_mpmath(row)


r00 = mp(row['rho_in_0'])
r11 = mp(row['rho_in_1'])
r22 = mp(row['rho_in_2'])
r33 = mp(row['rho_in_3'])

r01 = mp(row['rho_in_4'])
i01 = mp(row['rho_in_5'])
r02 = mp(row['rho_in_6'])
i02 = mp(row['rho_in_7'])
r03 = mp(row['rho_in_8'])
i03 = mp(row['rho_in_9'])

r12 = mp(row['rho_in_10'])
i12 = mp(row['rho_in_11'])
r13 = mp(row['rho_in_12'])
i13 = mp(row['rho_in_13'])
r23 = mp(row['rho_in_14'])
i23 = mp(row['rho_in_15'])











'''
#####
# unrolled expressions v1 (for interval 4)


g_p = gamma_plus_pyth
g_m = gamma_minus_pyth







drho_out_D_r00 = -Gamma_lmrp*r00 - Gamma_lprm*r00;


drho_out_D_r11 = Gamma_lmrp*g_m**2*r33 + Gamma_lmrp*g_p**2*r00 + Gamma_lprm*g_m**2*r00 + Gamma_lprm*g_p**2*r33;


drho_out_D_r22 = Gamma_lmrp*g_m**2*r00 + Gamma_lmrp*g_p**2*r33 + Gamma_lprm*g_m**2*r33 + Gamma_lprm*g_p**2*r00;


drho_out_D_r33 = -Gamma_lmrp*r33 - Gamma_lprm*r33;


drho_out_D_r01 = -Gamma_lmrp*r01/2 - Gamma_lprm*r01/2;


drho_out_D_i01 = -Gamma_lmrp*i01/2 - Gamma_lprm*i01/2;


drho_out_D_r02 = -Gamma_lmrp*r02/2 - Gamma_lprm*r02/2;


drho_out_D_i02 = -Gamma_lmrp*i02/2 - Gamma_lprm*i02/2;


drho_out_D_r03 = -Gamma_lmrp*r03 - Gamma_lprm*r03;


drho_out_D_i03 = -Gamma_lmrp*i03 - Gamma_lprm*i03;


drho_out_D_r12 = Gamma_lmrp*g_m*g_p*r00 - Gamma_lmrp*g_m*g_p*r33 - Gamma_lprm*g_m*g_p*r00 + Gamma_lprm*g_m*g_p*r33;


drho_out_D_i12 = zero;


drho_out_D_r13 = -Gamma_lmrp*r13/2 - Gamma_lprm*r13/2;


drho_out_D_i13 = -Gamma_lmrp*i13/2 - Gamma_lprm*i13/2;


drho_out_D_r23 = -Gamma_lmrp*r23/2 - Gamma_lprm*r23/2;


drho_out_D_i23 = -Gamma_lmrp*i23/2 - Gamma_lprm*i23/2;




# end of unrolled expressions v1
####
'''



#####
# unrolled expressions v2 (for interval 4)


GammaLR0 = Gamma_L0 + Gamma_R0

gp_sqr = gp_sqr_pyth
gm_sqr = gm_sqr_pyth
gp_gm  = gp_gm_pyth







tmp = 0
tmp = -2*GammaLR0*r00;
tmp *= half
drho_out_D_r00 = tmp


tmp = 0
tmp = 2*r00*(Gamma_lmrp*gp_sqr + Gamma_lprm*gm_sqr);
tmp += 2*r33*(Gamma_lmrp*gm_sqr + Gamma_lprm*gp_sqr);
tmp *= half
drho_out_D_r11 = tmp


tmp = 0
tmp = 2*r00*(Gamma_lmrp*gm_sqr + Gamma_lprm*gp_sqr);
tmp += 2*r33*(Gamma_lmrp*gp_sqr + Gamma_lprm*gm_sqr);
tmp *= half
drho_out_D_r22 = tmp


tmp = 0
tmp = -2*GammaLR0*r33;
tmp *= half
drho_out_D_r33 = tmp


tmp = 0
tmp = -GammaLR0*r01;
tmp *= half
drho_out_D_r01 = tmp


tmp = 0
tmp = -GammaLR0*i01;
tmp *= half
drho_out_D_i01 = tmp


tmp = 0
tmp = -GammaLR0*r02;
tmp *= half
drho_out_D_r02 = tmp


tmp = 0
tmp = -GammaLR0*i02;
tmp *= half
drho_out_D_i02 = tmp


tmp = 0
tmp = -2*GammaLR0*r03;
tmp *= half
drho_out_D_r03 = tmp


tmp = 0
tmp = -2*GammaLR0*i03;
tmp *= half
drho_out_D_i03 = tmp


tmp = 0
tmp = 2*gp_gm*r00*(Gamma_lmrp - Gamma_lprm);
tmp += 2*gp_gm*r33*(-Gamma_lmrp + Gamma_lprm);
tmp *= half
drho_out_D_r12 = tmp


tmp = 0
tmp = 0;
tmp *= half
drho_out_D_i12 = tmp


tmp = 0
tmp = -GammaLR0*r13;
tmp *= half
drho_out_D_r13 = tmp


tmp = 0
tmp = -GammaLR0*i13;
tmp *= half
drho_out_D_i13 = tmp


tmp = 0
tmp = -GammaLR0*r23;
tmp *= half
drho_out_D_r23 = tmp


tmp = 0
tmp = -GammaLR0*i23;
tmp *= half
drho_out_D_i23 = tmp




# end of unrolled expressions v2
####










diss_db_vec_pyth_code_for_cuda = [
    drho_out_D_r00,
    drho_out_D_r11,
    drho_out_D_r22,
    drho_out_D_r33,
    
    drho_out_D_r01,
    drho_out_D_i01,
    
    drho_out_D_r02,
    drho_out_D_i02,
    
    drho_out_D_r03,
    drho_out_D_i03,
    
    drho_out_D_r12,
    drho_out_D_i12,
    
    drho_out_D_r13,
    drho_out_D_i13,
    
    drho_out_D_r23,
    drho_out_D_i23
    
]




rel_diff_dissipator_db_pyth_with_pyth = relative_difference_vec_mpmath(diss_db_vec_pyth_code_for_cuda, diss_db_vec_pyth)



diss_db_vec_pyth

diss_db_vec_pyth_code_for_cuda

diss_db_vec_cuda






























