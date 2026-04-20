# Code Overview

This document provides a detailed breakdown of the source code structure, explaining the purpose and functionality of each file in the project.

## Table of Contents

- [Directory Structure](#directory-structure)
- [CUDA Backend](#cuda-backend)
- [Python Interface](#python-interface)
- [Entry Points and Launchers](#entry-points-and-launchers)
- [Code Generation](#code-generation)
- [Tests and Validation](#tests-and-validation)
- [Plotting Scripts](#plotting-scripts)
- [Build System](#build-system)

---

## Directory Structure

```
dqd-lzsm-simulator/
├── app/                          # Core application
│   ├── cuda/                     # CUDA simulation engine
│   │   ├── src/                  # Source code
│   │   ├── input/                # Configuration files
│   │   ├── output/               # Simulation results
│   │   ├── bin/                  # Compiled executables
│   │   ├── external/             # External libraries
│   │   └── Makefile              # Build configuration
│   ├── python/                   # Python interface
│   │   ├── config.py             # Configuration classes
│   │   ├── simulation.py         # Simulation orchestration
│   │   ├── cuda_runner.py        # CUDA process management
│   │   ├── file_io.py            # Binary file I/O
│   │   ├── helpers.py            # Utility functions
│   │   └── app_class_*.py        # Dashboard components
│   ├── launcher/                 # Application entry points
│   │   └── local_app_launcher.py # Main launcher script
│   └── build_scripts/            # Platform-specific builds
│       ├── build_local_windows.bat
│       ├── build_local_linux.sh
│       └── setup_colab.sh
├── codegen/                      # Code generation utilities
│   ├── generate_commutator_code.py
│   ├── generate_dot_lead_dissipators_codes.py
│   └── generate_eg_phi_dissipators_codes.py
├── docs/                         # Documentation
├── plots/                        # Standalone plotting scripts
├── tests/                        # Validation and testing
└── run_in_colab.ipynb            # Google Colab notebook
```

---

## CUDA Backend

Located in `app/cuda/`, this is the high-performance simulation engine.

### Entry Point

#### `src/main.cu`

The main entry point for the CUDA program.

**Responsibilities**:
- Parse command-line arguments
- Read JSON configuration from `input/run_config.json`
- Validate ~50 physical and numerical parameters
- Convert physical units to dimensionless quantities
- Branch to appropriate simulation mode (grid or single)
- Handle errors and provide diagnostic output

**Key Functions**:
```cpp
int main(int argc, char** argv)
  └── Reads JSON config
  └── Validates parameters
  └── Calls host_branch_grid() or host_branch_single()
```

### Constants and Configuration

#### `src/constants.cuh`

Shared constants and device memory declarations.

**Contents**:
- Physical constants (ℏ, π, etc.)
- Device parameters (lattice constant a=0.1, m=10)
- `__constant__` memory declarations for GPU
- `LogEntry` structure for detailed debugging (100+ fields)

**Key Definitions**:
```cpp
#define PI 3.14159265358979323846
#define HBAR 6.582119569e-16  // eV·s

struct LogEntry {
    double t, eps0, A;
    double rho_00, rho_01, rho_10, rho_11;
    // ... 100+ fields for debugging
};
```

### Host-Side Logic

Located in `src/host/`, these files orchestrate GPU execution from the CPU.

#### `src/host/host_branch_grid.cuh`

Grid mode execution logic.

**Responsibilities**:
- Allocate GPU memory for grid results
- Configure kernel launch parameters
- Launch `lindblad_kernel_grid` kernel
- Copy results back to host
- Write binary output file

**Key Functions**:
```cpp
void host_branch_grid(const Config& cfg)
  └── cudaMalloc for grid arrays
  └── lindblad_kernel_grid<<<blocks, threads>>>()
  └── cudaMemcpy D2H
  └── Write rho_avg_out.bin
```

#### `src/host/host_branch_single.cuh`

Single-point mode execution logic.

**Responsibilities**:
- Allocate GPU memory for time series
- Launch `lindblad_kernel_single` kernel
- Copy time-resolved results to host
- Write binary output file

#### `src/host/host_helpers.cuh`

Utility functions for host-side operations.

**Contents**:
- GPU error checking macros
- Memory allocation helpers
- Timer utilities
- Device property queries

#### `src/host/log_writer.cuh`

Debug logging utilities.

**Responsibilities**:
- Write detailed simulation logs
- Format `LogEntry` structures to files
- Support for conditional logging

### GPU Kernels

Located in `src/kernels/`, these are the core computational routines.

#### `src/kernels/lindblad_kernel_grid.cuh`

GPU kernel for grid mode parameter sweeps.

**Algorithm**:
```
For each thread (maps to one (ε₀, A) point):
  1. Initialize density matrix ρ(0) = |00⟩⟨00|
  2. For each period:
       For each timestep:
         - Compute ε(t) = ε₀ + A·sin(ωt)
         - Call rk4_step() to evolve ρ
         - Accumulate time-averaged populations
  3. Store ⟨ρ_ii⟩_t to output array
```

**Parallelization**: One thread per grid point (ε₀, A)

#### `src/kernels/lindblad_kernel_single.cuh`

GPU kernel for single-point time dynamics.

**Algorithm**:
```
For fixed (ε₀, A):
  1. Initialize density matrix ρ(0)
  2. For each timestep:
       - Compute ε(t)
       - Call rk4_step() to evolve ρ
       - Store full ρ(t) to output array
  3. Return complete time series
```

#### `src/kernels/lindblad_helpers.cuh`

Utility functions for kernel execution.

**Contents**:
- Density matrix renormalization
- Trace clamping (ensure Tr(ρ) = 1)
- Hermiticity enforcement
- Positivity enforcement

### Runge-Kutta Integration

Located in `src/rk4/`, implements the 4th-order Runge-Kutta method.

#### `src/rk4/rk4_step.cuh`

Main RK4 time stepper.

**Algorithm**:
```
Given dρ/dt = f(ρ, t):
  k1 = f(ρ_n, t_n)
  k2 = f(ρ_n + dt/2·k1, t_n + dt/2)
  k3 = f(ρ_n + dt/2·k2, t_n + dt/2)
  k4 = f(ρ_n + dt·k3, t_n + dt)
  ρ_{n+1} = ρ_n + dt/6·(k1 + 2k2 + 2k3 + k4)
```

**Key Function**:
```cpp
__device__ void rk4_step(
    Complex rho[4][4],      // Density matrix (in/out)
    double t,               // Current time
    double dt,              // Timestep
    const Params& params    // System parameters
)
```

#### `src/rk4/rk4_substep.cuh`

Individual RK4 substep computation.

**Responsibilities**:
- Compute dρ/dt = -i[H,ρ]/ℏ + Σ_k D[L_k]ρ
- Combine commutator and dissipator contributions

### Hamiltonian Evolution

#### `src/commutator/commutator.cuh`

Computes the coherent evolution term -i[H,ρ]/ℏ.

**Hamiltonian Structure** (4×4 diabatic basis):
```
H = | ε₀₀   Δ_C    0     0   |
    | Δ_C   ε₀₁    0     0   |
    | 0      0    ε₁₀   Δ_C  |
    | 0      0    Δ_C   ε₁₁  |

Where:
  - ε_ij = (E_C/2)(i-j)² + ε(t)·(i-j)
  - Δ_C = tunnel coupling
  - ε(t) = ε₀ + A·sin(ωt) = driven detuning
```

**Key Function**:
```cpp
__device__ void compute_commutator(
    const Complex H[4][4],   // Hamiltonian
    const Complex rho[4][4], // Density matrix
    Complex result[4][4]     // Output: -i[H,ρ]/ℏ
)
```

### Dissipators

Located in `src/dissipators/`, these implement Lindblad superoperator terms.

#### `src/dissipators/dissipator_dot_lead_sparse.cuh`

Dot-lead coupling dissipator.

**Physics**: Models charge tunneling between quantum dots and electron reservoirs (leads).

**Lindblad Form**:
```
D[L]ρ = L·ρ·L† - (1/2){L†L, ρ}
```

Where L are jump operators for electron tunneling in/out.

**Parameters**:
- Γ_L: Left lead coupling rate
- Γ_R: Right lead coupling rate

#### `src/dissipators/dissipator_qubit_dephase_quasi_static.cuh`

Quasi-static dephasing dissipator.

**Physics**: Models charge noise causing random phase fluctuations.

**Effect**: Destroys off-diagonal coherences without changing populations.

**Parameter**: Γ_φ (dephasing rate)

#### `src/dissipators/dissipator_qubit_relax.cuh`

Phonon-induced relaxation dissipator.

**Physics**: Models energy relaxation due to phonon emission/absorption.

**Effect**: Drives system toward thermal equilibrium.

**Parameter**: Γ_eg (relaxation rate)

### External Dependencies

#### `external/nlohmann/json.hpp`

Single-header JSON library for C++11.

**Usage**: Parse JSON configuration files in `main.cu`.

---

## Python Interface

Located in `app/python/`, this provides the high-level Python API.

### Configuration

#### `config.py`

Defines dataclasses for simulation configuration.

**Classes**:

```python
@dataclass
class SimulationConfig:
    """Core simulation parameters (50+ fields)"""
    delta_C: float          # Tunnel coupling (meV)
    GammaL0: float          # Left lead coupling
    GammaR0: float          # Right lead coupling
    Gamma_eg0: float        # Phonon relaxation rate
    Gamma_phi0: float       # Dephasing rate
    nu: float               # Drive frequency (GHz)
    E_C: float              # Charging energy (meV)
    # ... many more fields

@dataclass
class SimRunGridMode:
    """Grid mode configuration"""
    eps0_min: float
    eps0_max: float
    A_min: float
    A_max: float
    N_eps0: int
    N_A: int

@dataclass
class SimRunSingleMode:
    """Single-point mode configuration"""
    eps0: float
    A: float
    N_steps_period: int
    N_periods: int

@dataclass
class SimRunGridSingleMode:
    """Combined grid + single mode"""
    grid: SimRunGridMode
    single: SimRunSingleMode
```

### Simulation Orchestration

#### `simulation.py`

High-level simulation runner.

**Key Functions**:

```python
def run_simulation(config: SimulationConfig,
                   mode: Union[SimRunGridMode, SimRunSingleMode]) -> np.ndarray:
    """
    Main entry point for running simulations.

    1. Validates configuration
    2. Computes grid dimensions (for grid mode)
    3. Calls CUDA program via cuda_runner
    4. Reads and returns results
    """

def compute_grid(target_points: int, aspect_ratio: float) -> Tuple[int, int]:
    """
    Compute grid dimensions for given total points and aspect ratio.

    Example: compute_grid(500000, 1.2) → (774, 645)
    """
```

#### `cuda_runner.py`

CUDA subprocess management.

**Key Functions**:

```python
def run_gpu_lindblad_program(config_path: str,
                             executable_path: str) -> subprocess.Popen:
    """
    Spawn CUDA executable as subprocess.

    Features:
    - Streams stdout/stderr in real-time
    - Handles Ctrl+C gracefully
    - Platform-specific signal handling
    - Returns process handle for monitoring
    """
```

#### `file_io.py`

Binary file I/O and data processing.

**Key Functions**:

```python
def read_bin_file_gridmode_and_calculate_deriv(
    filepath: str,
    N_eps0: int,
    N_A: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read grid mode results and compute quantum capacitance.

    Returns:
    - rho_avg: Time-averaged populations [N_eps0, N_A, 4]
    - dCQ_deps: Quantum capacitance derivative [N_eps0, N_A]
    """

def read_bin_file_singlemode(
    filepath: str,
    N_timesteps: int
) -> np.ndarray:
    """
    Read single-point mode results.

    Returns:
    - rho_t: Full density matrix time series [N_timesteps, 4, 4]
    """
```

#### `helpers.py`

Utility functions.

**Key Functions**:

```python
def detect_cupy_cudf() -> bool:
    """Check if GPU acceleration (CuPy/cuDF) is available."""

def run_build_script(platform: str) -> int:
    """Execute platform-specific CUDA build script."""

def run_test_cuda_program() -> int:
    """Run quick test of CUDA executable."""

def launch_app_colab(port: int = 5006) -> None:
    """Set up and launch server for Google Colab."""
```

### Dashboard Components

#### `app_class_simulation_parameters.py`

Parameter slider widgets.

**Class**: `SimulationParameters`

**Features**:
- ~15 physics parameters with sliders
- Defined ranges and defaults
- Validation and callbacks
- Integration with Panel widgets

**Parameters exposed**:
- Tunnel coupling (Δ_C)
- Lead coupling rates (Γ_L, Γ_R)
- Relaxation/dephasing rates
- Drive frequency and amplitude
- Charging energy

#### `app_class_interferogram_plot.py`

2D heatmap visualization.

**Class**: `InterferogramPlot`

**Features**:
- Renders C_Q(ε₀, A) as 2D heatmap
- Multiple render modes (raster/vector)
- GPU-accelerated rendering (Datashader)
- Interactive point selection (click)
- Colormap customization
- Axis labeling and scaling

**Methods**:
```python
def update(self, data: np.ndarray, eps0_range, A_range):
    """Update plot with new simulation data."""

def on_click(self, event):
    """Handle click events for point selection."""
```

#### `app_class_dynamics_plot.py`

Time evolution visualization.

**Class**: `DynamicsPlot`

**Features**:
- Plot populations P_ij(t) vs. time
- Multiple traces for different states
- Legend with state labels
- Customizable time range
- Integration with Matplotlib/HoloViews

**Methods**:
```python
def update(self, rho_t: np.ndarray, time_array: np.ndarray):
    """Update plot with new time dynamics data."""
```

#### `app_class_interactive_interferogram_dynamics.py`

Main dashboard coordinator.

**Class**: `InteractiveInterferogramDynamics`

**Features**:
- Coordinates all dashboard components
- Parameter change callbacks
- Auto-update mode
- Click-to-compute workflow
- Layout management

**Workflow**:
```python
def run_grid_simulation(self):
    """Run grid mode and update interferogram."""

def run_single_simulation(self, eps0: float, A: float):
    """Run single-point and update dynamics plot."""

def on_parameter_change(self, event):
    """Handle parameter slider changes."""

def on_interferogram_click(self, event):
    """Handle clicks on interferogram for point selection."""
```

---

## Entry Points and Launchers

### `app/launcher/local_app_launcher.py`

Main application entry point.

**Responsibilities**:
- Platform detection (Windows/Linux/WSL2/Colab)
- Repository path configuration
- Default parameter setup
- Panel server launch

**Usage**:
```bash
panel serve app/launcher/local_app_launcher.py --show
```

**Platform-Specific Behavior**:
```python
if platform == "windows":
    repo_path = r"C:\path\to\dqd-lzsm-simulator"
elif platform == "wsl2":
    repo_path = "/mnt/c/path/to/dqd-lzsm-simulator"
elif platform == "colab":
    repo_path = "/content/dqd-lzsm-simulator"
```

---

## Code Generation

Located in `codegen/`, these scripts auto-generate complex CUDA code using symbolic computation.

### `generate_commutator_code.py`

Generates optimized commutator code.

**Approach**:
1. Define symbolic 4×4 Hamiltonian using SymPy
2. Define symbolic density matrix
3. Compute [H, ρ] symbolically
4. Apply simplifications
5. Generate C++/CUDA code with loop unrolling

**Output**: Optimized CUDA code for `commutator.cuh`

### `generate_dot_lead_dissipators_codes.py`

Generates dot-lead coupling dissipators.

**Approach**:
1. Define Lindblad operators for electron tunneling
2. Compute D[L]ρ = LρL† - (1/2){L†L, ρ}
3. Exploit sparsity structure
4. Generate interval-specific code

### `generate_eg_phi_dissipators_codes.py`

Generates relaxation and dephasing dissipators.

**Approach**:
1. Define phonon operators
2. Define dephasing operators
3. Compute dissipator terms
4. Generate optimized CUDA code

---

## Tests and Validation

Located in `tests/`, these scripts verify simulation correctness.

### `cuda_program_runner_json.py`

Direct CUDA execution test.

**Purpose**: Run CUDA program with specific JSON config and verify execution.

**Usage**:
```bash
python tests/cuda_program_runner_json.py
```

### `check_log_data.py`

Numerical validation of logged data.

**Purpose**:
- Parse debug log files
- Verify numerical accuracy
- Check conservation laws
- Detect numerical instabilities

### `check_nan_in_outpud_data.py`

NaN detection in output data.

**Purpose**:
- Read binary output files
- Scan for NaN/Inf values
- Report problematic parameter points

### `log_reader.py`

Debug log parser.

**Purpose**:
- Parse LogEntry structures from log files
- Format for analysis
- Support for selective field extraction

---

## Plotting Scripts

Located in `plots/`, standalone scripts for specific visualizations.

### `single_dynamics.py`

Plot time dynamics only.

**Usage**:
```bash
python plots/single_dynamics.py --eps0 0.1 --A 0.5
```

### `single_interferogram.py`

Plot interferogram only.

**Usage**:
```bash
python plots/single_interferogram.py --output interferogram.png
```

### `single_interferogram_and_dynamics.py`

Combined interferogram and dynamics plot.

**Usage**:
```bash
python plots/single_interferogram_and_dynamics.py
```

---

## Build System

### `app/cuda/Makefile`

CUDA build configuration.

**Targets**:
```makefile
all: lindblad_gpu     # Build main executable
clean:                # Remove build artifacts
test: lindblad_gpu    # Build and run tests
```

**Key Variables**:
```makefile
NVCC = nvcc
ARCH = -arch=sm_75    # Set by build script
CFLAGS = -O3 -Isrc -Iexternal
OUTPUT = bin/lindblad_gpu
```

### `app/build_scripts/build_local_windows.bat`

Windows build script.

**Features**:
- GPU architecture detection via `deviceQuery`
- Sets appropriate `-arch` flag
- Calls `make` in `app/cuda/`

### `app/build_scripts/build_local_linux.sh`

Linux/WSL2 build script.

**Features**:
- Similar to Windows version
- Uses bash environment
- Proper path handling

### `app/build_scripts/setup_colab.sh`

Google Colab setup script.

**Features**:
- Install CUDA toolkit if needed
- Configure for Colab GPU (typically sm_75)
- Compile CUDA code
- Install Python dependencies
