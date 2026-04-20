# API Reference

This document provides a complete reference for the simulation parameters, configuration options, and Python API.

## Table of Contents

- [Simulation Parameters](#simulation-parameters)
- [Configuration Classes](#configuration-classes)
- [Python API](#python-api)
- [JSON Configuration](#json-configuration)
- [Output Data Formats](#output-data-formats)
- [Physical Units](#physical-units)

---

## Simulation Parameters

### Physical Parameters

These parameters define the quantum system being simulated.

| Parameter | Symbol | Units | Default | Description |
|-----------|--------|-------|---------|-------------|
| `delta_C` | Δ_C | meV | 0.01 | Tunnel coupling between dots |
| `E_C` | E_C | meV | 1.0 | Charging energy |
| `nu` | ν | GHz | 1.0 | Drive frequency |
| `GammaL0` | Γ_L | meV | 0.001 | Left lead coupling rate |
| `GammaR0` | Γ_R | meV | 0.001 | Right lead coupling rate |
| `Gamma_eg0` | Γ_eg | meV | 0.001 | Phonon relaxation rate |
| `Gamma_phi0` | Γ_φ | meV | 0.001 | Dephasing rate |

### Drive Parameters

These parameters control the time-dependent driving of the system.

| Parameter | Symbol | Units | Default | Description |
|-----------|--------|-------|---------|-------------|
| `eps0_min` | ε₀_min | meV | -0.5 | Minimum detuning offset (grid mode) |
| `eps0_max` | ε₀_max | meV | 0.5 | Maximum detuning offset (grid mode) |
| `A_min` | A_min | meV | 0.0 | Minimum drive amplitude (grid mode) |
| `A_max` | A_max | meV | 1.0 | Maximum drive amplitude (grid mode) |
| `eps0` | ε₀ | meV | 0.0 | Fixed detuning offset (single mode) |
| `A` | A | meV | 0.5 | Fixed drive amplitude (single mode) |

### Grid Parameters

These parameters define the resolution of parameter sweeps.

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `N_eps0` | - | 100 | Number of grid points in ε₀ direction |
| `N_A` | - | 100 | Number of grid points in A direction |
| `target_grid_points` | - | 500000 | Target total grid points (auto-compute N_eps0, N_A) |
| `grid_aspect_ratio` | - | 1.0 | Aspect ratio of grid (N_eps0/N_A) |

### Numerical Parameters

These parameters control the numerical integration.

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `N_steps_period` | - | 1000 | Time steps per drive period |
| `N_periods` | - | 100 | Number of drive periods to simulate |
| `N_transient_periods` | - | 10 | Periods to discard as transient |
| `dt_factor` | - | 1.0 | Timestep scaling factor |

### Mode Selection

| Parameter | Values | Description |
|-----------|--------|-------------|
| `grid_single_mode` | 0, 1, 2 | 0=grid only, 1=single only, 2=both |
| `output_option` | 0, 1, 2 | Output verbosity level |
| `threads_per_traj_opt` | 32, 64, 128, 256, 512 | CUDA block size |

---

## Configuration Classes

### SimulationConfig

The main configuration dataclass containing all simulation parameters.

```python
from app.python.config import SimulationConfig

config = SimulationConfig(
    # Physical parameters
    delta_C=0.01,           # Tunnel coupling (meV)
    E_C=1.0,                # Charging energy (meV)
    nu=1.0,                 # Drive frequency (GHz)
    GammaL0=0.001,          # Left lead coupling (meV)
    GammaR0=0.001,          # Right lead coupling (meV)
    Gamma_eg0=0.001,        # Phonon relaxation (meV)
    Gamma_phi0=0.001,       # Dephasing rate (meV)

    # Grid parameters
    eps0_min=-0.5,
    eps0_max=0.5,
    A_min=0.0,
    A_max=1.0,
    N_eps0=100,
    N_A=100,

    # Numerical parameters
    N_steps_period=1000,
    N_periods=100,
    N_transient_periods=10,

    # Mode selection
    grid_single_mode=0,     # 0=grid, 1=single, 2=both
    output_option=1,
    threads_per_traj_opt=256
)
```

### SimRunGridMode

Configuration for grid mode (parameter sweeps).

```python
from app.python.config import SimRunGridMode

grid_config = SimRunGridMode(
    eps0_min=-0.5,
    eps0_max=0.5,
    A_min=0.0,
    A_max=1.0,
    N_eps0=100,
    N_A=100
)
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `eps0_min` | float | Minimum detuning offset |
| `eps0_max` | float | Maximum detuning offset |
| `A_min` | float | Minimum drive amplitude |
| `A_max` | float | Maximum drive amplitude |
| `N_eps0` | int | Grid resolution in ε₀ |
| `N_A` | int | Grid resolution in A |

### SimRunSingleMode

Configuration for single-point mode (time dynamics).

```python
from app.python.config import SimRunSingleMode

single_config = SimRunSingleMode(
    eps0=0.1,
    A=0.5,
    N_steps_period=1000,
    N_periods=100
)
```

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `eps0` | float | Fixed detuning offset |
| `A` | float | Fixed drive amplitude |
| `N_steps_period` | int | Time steps per period |
| `N_periods` | int | Number of periods |

### SimRunGridSingleMode

Combined configuration for grid + single mode.

```python
from app.python.config import SimRunGridSingleMode

combined_config = SimRunGridSingleMode(
    grid=grid_config,
    single=single_config
)
```

---

## Python API

### simulation.py

#### `run_simulation()`

Main entry point for running simulations.

```python
def run_simulation(
    config: SimulationConfig,
    mode: Union[SimRunGridMode, SimRunSingleMode, SimRunGridSingleMode],
    repo_path: str = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]
```

**Parameters**:
- `config`: Simulation configuration
- `mode`: Mode-specific configuration
- `repo_path`: Path to repository root
- `verbose`: Enable verbose output

**Returns**: Dictionary with simulation results

**Example**:
```python
from app.python.simulation import run_simulation
from app.python.config import SimulationConfig, SimRunGridMode

config = SimulationConfig(...)
mode = SimRunGridMode(eps0_min=-0.5, eps0_max=0.5, ...)

results = run_simulation(config, mode)
rho_avg = results['rho_avg']      # [N_eps0, N_A, 4]
dCQ_deps = results['dCQ_deps']    # [N_eps0, N_A]
```

#### `compute_grid()`

Compute grid dimensions from target point count.

```python
def compute_grid(
    target_points: int,
    aspect_ratio: float = 1.0
) -> Tuple[int, int]
```

**Parameters**:
- `target_points`: Target total number of grid points
- `aspect_ratio`: Desired N_eps0/N_A ratio

**Returns**: Tuple of (N_eps0, N_A)

**Example**:
```python
N_eps0, N_A = compute_grid(500000, aspect_ratio=1.2)
# Returns: (774, 645)
```

### cuda_runner.py

#### `run_gpu_lindblad_program()`

Execute CUDA program as subprocess.

```python
def run_gpu_lindblad_program(
    config_path: str,
    executable_path: str,
    timeout: int = None
) -> subprocess.Popen
```

**Parameters**:
- `config_path`: Path to JSON configuration file
- `executable_path`: Path to CUDA executable
- `timeout`: Optional timeout in seconds

**Returns**: subprocess.Popen handle

**Example**:
```python
from app.python.cuda_runner import run_gpu_lindblad_program

process = run_gpu_lindblad_program(
    "app/cuda/input/run_config.json",
    "app/cuda/bin/lindblad_gpu.exe"
)
process.wait()
```

### file_io.py

#### `read_bin_file_gridmode_and_calculate_deriv()`

Read grid mode results and compute quantum capacitance.

```python
def read_bin_file_gridmode_and_calculate_deriv(
    filepath: str,
    N_eps0: int,
    N_A: int,
    eps0_min: float,
    eps0_max: float
) -> Tuple[np.ndarray, np.ndarray]
```

**Parameters**:
- `filepath`: Path to binary output file
- `N_eps0`: Grid resolution in ε₀
- `N_A`: Grid resolution in A
- `eps0_min`, `eps0_max`: Detuning range (for derivative)

**Returns**:
- `rho_avg`: Time-averaged populations [N_eps0, N_A, 4]
- `dCQ_deps`: Quantum capacitance derivative [N_eps0, N_A]

**Example**:
```python
from app.python.file_io import read_bin_file_gridmode_and_calculate_deriv

rho_avg, dCQ_deps = read_bin_file_gridmode_and_calculate_deriv(
    "app/cuda/output/rho_avg_out.bin",
    N_eps0=100, N_A=100,
    eps0_min=-0.5, eps0_max=0.5
)
```

#### `read_bin_file_singlemode()`

Read single-point mode results.

```python
def read_bin_file_singlemode(
    filepath: str,
    N_timesteps: int
) -> np.ndarray
```

**Parameters**:
- `filepath`: Path to binary output file
- `N_timesteps`: Total number of timesteps

**Returns**: Density matrix time series [N_timesteps, 4, 4]

**Example**:
```python
from app.python.file_io import read_bin_file_singlemode

N_timesteps = 1000 * 100  # N_steps_period * N_periods
rho_t = read_bin_file_singlemode(
    "app/cuda/output/rho_single_out.bin",
    N_timesteps
)
```

### helpers.py

#### `detect_cupy_cudf()`

Check for GPU acceleration availability.

```python
def detect_cupy_cudf() -> bool
```

**Returns**: True if CuPy/cuDF are available

#### `run_build_script()`

Execute platform-specific build script.

```python
def run_build_script(platform: str) -> int
```

**Parameters**:
- `platform`: One of "windows", "linux", "wsl2", "colab"

**Returns**: Exit code (0 = success)

---

## JSON Configuration

The CUDA program reads configuration from `app/cuda/input/run_config.json`.

### Complete JSON Schema

```json
{
  // Mode Selection
  "grid_single_mode": 0,
  "ouput_option": 1,
  "threads_per_traj_opt": 256,

  // Physical Parameters
  "delta_C": 0.01,
  "E_C": 1.0,
  "nu": 1.0,
  "GammaL0": 0.001,
  "GammaR0": 0.001,
  "Gamma_eg0": 0.001,
  "Gamma_phi0": 0.001,

  // Lead Parameters
  "muL": 0.0,
  "muR": 0.0,
  "T_leads": 0.0,

  // Grid Mode Parameters
  "eps0_min": -0.5,
  "eps0_max": 0.5,
  "A_min": 0.0,
  "A_max": 1.0,
  "N_eps0": 100,
  "N_A": 100,

  // Single Mode Parameters
  "eps0_single": 0.0,
  "A_single": 0.5,

  // Numerical Parameters
  "N_steps_period": 1000,
  "N_periods": 100,
  "N_transient_periods": 10,

  // Initial State
  "rho_init_00": 1.0,
  "rho_init_01": 0.0,
  "rho_init_10": 0.0,
  "rho_init_11": 0.0,

  // Advanced Options
  "renormalize_every_n_steps": 100,
  "clamp_populations": true,
  "enforce_hermiticity": true,

  // Logging
  "log_enabled": false,
  "log_every_n_steps": 10,
  "log_max_entries": 10000
}
```

### Parameter Descriptions

#### Mode Selection

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `grid_single_mode` | int | 0, 1, 2 | 0=grid, 1=single, 2=both |
| `ouput_option` | int | 0, 1, 2 | 0=silent, 1=normal, 2=verbose |
| `threads_per_traj_opt` | int | 32-1024 | CUDA block size |

#### Physical Parameters

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `delta_C` | float | meV | Tunnel coupling |
| `E_C` | float | meV | Charging energy |
| `nu` | float | GHz | Drive frequency |
| `GammaL0` | float | meV | Left lead coupling |
| `GammaR0` | float | meV | Right lead coupling |
| `Gamma_eg0` | float | meV | Phonon relaxation |
| `Gamma_phi0` | float | meV | Dephasing rate |

#### Lead Parameters

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `muL` | float | meV | Left lead chemical potential |
| `muR` | float | meV | Right lead chemical potential |
| `T_leads` | float | K | Lead temperature |

---

## Output Data Formats

### Grid Mode Output

**File**: `app/cuda/output/rho_avg_out.bin`

**Format**: Raw binary, IEEE 754 double-precision

**Structure**:
```
[N_eps0 × N_A × 4] doubles
  └── Flattened 3D array: rho_avg[i_eps0][i_A][i_state]
      └── i_state: 0=|00⟩, 1=|01⟩, 2=|10⟩, 3=|11⟩
```

**Reading in Python**:
```python
import numpy as np

data = np.fromfile("rho_avg_out.bin", dtype=np.float64)
rho_avg = data.reshape(N_eps0, N_A, 4)
```

**Memory Layout**: Row-major (C-order)

### Single Mode Output

**File**: `app/cuda/output/rho_single_out.bin`

**Format**: Raw binary, IEEE 754 double-precision

**Structure**:
```
[N_timesteps × 16] doubles
  └── Flattened density matrix at each timestep
      └── ρ[i,j] stored as [ρ_00, ρ_01, ρ_02, ρ_03, ρ_10, ...]
```

**Reading in Python**:
```python
import numpy as np

data = np.fromfile("rho_single_out.bin", dtype=np.float64)
N_timesteps = len(data) // 16
rho_t = data.reshape(N_timesteps, 4, 4)
```

### Debug Log Output

**File**: `app/cuda/output/debug_log.bin`

**Format**: Raw binary, LogEntry structures

**Structure**: See `constants.cuh` for LogEntry definition

---

## Physical Units

### Internal Units (Dimensionless)

The CUDA code uses dimensionless units internally:

| Quantity | Internal Unit | Conversion |
|----------|---------------|------------|
| Energy | E_C | E_physical = E_internal × E_C |
| Time | ℏ/E_C | t_physical = t_internal × ℏ/E_C |
| Rate | E_C/ℏ | Γ_physical = Γ_internal × E_C/ℏ |

### Physical Units (Input/Output)

All input parameters and output results use physical units:

| Quantity | Unit |
|----------|------|
| Energy | meV |
| Frequency | GHz |
| Time | ns |
| Rate | GHz (or meV via ℏ) |
| Temperature | K |

### Conversion Factors

```python
# Physical constants
hbar = 6.582119569e-16  # eV·s = 0.6582 meV·ps

# Example conversions
E_C = 1.0  # meV (charging energy)

# Internal time to physical time (ps)
t_phys = t_internal * hbar / (E_C * 1e-3)

# Physical rate to internal rate
Gamma_internal = Gamma_meV / E_C
```

---

## Dashboard API

### InteractiveInterferogramDynamics

Main dashboard class for interactive exploration.

```python
from app.python.app_class_interactive_interferogram_dynamics import (
    InteractiveInterferogramDynamics
)

dashboard = InteractiveInterferogramDynamics(
    repo_path="/path/to/dqd-lzsm-simulator",
    default_config=config
)

# Get Panel layout for serving
layout = dashboard.get_layout()
```

**Methods**:

| Method | Description |
|--------|-------------|
| `get_layout()` | Return Panel layout object |
| `run_grid_simulation()` | Execute grid mode simulation |
| `run_single_simulation(eps0, A)` | Execute single-point simulation |
| `update_interferogram(data)` | Update interferogram plot |
| `update_dynamics(rho_t)` | Update dynamics plot |
| `on_parameter_change(event)` | Handle parameter changes |
| `on_interferogram_click(event)` | Handle plot clicks |

### SimulationParameters

Parameter slider widget manager.

```python
from app.python.app_class_simulation_parameters import SimulationParameters

params = SimulationParameters(
    delta_C_range=(0.001, 0.1),
    nu_range=(0.1, 10.0),
    # ... other parameter ranges
)

# Get current values
config = params.get_config()

# Register callback
params.on_change(callback_function)
```

### InterferogramPlot

2D heatmap visualization.

```python
from app.python.app_class_interferogram_plot import InterferogramPlot

plot = InterferogramPlot(
    width=600,
    height=500,
    colormap='viridis'
)

# Update with data
plot.update(
    data=dCQ_deps,           # [N_eps0, N_A] array
    eps0_range=(-0.5, 0.5),
    A_range=(0.0, 1.0)
)

# Register click callback
plot.on_click(callback_function)
```

### DynamicsPlot

Time evolution visualization.

```python
from app.python.app_class_dynamics_plot import DynamicsPlot

plot = DynamicsPlot(
    width=600,
    height=400
)

# Update with data
plot.update(
    rho_t=rho_t,             # [N_timesteps, 4, 4] array
    time_array=t             # [N_timesteps] array
)
```
