# System Architecture

This document describes the system architecture of the Quantum Interferogram Dynamics Simulator, including the design philosophy, component interactions, and data flow.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Component Diagram](#component-diagram)
- [Layer Descriptions](#layer-descriptions)
- [Data Flow](#data-flow)
- [Simulation Modes](#simulation-modes)
- [Communication Protocol](#communication-protocol)
- [Design Decisions](#design-decisions)
- [Performance Considerations](#performance-considerations)

---

## Architecture Overview

The simulator follows a **three-layer architecture** with clear separation of concerns:

1. **Presentation Layer** (Python/Panel) - User interface and visualization
2. **Orchestration Layer** (Python) - Simulation coordination and data processing
3. **Computation Layer** (CUDA) - High-performance numerical simulation

This design enables:
- Independent development and testing of each layer
- Easy substitution of visualization frameworks
- Maximum GPU utilization for computation
- Cross-platform compatibility

---

## Component Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐          │
│  │ SimulationParams │  │ InterferogramPlot│  │  DynamicsPlot   │          │
│  │    (Sliders)     │  │   (2D Heatmap)   │  │ (Time Series)   │          │
│  └────────┬─────────┘  └────────▲─────────┘  └────────▲────────┘          │
│           │                     │                     │                   │
│           └─────────────────────┼─────────────────────┘                   │
│                                 │                                         │
│              ┌──────────────────┴──────────────────┐                      │
│              │  InteractiveInterferogramDynamics   │                      │
│              │        (Dashboard Coordinator)      │                      │
│              └──────────────────┬──────────────────┘                      │
└─────────────────────────────────┼─────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼─────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                                  │
│                                 │                                         │
│  ┌────────────────┐  ┌─────────▼──────────┐  ┌────────────────┐           │
│  │   config.py    │──│   simulation.py    │──│   file_io.py   │           │
│  │ (Data Classes) │  │  (Coordination)    │  │ (Binary I/O)   │           │
│  └────────────────┘  └─────────┬──────────┘  └────────────────┘           │
│                                │                                          │
│                     ┌──────────▼──────────┐                               │
│                     │   cuda_runner.py    │                               │
│                     │ (Process Spawning)  │                               │
│                     └──────────┬──────────┘                               │
└────────────────────────────────┼──────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     JSON Config File    │
                    │ (app/cuda/input/*.json) │
                    └────────────┬────────────┘
                                 │
┌────────────────────────────────┼───────────────────────────────────────────┐
│                       COMPUTATION LAYER (CUDA)                             │
│                                │                                           │
│                     ┌──────────▼──────────┐                                │
│                     │       main.cu       │                                │
│                     │   (Entry Point)     │                                │
│                     └──────────┬──────────┘                                │
│                                │                                           │
│           ┌────────────────────┼────────────────────┐                      │
│           │                    │                    │                      │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼─────────┐            │
│  │ host_branch_    │  │ host_branch_    │  │   constants.cuh  │            │
│  │   grid.cuh      │  │   single.cuh    │  │ (Shared Config)  │            │
│  └────────┬────────┘  └────────┬────────┘  └──────────────────┘            │
│           │                    │                                           │
│           └────────┬───────────┘                                           │
│                    │                                                       │
│         ┌──────────▼──────────┐                                            │
│         │    GPU Kernels      │                                            │
│         │  lindblad_kernel_*  │                                            │
│         └──────────┬──────────┘                                            │
│                    │                                                       │
│    ┌───────────────┼───────────────┬───────────────────┐                   │
│    │               │               │                   │                   │
│ ┌──▼───┐    ┌──────▼──────┐  ┌─────▼─────┐    ┌────────▼────────┐          │
│ │ RK4  │    │ Commutator  │  │Dissipators│    │  lindblad_      │          │
│ │Step  │    │  [H,ρ]      │  │  D[L]ρ    │    │  helpers        │          │
│ └──────┘    └─────────────┘  └───────────┘    └─────────────────┘          │
│                                                                            │
│                     ┌──────────────────────┐                               │
│                     │   Binary Output      │                               │
│                     │ (app/cuda/output/*)  │                               │
│                     └──────────────────────┘                               │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Descriptions

### Presentation Layer

**Purpose**: User interaction and data visualization

| Component | File | Responsibility |
|-----------|------|----------------|
| SimulationParameters | `app_class_simulation_parameters.py` | Parameter slider widgets for ~15 physics parameters |
| InterferogramPlot | `app_class_interferogram_plot.py` | 2D heatmap of quantum capacitance C_Q(ε₀, A) |
| DynamicsPlot | `app_class_dynamics_plot.py` | Time-evolution plots of state populations |
| InteractiveInterferogramDynamics | `app_class_interactive_interferogram_dynamics.py` | Main dashboard coordinator |

**Technologies**: Panel, HoloViews, Datashader, Matplotlib

### Orchestration Layer

**Purpose**: Coordinate simulation execution and data transformation

| Component | File | Responsibility |
|-----------|------|----------------|
| SimulationConfig | `config.py` | Dataclass with ~50 simulation parameters |
| simulation | `simulation.py` | High-level simulation runner, grid computation |
| cuda_runner | `cuda_runner.py` | Subprocess spawning, JSON config writing |
| file_io | `file_io.py` | Binary file reading, unit conversion |
| helpers | `helpers.py` | Platform detection, build script execution |

### Computation Layer

**Purpose**: High-performance numerical simulation on GPU

| Component | File | Responsibility |
|-----------|------|----------------|
| main | `main.cu` | Entry point, JSON parsing, mode branching |
| constants | `constants.cuh` | Physical constants, device memory declarations |
| host_branch_grid | `host_branch_grid.cuh` | Grid mode host orchestration |
| host_branch_single | `host_branch_single.cuh` | Single-point mode host orchestration |
| lindblad_kernel_grid | `kernels/lindblad_kernel_grid.cuh` | GPU kernel for parameter sweeps |
| lindblad_kernel_single | `kernels/lindblad_kernel_single.cuh` | GPU kernel for time dynamics |
| rk4_step | `rk4/rk4_step.cuh` | 4th-order Runge-Kutta integrator |
| commutator | `commutator/commutator.cuh` | Hamiltonian evolution -i[H,ρ]/ℏ |
| dissipators | `dissipators/*.cuh` | Lindblad dissipator terms |

---

## Data Flow

### Configuration Flow (Python → CUDA)

```
SimulationConfig (Python dataclass)
         │
         ▼
    JSON serialization
         │
         ▼
run_config.json (file on disk)
         │
         ▼
nlohmann/json parsing (C++)
         │
         ▼
Validated parameters in device memory
```

### Execution Flow (Grid Mode)

```
1. User adjusts parameters in dashboard
         │
         ▼
2. Dashboard calls simulation.run_simulation()
         │
         ▼
3. compute_grid() calculates resolution (e.g., 774×645 points)
         │
         ▼
4. cuda_runner writes JSON, spawns CUDA subprocess
         │
         ▼
5. main.cu parses config, validates, converts units
         │
         ▼
6. host_branch_grid allocates GPU memory
         │
         ▼
7. lindblad_kernel_grid launches with one thread per (ε₀, A) point
         │
         ▼
8. Each thread:
   ├── Initializes density matrix ρ(0)
   ├── Loops over N_periods × N_steps_period
   │   └── Calls rk4_step() for each timestep
   │       ├── Computes commutator [H, ρ]
   │       └── Computes dissipators D[L]ρ
   └── Accumulates time-averaged populations
         │
         ▼
9. Results copied back to host, written to rho_avg_out.bin
         │
         ▼
10. file_io reads binary, converts to NumPy arrays
          │
          ▼
11. Calculates quantum capacitance derivative dCQ/dε₀
          │
          ▼
12. InterferogramPlot renders 2D heatmap
```

### Results Flow (CUDA → Python)

```
Binary output file (rho_avg_out.bin)
         │
         ▼
file_io.read_bin_file_gridmode_and_calculate_deriv()
         │
         ▼
NumPy arrays + unit conversion
         │
         ▼
Visualization components render plots
```

---

## Simulation Modes

### Grid Mode

**Purpose**: Sweep over 2D parameter space (ε₀, A) to generate interferograms

**Parallelization**: One GPU thread per (ε₀, A) point

**Output**: Time-averaged populations ⟨ρ_ii⟩_t for each grid point

```
Grid: [ε₀_min, ε₀_max] × [A_min, A_max]
       └── N_eps0 points × N_A points
           └── Each point: full time evolution
```

### Single-Point Mode

**Purpose**: Detailed time dynamics at a specific (ε₀, A) point

**Parallelization**: Time-parallel (multiple periods, or ensemble averaging)

**Output**: Full time series ρ(t) at high temporal resolution

```
Single point: (ε₀, A) fixed
              └── N_periods × N_steps_period timesteps
                  └── Full density matrix at each step
```

### Grid-Single Mode

**Purpose**: Combined mode for interactive exploration

**Workflow**:
1. Run grid mode to generate interferogram
2. User clicks on interferogram point
3. Run single-point mode at clicked location

---

## Communication Protocol

### Python ↔ CUDA Communication

The layers communicate through **files**, not shared memory or direct linking:

| Direction | Mechanism | Format |
|-----------|-----------|--------|
| Python → CUDA | JSON configuration file | UTF-8 JSON |
| CUDA → Python | Binary output files | IEEE 754 double-precision |
| CUDA → Python | Console output (streaming) | UTF-8 text |

### JSON Configuration Schema

```json
{
  "grid_single_mode": 0,        // Mode selector (0=grid, 1=single, 2=both)
  "ouput_option": 1,            // Output verbosity

  // Physical parameters
  "delta_C": 0.01,              // Tunnel coupling (meV)
  "GammaL0": 0.001,             // Left lead coupling
  "GammaR0": 0.001,             // Right lead coupling
  "Gamma_eg0": 0.001,           // Phonon relaxation rate
  "Gamma_phi0": 0.001,          // Dephasing rate
  "nu": 1.0,                    // Drive frequency (GHz)
  "E_C": 1.0,                   // Charging energy (meV)

  // Grid parameters
  "eps0_min": -0.5,
  "eps0_max": 0.5,
  "A_min": 0.0,
  "A_max": 1.0,
  "N_eps0": 100,
  "N_A": 100,

  // Numerical parameters
  "N_steps_period": 1000,
  "N_periods": 100
}
```

### Binary Output Format

**Grid mode** (`rho_avg_out.bin`):
```
[N_eps0 × N_A × 4 doubles]
  └── Flattened 3D array: rho_avg[i_eps0][i_A][i_state]
      └── States: |00⟩, |01⟩, |10⟩, |11⟩ populations
```

**Single mode** (`rho_single_out.bin`):
```
[N_timesteps × 16 doubles]
  └── Full 4×4 density matrix at each timestep
```

---

## Design Decisions

### Why Subprocess Communication?

**Decision**: Python spawns CUDA as a separate process rather than using Python/C++ bindings.

**Rationale**:
1. **Simplicity**: No complex build system for Python bindings
2. **Debugging**: CUDA executable can be tested independently
3. **Flexibility**: Easy to swap Python frontend or add other interfaces
4. **Reliability**: Process isolation prevents GPU errors from crashing Python

**Trade-offs**:
- File I/O overhead (minimal compared to computation time)
- No shared memory (not needed for batch processing)

### Why JSON for Configuration?

**Decision**: Use JSON files for Python → CUDA parameter passing.

**Rationale**:
1. **Human-readable**: Easy to inspect and debug
2. **Widely supported**: nlohmann/json for C++, built-in for Python
3. **Schema flexibility**: Easy to add/remove parameters
4. **Reproducibility**: Config files serve as simulation records

### Why Binary for Results?

**Decision**: Use raw binary files for CUDA → Python data transfer.

**Rationale**:
1. **Performance**: No parsing overhead for large arrays
2. **Precision**: IEEE 754 ensures exact representation
3. **Simplicity**: Direct memory mapping in NumPy

### GPU Memory Management

**Strategy**: Pre-allocate all memory before kernel launch

- Grid mode: One allocation for entire parameter grid
- Single mode: Smaller allocation for time series
- No dynamic allocation during kernel execution

---

## Performance Considerations

### GPU Utilization

- **Thread mapping**: One thread per (ε₀, A) grid point
- **Memory coalescing**: Adjacent threads access adjacent memory
- **Occupancy**: Block size tuned for register usage

### Bottlenecks

| Component | Bottleneck | Mitigation |
|-----------|------------|------------|
| RK4 integration | Register pressure | Manual loop unrolling |
| Dissipator calculation | Memory bandwidth | Sparse matrix representation |
| Grid mode | Launch overhead | Large block sizes |
| Python I/O | File read latency | Memory-mapped files (NumPy) |

### Scaling

| Parameter | Scaling | Notes |
|-----------|---------|-------|
| Grid size | O(N_eps0 × N_A) | Fully parallel |
| Time steps | O(N_periods × N_steps) | Sequential within thread |
| Problem size | O(16) | Fixed 4×4 density matrix |

### Typical Performance

| GPU | Grid Size | Time (Grid Mode) |
|-----|-----------|------------------|
| RTX 2070 | 500×500 | ~5 seconds |
| RTX 3080 | 500×500 | ~2 seconds |
| Tesla T4 (Colab) | 500×500 | ~8 seconds |

---

## Future Architecture Considerations

### Potential Improvements

1. **Multi-GPU support**: Distribute grid points across GPUs
2. **Streaming computation**: Overlap I/O with computation
3. **Mixed precision**: Use FP32 for less critical calculations
4. **WebSocket communication**: Replace file I/O for faster feedback

### Extension Points

- **New physics models**: Add dissipators in `dissipators/` directory
- **Alternative integrators**: Replace RK4 in `rk4/` directory
- **Additional observables**: Extend output in kernel files
- **New visualization**: Add components in `app/python/`
