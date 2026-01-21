# Quantum Interferogram Dynamics Simulator - Documentation

This documentation provides a comprehensive guide to the Quantum Interferogram Dynamics Simulator, a GPU-accelerated tool for simulating quantum dynamics in open quantum systems.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Documentation Index](#documentation-index)
- [Physical Background](#physical-background)
- [System Requirements](#system-requirements)

---

## Overview

The Quantum Interferogram Dynamics Simulator is a research-grade computational tool designed to solve the **Lindblad master equation** for double quantum dot (DQD) systems. It simulates **Landau-Zener-Stückelberg-Majorana (LZSM) interference patterns** in semiconductor qubits, enabling researchers to:

- Explore quantum dynamics under various physical conditions
- Visualize interferogram patterns (quantum capacitance vs. detuning and amplitude)
- Analyze time-evolution dynamics of quantum states
- Identify operational regimes of quantum devices
- Extract device parameters from experimental data

### Key Features

| Feature | Description |
|---------|-------------|
| **GPU Acceleration** | CUDA-based parallel computation for 10-100x speedup |
| **Interactive Dashboard** | Real-time parameter adjustment and visualization |
| **Multiple Simulation Modes** | Grid mode (parameter sweeps) and single-point mode (time dynamics) |
| **Cross-Platform** | Windows, Linux, WSL2, and Google Colab support |
| **Modular Architecture** | Extensible CUDA kernels and Python visualization |

### Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface                         │
│          Panel + HoloViews + Matplotlib                  │
├─────────────────────────────────────────────────────────┤
│                   Python Layer                           │
│    Configuration │ Simulation │ Visualization │ I/O     │
├─────────────────────────────────────────────────────────┤
│                   CUDA Backend                           │
│   Lindblad Solver │ RK4 Integration │ GPU Kernels       │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option 1: Google Colab (Recommended for First-Time Users)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/artem-ryzhov-1/cuda_python_project/blob/main/run_in_colab.ipynb)

No installation required. Click the badge to launch in your browser.

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/artem-ryzhov-1/cuda_python_project.git
cd cuda_python_project

# Install Python dependencies
pip install -r app/requirements.txt

# Compile CUDA code (Windows)
.\app\build_scripts\build_local_windows.bat

# Or on Linux/WSL2
./app/build_scripts/build_local_linux.sh

# Launch the dashboard
panel serve app/launcher/local_app_launcher.py --show
```

---

## Documentation Index

| Document | Description |
|----------|-------------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design, data flow, and technical decisions |
| **[CODE_OVERVIEW.md](CODE_OVERVIEW.md)** | Detailed breakdown of source code structure |
| **[API.md](API.md)** | Simulation parameters and configuration reference |

---

## Physical Background

### The Quantum System

The simulator models a **double quantum dot (DQD)** system with four charge states:

```
|N₁, N₂⟩ = |00⟩, |01⟩, |10⟩, |11⟩
```

Where N₁ and N₂ represent the number of electrons on the left and right dots, respectively.

### Lindblad Master Equation

The density matrix ρ evolves according to:

```
dρ/dt = -i[H, ρ]/ℏ + Σₖ D[Lₖ]ρ
```

Where:
- **H** is the 4×4 diabatic Hamiltonian (capacitive coupling, tunnel coupling, driven detuning)
- **D[Lₖ]** are dissipator terms modeling:
  - Dot-lead coupling (charge transitions with reservoirs)
  - Phonon-induced relaxation (energy relaxation)
  - Quasi-static dephasing (charge noise)

### Observable: Quantum Capacitance

The primary observable is the **quantum capacitance** C_Q(ε₀, A), which displays characteristic interference fringes when plotted against detuning offset (ε₀) and drive amplitude (A).

### Operational Regimes

The simulator helps identify five distinct regimes:
1. **Multi-passage** - Multiple Landau-Zener transitions per period
2. **Double-passage** - Two transitions per period
3. **Single-passage** - One transition per period
4. **Incoherent** - Decoherence-dominated dynamics
5. **No-passage** - Drive amplitude below threshold

---

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | CUDA-capable (Compute 5.0+) | NVIDIA RTX 2070 or better |
| RAM | 8 GB | 16 GB+ |
| Storage | 1 GB | 5 GB |

### Software

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 11.0+ |
| Python | 3.8+ |
| Build Tools | make, g++ (Linux) or MSVC (Windows) |

### Python Dependencies

Core dependencies (see `app/requirements.txt`):
- NumPy, SciPy (numerical computation)
- Panel, HoloViews (interactive dashboard)
- Datashader (large-scale visualization)
- Matplotlib (plotting)

---

## Project Structure Overview

```
cuda_python_project/
├── app/                      # Core application
│   ├── cuda/                 # CUDA simulation engine
│   ├── python/               # Python interface & visualization
│   ├── launcher/             # Application entry points
│   └── build_scripts/        # Platform-specific build scripts
├── codegen/                  # Symbolic code generation utilities
├── docs/                     # This documentation
├── plots/                    # Standalone plotting scripts
├── tests/                    # Validation and testing
└── run_in_colab.ipynb        # Google Colab notebook
```

For detailed structure, see [CODE_OVERVIEW.md](CODE_OVERVIEW.md).

---

## Getting Help

- **Issues**: Report bugs at [GitHub Issues](https://github.com/artem-ryzhov-1/cuda_python_project/issues)
- **Documentation**: See the [documentation index](#documentation-index) above

---

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Citation

```bibtex
@misc{quantum-interferogram-simulator,
  author = {Artem Ryzhov},
  title = {Quantum Interferogram Dynamics Simulator},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/artem-ryzhov-1/cuda_python_project}
}
```
