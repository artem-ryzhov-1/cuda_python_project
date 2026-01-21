# Quantum Interferogram Dynamics Simulator

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/artem-ryzhov-1/cuda_python_project/actions) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project provides a high-performance tool for simulating and visualizing quantum dynamics, specifically focusing on interferometry in open quantum systems. It leverages CUDA for GPU-accelerated solutions of the Lindblad master equation and a Python-based interactive dashboard for real-time parameter tuning and visualization.

The simulator integrates CUDA for GPU-accelerated computations with Python for high-level scripting and data handling, featuring custom CUDA kernels with a 4th-order Runge-Kutta (RK4) method. It is designed for researchers and students in quantum physics, enabling detailed exploration of system dynamics under various physical conditions.

## 🚀 Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/artem-ryzhov-1/cuda_python_project/blob/main/run_in_colab.ipynb)

> No installation required. Click the badge to launch the interactive simulation environment directly in your browser.

## Features

- **High-Performance Simulation**: GPU-accelerated CUDA backend for solving the Lindblad master equation using a 4th-order Runge-Kutta (RK4) method with custom kernels.
- **Interactive Dashboard**: A web-based interface built with Panel and HoloViews allows for real-time adjustment of physical parameters and immediate visualization of results.
- **Cross-Platform Compatibility**: Fully compatible with Linux, WSL2, Windows (with CUDA), and Google Colab.
- **Dynamic GPU Detection**: Build scripts automatically detect and target your GPU architecture for optimal performance.
- **Modular CUDA Kernels**: The CUDA code is structured with modular components for commutators and dissipators, separating kernels, host code, and Python orchestration for maintainability and extensibility.
- **Advanced Visualization**: Generate and inspect detailed interferograms and time-evolution dynamics plots.

## Project Structure

```
.
├── app/
│   ├── requirements.txt       # Python dependencies
│   ├── build_scripts/
│   │   ├── build_local_linux.sh    # Local build script (Linux/WSL2)
│   │   ├── build_local_windows.bat # Local build script (Windows)
│   │   └── setup_colab.sh          # Colab-specific setup and build
│   ├── cuda/                  # Core CUDA source code for the simulation kernel
│   │   ├── Makefile           # CUDA build configuration
│   │   ├── bin/               # Compiled executables (generated)
│   │   │   └── lindblad_gpu.exe
│   │   ├── external/          # External dependencies (nlohmann JSON)
│   │   ├── input/             # Configuration files
│   │   │   └── run_config.json
│   │   ├── output/            # Simulation output data (binary files)
│   │   └── src/
│   │       ├── main.cu        # Entry point for CUDA kernels
│   │       ├── constants.cuh  # Shared constants for device/host
│   │       ├── commutator/    # Commutator computation kernels
│   │       ├── dissipators/   # Dissipator kernels for various physical processes
│   │       ├── host/          # Host-side logic and helpers
│   │       │   ├── host_branch_grid.cuh   # Grid branching logic
│   │       │   ├── host_branch_single.cuh # Single branching logic
│   │       │   ├── host_helpers.cuh       # Utility functions
│   │       │   └── log_writer.cuh         # Logging utilities
│   │       ├── kernels/       # Core Lindblad evolution kernels
│   │       └── rk4/           # Runge-Kutta 4th order integration
│   ├── python/                # Python classes for simulation logic and visualization
│   │   ├── app_class_dynamics_plot.py
│   │   ├── app_class_interactive_interferogram_dynamics.py
│   │   ├── app_class_interferogram_plot.py
│   │   ├── app_class_simulation_parameters.py
│   │   ├── config.py
│   │   ├── cuda_runner.py
│   │   ├── file_io.py
│   │   ├── helpers.py
│   │   ├── simulation.py
│   │   └── deprecated/        # Legacy code versions
│   └── launcher/              # Entry points for launching the application
│       └── local_app_launcher.py
├── codegen/                   # Scripts to auto-generate CUDA code for complex models
│   ├── generate_commutator_code.py
│   ├── generate_dot_lead_dissipators_codes.py
│   └── generate_eg_phi_dissipators_codes.py
├── docs/                      # Detailed documentation
│   ├── ARCHITECTURE.md        # System design and technical decisions
│   ├── CODE_OVERVIEW.md       # Source code structure breakdown
│   └── API.md                 # Simulation parameters and configuration guide
├── plots/                     # Scripts for generating specific plots
│   ├── single_dynamics.py
│   ├── single_interferogram.py
│   └── single_interferogram_and_dynamics.py
├── tests/                     # Tests for verifying simulation output
│   ├── check_log_data.py
│   ├── check_nan_in_outpud_data.py
│   ├── cuda_program_runner_json.py
│   └── log_reader.py
├── run_in_colab.ipynb         # Google Colab notebook for easy cloud execution
├── README.md                  # This file
├── LICENSE                    # Project license (MIT)
└── .gitignore                 # Git ignore rules
```

## Prerequisites

### Local Setup (Linux/WSL2/Windows)

- **NVIDIA GPU**: A CUDA-enabled GPU is required.
- **CUDA Toolkit**: Version 11.0 or later. Ensure `nvcc` is in your system's PATH and verify with `nvcc --version`.
- **Python**: Version 3.8 or higher.
- **Build Tools**: `make` and `g++` (or an equivalent C++ compiler).

### Google Colab

- A Google account with access to a GPU runtime (T4 or better recommended for performance).

## How to Run

### 1. Google Colab (Recommended)

The easiest way to get started is by using the provided Google Colab notebook. Simply click the "Open in Colab" badge at the top of this README. The notebook contains all the necessary steps to set up the environment, compile the code, and launch the interactive dashboard.

**Steps:**

1. Open the Colab notebook and enable GPU runtime: `Runtime > Change runtime type > Hardware accelerator > GPU`.

2. The notebook will automatically:
   - Clone the repository
   - Install Python dependencies
   - Compile the CUDA program using the Colab setup script
   - Launch the interactive dashboard

3. Expected output: An interactive simulation environment with real-time visualization, timing benchmarks, and Lindblad evolution metrics.

### 2. Local Environment (Linux/WSL2/Windows)

**Step 1: Clone the Repository**

```bash
git clone https://github.com/artem-ryzhov-1/cuda_python_project.git
cd cuda_python_project
```

**Step 2: Install Python Dependencies**

```bash
pip install -r app/requirements.txt
```

**Step 3: Compile the CUDA Program**

The build scripts automatically detect your platform and GPU architecture.

- **On Windows:**
  ```cmd
  .\app\build_scripts\build_local_windows.bat
  ```

- **On Linux or WSL2:**
  ```bash
  chmod +x app/build_scripts/build_local_linux.sh
  ./app/build_scripts/build_local_linux.sh
  ```

This will compile the CUDA source and create an executable at `app/cuda/bin/lindblad_gpu`.

**Step 4: Launch the Interactive Dashboard**

Use `panel serve` to run the application launcher:

```bash
panel serve app/launcher/local_app_launcher.py --show
```

This will open the interactive dashboard in your default web browser.

## Usage

The simulation can be customized through the interactive dashboard or via command-line arguments:

- **Grid size**: Set simulation grid dimensions
- **Iterations**: Number of time steps for evolution
- **Physical parameters**: Adjust detuning, coupling strength, decay rates
- **Output format**: Save results to CSV or other formats

Example command-line usage:
```bash
python scripts/run.py --grid-size 512 --iterations 5000 --output sim_results.csv
```

For debugging, set the `DEBUG=1` environment variable before running.

## Performance Notes

- **GPU Architecture**: Scripts use `deviceQuery` to auto-detect compute capability and set appropriate `-arch` flags for optimal compilation.
- **Expected Speedup**: 10-100x speedup over CPU baselines for large grids, depending on GPU and problem size.
- **Dependencies**: Includes NumPy, SciPy, Panel, HoloViews, and other scientific computing packages. See `app/requirements.txt` for the complete list.

## Documentation

For a deeper understanding of the project, please refer to the detailed documentation:

- **[Architecture](./docs/ARCHITECTURE.md)**: An overview of the system design, data flows, and key technical decisions.
- **[Code Overview](./docs/CODE_OVERVIEW.md)**: A detailed breakdown of the source code structure.
- **[API Reference](./docs/API.md)**: A guide to the simulation parameters and configurable options available in the dashboard.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite the following GitHub repository:

```bibtex
@misc{quantum-interferogram-simulator,
  author = {Artem Ryzhov},
  title = {Quantum Interferogram Dynamics Simulator},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/artem-ryzhov-1/cuda_python_project}
}
```

Thank you for giving credit to this work. This helps support the project and acknowledge the effort that has gone into developing it.