# CUDA + Python Cross-Platform Project

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/yourusername/yourrepo/actions) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project integrates CUDA for GPU-accelerated computations with Python for high-level scripting and data handling. It focuses on efficient simulation of Lindblad master equations using custom CUDA kernels, supporting both local environments (Linux/WSL2/Windows) and Google Colab for easy experimentation.

## Features
- **Cross-Platform Compatibility**: Works on Linux, WSL2, Windows (with CUDA), and Google Colab.
- **Dynamic GPU Detection**: Build scripts automatically detect and target your GPU architecture.
- **Modular Design**: Separates CUDA kernels, host code, and Python orchestration for maintainability.

## Project Structure
```
.
├── cuda/
│   ├── Makefile                  # CUDA build configuration
│   ├── lindblad_gpu              # Compiled executable (generated)
│   ├── src/
│   │   ├── constants.cuh         # Shared constants for device/host
│   │   └── host/
│   │       ├── host_branch_grid.cuh   # Host-side grid branching logic
│   │       ├── host_branch_single.cuh # Host-side single branching logic
│   │       └── host_helpers.cuh       # Utility functions for host
│   └── main.cu                   # Entry point for CUDA kernels
├── scripts/
│   ├── build_local.sh            # Local build script (Linux/WSL2/Windows)
│   ├── setup_colab.sh            # Colab-specific setup and build
│   └── run.py                    # Python runner for the CUDA executable
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # Project license
```

## Prerequisites

### Local Setup (Linux/WSL2/Windows)
- **CUDA Toolkit**: Version 11.0 or later, with `nvcc` in your PATH.
- **Python**: 3.8+.
- **Build Tools**: `make`, `g++` (or equivalent compiler).
- **NVIDIA GPU**: Compatible with your CUDA version.

### Google Colab
- A free Google account with access to a GPU runtime (T4 or better recommended).

## Setup and Run on Google Colab

1. Open a new Google Colab notebook and enable GPU runtime: `Runtime > Change runtime type > Hardware accelerator > GPU`.

2. Clone the repository and navigate into it (private repo will require authentication):
```bash
!git clone https://github.com/account548567/cuda_python_project.git
%cd cuda_python_project
```

3. Install Python dependencies:
```bash
!pip install -r requirements.txt
```

4. Compile the CUDA program:
```bash
!bash scripts/setup_colab.sh
```

5. Run the Python + CUDA program:
```bash
!python scripts/run.py
```

Expected output: Simulation results printed to the notebook, including timing benchmarks and Lindblad evolution metrics.

## Setup and Run Locally (Linux/WSL2/Windows)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

2. Ensure CUDA Toolkit is installed and `nvcc --version` works.

3. Build the CUDA program:
```bash
chmod +x scripts/build_local.sh  # Run once to make executable
./scripts/build_local.sh
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

5. Run the program:
```bash
python scripts/run.py
```

For debugging, set `DEBUG=1` environment variable before running: `DEBUG=1 python scripts/run.py`.

## Usage

The `run.py` script accepts command-line arguments for customization:
- `--grid-size N`: Set grid dimension (default: 256).
- `--iterations T`: Number of time steps (default: 1000).
- `--output FILE`: Save results to CSV (default: `results.csv`).

Example:
```bash
python scripts/run.py --grid-size 512 --iterations 5000 --output sim_results.csv
```

## Notes
- **GPU Architecture**: Scripts use `deviceQuery` to auto-detect compute capability and set `-arch` flags.
- **Dependencies**: See `requirements.txt` for packages like NumPy, SciPy, and PyTorch (for tensor ops).
- **Performance**: Expect 10-100x speedup over CPU baselines for large grids.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite the following GitHub repository:

@misc{your-repo,
author = {Your Name},
title = {Project Title},
year = {2025},
publisher = {GitHub},
url = {https://github.com/yourusername/yourrepo}
}

Thank you for giving credit to this work. This helps support the project and acknowledge the effort that has gone into developing it.
