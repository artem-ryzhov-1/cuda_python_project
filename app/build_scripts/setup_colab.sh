########################################
# app/build_scripts/setup_colab.sh
########################################

#!/bin/bash
cd "$(dirname "$0")/.."   # go to repo root
echo "Now working in: $(pwd)"

#%%bash

# Install necessary packages for Colab
sudo apt-get update
sudo apt-get install -y build-essential

# Confirm nvcc is available (should be preinstalled in Colab's CUDA environment)
nvcc --version

# Detect GPU model
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
echo "Detected GPU: $GPU_NAME"

# Map GPU model to CUDA architecture (sm_xx)
if [[ "$GPU_NAME" == *"T4"* ]]; then
    ARCH="sm_75"
elif [[ "$GPU_NAME" == *"V100"* ]]; then
    ARCH="sm_70"
elif [[ "$GPU_NAME" == *"A100"* ]]; then
    ARCH="sm_80"
elif [[ "$GPU_NAME" == *"P100"* ]]; then
    ARCH="sm_60"
else
    echo "Unknown GPU architecture, defaulting to sm_75"
    ARCH="sm_75"
fi

echo "Using CUDA architecture: $ARCH"

# Compile your CUDA program (adjust paths as needed)
nvcc -O3 -std=c++17 -arch=$ARCH -I cuda/src -I cuda/external cuda/src/main.cu -o cuda/bin/lindblad_gpu


echo "Build complete."
