########################################
# app/build_scripts/build_local_linux.sh
########################################

#!/bin/bash
clear # temp
echo "Building CUDA code locally..."

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

if [[ $gpu_name == *"RTX 3050 Ti"* ]]; then
    ARCH="sm_86"
else
    echo "Warning: GPU '$gpu_name' not recognized."
    echo "Please add your GPU and its architecture to the script."
    echo "Falling back to default ARCH=sm_86"
    ARCH="sm_86"
fi

echo "Detected GPU: $gpu_name"
echo "Using ARCH=$ARCH"

cd app/cuda || exit 1
make clean
make ARCH=$ARCH
cd ../scripts || exit 1

echo "Build complete."
