// src/host_helpers.cuh

#pragma once

#include <cuda_runtime.h>
#include <cstdio>   // for fprintf
#include <cstdlib>  // for abort, exit



// small GPU check
__host__ inline void gpuCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Error (%s): %s\n", msg, cudaGetErrorString(err));
        std::abort();
        exit(1);
        exit(EXIT_FAILURE);
    }
}


