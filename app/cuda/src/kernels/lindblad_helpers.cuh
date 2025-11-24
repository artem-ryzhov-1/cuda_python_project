// src/kernels/lindblad_helpers.cuh

#pragma once

#include <cuda_runtime.h>
#include <math.h>
//#include <cmath>
#include <iostream>

// Clamp diagonals to >= 0, then renormalize the trace to 1.
// This operates on the 16-float vector layout used by rk4_step_vec16_full.
static __device__ __forceinline__
void clamp_and_renormalize_vec16(float v[16]) {
    float trace = 0.f;
    #pragma unroll
    for (int i=0;i<4;i++){
        v[i] = fmaxf(0.f, v[i]);
        trace += v[i];
    }
    if (trace > 0.f) {
        float invt = 1.f / trace;
        #pragma unroll
        for (int i=0;i<16;i++) v[i] *= invt;
    }
}


__device__ __forceinline__
void clamp_and_renormalize_vec16_unrolled(
    float& v_0, float& v_1, float& v_2, float& v_3,
    float& v_4, float& v_5, float& v_6, float& v_7,
    float& v_8, float& v_9, float& v_10, float& v_11,
    float& v_12, float& v_13, float& v_14, float& v_15
) {
    float v0 = fmaxf(0.f, v_0);
    float v1 = fmaxf(0.f, v_1);
    float v2 = fmaxf(0.f, v_2);
    float v3 = fmaxf(0.f, v_3);

    float trace = v0 + v1 + v2 + v3;

    if (trace > 0.f) {
        float invt = 1.f / trace;

        v_0 = v0 * invt;
        v_1 = v1 * invt;
        v_2 = v2 * invt;
        v_3 = v3 * invt;

        v_4 *= invt;
        v_5 *= invt;
        v_6 *= invt;
        v_7 *= invt;
        v_8 *= invt;
        v_9 *= invt;
        v_10 *= invt;
        v_11 *= invt;
        v_12 *= invt;
        v_13 *= invt;
        v_14 *= invt;
        v_15 *= invt;
    }
}




__device__ __forceinline__
void clamp_and_renormalize_vec16_unrolled_old(
    float& v_0, float& v_1, float& v_2, float& v_3,
    float& v_4, float& v_5, float& v_6, float& v_7,
    float& v_8, float& v_9, float& v_10, float& v_11,
    float& v_12, float& v_13, float& v_14, float& v_15
) {
    float trace = 0.f;

    // Clamp and compute trace (only diagonal elements v_0, v_1, v_2, v_3)
    v_0 = fmaxf(0.f, v_0); trace += v_0;
    v_1 = fmaxf(0.f, v_1); trace += v_1;
    v_2 = fmaxf(0.f, v_2); trace += v_2;
    v_3 = fmaxf(0.f, v_3); trace += v_3;

    if (trace > 0.f) {
        float invt = 1.f / trace;

        // Renormalize all 16 elements
        v_0 *= invt;
        v_1 *= invt;
        v_2 *= invt;
        v_3 *= invt;
        v_4 *= invt;
        v_5 *= invt;
        v_6 *= invt;
        v_7 *= invt;
        v_8 *= invt;
        v_9 *= invt;
        v_10 *= invt;
        v_11 *= invt;
        v_12 *= invt;
        v_13 *= invt;
        v_14 *= invt;
        v_15 *= invt;
    }
}





__device__ __forceinline__ float simulate_flops(int N) {
    float a = 1.000001f;
    float b = 2.000002f;
    float c = 0.0f;

#pragma unroll 1
    for (int i = 0; i < N; ++i) {
        c = fmaf(a, b, c);
    }

    if (c == -999999.0f) {
        printf("Impossible branch\n");
    }

    return c;
}

