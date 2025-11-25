////////////////////////////////////////
// app/cuda/src/kernels/lindblad_kernel_grid.cuh
////////////////////////////////////////

#pragma once
#include <cuda_runtime.h>
#include <math.h>
// #include "cuda_intellisense_fixes.cuh"
#include "rk4/rk4_step.cuh"
// #include "rk4_step_warp_shfl.cuh"
// #include "rk4_step_warp_shmem.cuh"
#include "kernels/lindblad_helpers.cuh" // inline clamp_and_renormalize_vec16
#include <cstdio>
#include "constants.cuh"

/* good. only for faster compilation

extern "C" __global__ void lindblad_rk4_kernel(
    const float* __restrict__ d_eps0,
    const float* __restrict__ d_A,
    float* __restrict__ d_out_avg
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //if (tid == 0) { // Limit to one thread to avoid excessive output
    //    printf("\n");
    //    printf("Npoints=%d\n", Npoints);
    //    printf("\n");
    //}

    if (tid >= Npoints) return;



    const float eps0 = d_eps0[tid];
    const float A = d_A[tid];

    // rho in vec16 form, initial state
    float rho_vec[16];
#pragma unroll
    for (int i = 0; i < 16; i++) rho_vec[i] = 0.f;
    rho_vec[0] = rho00_init;
    rho_vec[1] = rho11_init;
    rho_vec[2] = rho22_init;
    rho_vec[3] = rho33_init;

    if (tid == 0) {   // only one thread prints to avoid flooding
        printf("Thread tid=0 started. Message from thread tid=0.\n");
        printf("rho00_init = %f\n", rho_vec[0]);
        printf("rho11_init = %f\n", rho_vec[1]);
        printf("rho22_init = %f\n", rho_vec[2]);
        printf("rho33_init = %f\n", rho_vec[3]);
        printf("Npoints=%d\n", Npoints);
    }

    //float sum_whole[4] = { 0.f,0.f,0.f,0.f };
    float sum_last[4] = { 0.f,0.f,0.f,0.f };
    //float sum_2last[4] = { 0.f,0.f,0.f,0.f };
    //float sum_3last[4] = { 0.f,0.f,0.f,0.f };

    //const int total_steps = N_steps_per_period * N_periods;

    for (int t_idx_rk4_step = 0; t_idx_rk4_step < N_steps_per_period * N_periods; ++t_idx_rk4_step) {
        const float t_step = t_idx_rk4_step * dt;


        //if (tid == 0 && t_idx_rk4_step % 1000 == 0)
        //if (tid == 0 && t_idx_rk4_step < 100)
        //if (tid == 0)

        //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
        //    printf("before RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
        //}

        rk4_step(
            rho_vec, t_step,
            eps0, A
        );

        //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
        //    printf(" after RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
        //}

        // post-step stabilization
        clamp_and_renormalize_vec16(rho_vec);

        //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
        //    printf(" after ren: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
        //    printf("\n");
        //}


        // accumulate populations (diagonals are rho_vec[0..3])
//#pragma unroll
        //for (int q = 0; q < 4; q++) sum_whole[q] += rho_vec[q];

        const int period_idx = t_idx_rk4_step / N_steps_per_period;
        if (period_idx == N_periods - 1) {
#pragma unroll
            for (int q = 0; q < 4; q++) sum_last[q] += rho_vec[q];
        }
        //        else if (period_idx == N_periods - 2) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
        //        }
        //        else if (period_idx == N_periods - 3) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
        //        }

    }

    // final normalization safeguard
    clamp_and_renormalize_vec16(rho_vec);

    //const float inv_whole = 1.0f / float(N_steps_per_period * N_periods);
    const float inv_period = 1.0f / float(N_steps_per_period);

    //// for avg_periods_ouput_option == "whole_last_2last_3last":
    //const size_t base = (size_t)tid * 16u;
    //// write: first 4 = whole avg, then last, 2last, 3last
    //for (int d = 0; d < 4; ++d) d_out_avg[base + d] = sum_whole[d] * inv_whole;
    //for (int d = 0; d < 4; ++d) d_out_avg[base + 4 + d] = sum_last[d] * inv_period;
    //for (int d = 0; d < 4; ++d) d_out_avg[base + 8 + d] = sum_2last[d] * inv_period;
    //for (int d = 0; d < 4; ++d) d_out_avg[base + 12 + d] = sum_3last[d] * inv_period;

    //// for avg_periods_ouput_option == "last_2last":
    //const size_t base = (size_t)tid * 8u;
    //// write: first 2 = last, 2last
    //for (int d = 0; d < 4; ++d) d_out_avg[base + d] = sum_last[d] * inv_period;
    //for (int d = 0; d < 4; ++d) d_out_avg[base + 4 + d] = sum_2last[d] * inv_period;

    // for avg_periods_ouput_option == "last":
    const size_t base = (size_t)tid * 4u;
    // write: first 1 = last
    for (int d = 0; d < 4; ++d) d_out_avg[base + d] = sum_last[d] * inv_period;

}
*/



extern "C" __global__ void lindblad_rk4_kernel_unrolled(
    const float* __restrict__ d_eps0,
    const float* __restrict__ d_A,
    float* __restrict__ d_out_avg,

    const int N_periods_avg
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //if (tid == 0) { // Limit to one thread to avoid excessive output
    //    printf("\n");
    //    printf("Npoints=%d\n", Npoints);
    //    printf("\n");
    //}

    if (tid >= Npoints) return;



    const float eps0 = d_eps0[tid];
    const float A = d_A[tid];

    // rho in vec16 form, initial state
    float rho_vec_0 = rho00_init; float rho_vec_1 = rho11_init;
    float rho_vec_2 = rho22_init; float rho_vec_3 = rho33_init;
    float rho_vec_4 = 0.f;  float rho_vec_5 = 0.f;  float rho_vec_6 = 0.f;
    float rho_vec_7 = 0.f;  float rho_vec_8 = 0.f;  float rho_vec_9 = 0.f;
    float rho_vec_10 = 0.f; float rho_vec_11 = 0.f; float rho_vec_12 = 0.f;
    float rho_vec_13 = 0.f; float rho_vec_14 = 0.f; float rho_vec_15 = 0.f;


    if (tid == 0) {   // only one thread prints to avoid flooding
        printf("Thread tid=0 started. Message from thread tid=0.\n");
        printf("rho00_init = %f\n", rho_vec_0);
        printf("rho11_init = %f\n", rho_vec_1);
        printf("rho22_init = %f\n", rho_vec_2);
        printf("rho33_init = %f\n", rho_vec_3);
        printf("Npoints=%d\n", Npoints);
    }

    //float sum_whole[4] = { 0.f,0.f,0.f,0.f };

    float sum_last_0 = 0.f;
    float sum_last_1 = 0.f;
    float sum_last_2 = 0.f;
    float sum_last_3 = 0.f;

    //float sum_2last[4] = { 0.f,0.f,0.f,0.f };
    //float sum_3last[4] = { 0.f,0.f,0.f,0.f };

    //const int total_steps = N_steps_per_period * N_periods;

    for (int t_idx_rk4_step = 0; t_idx_rk4_step < N_steps_per_period * N_periods; ++t_idx_rk4_step) {
        const float t_step = t_idx_rk4_step * dt;


        /*if (tid == 0 && t_idx_rk4_step % 1000 == 0)*/
        /*if (tid == 0 && t_idx_rk4_step < 100)*/
        /*if (tid == 0)*/

        //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
        //    printf("before RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
        //}

        rk4_step_unrolled_v3_safe(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

            t_step,
            eps0,
            A
        );


        //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
        //    printf(" after RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
        //}

        // post-step stabilization
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

        //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
        //    printf(" after ren: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
        //    printf("\n");
        //}


        // accumulate populations (diagonals are rho_vec[0..3])
//#pragma unroll
        //for (int q = 0; q < 4; q++) sum_whole[q] += rho_vec[q];

        const int period_idx = t_idx_rk4_step / N_steps_per_period;
        if (period_idx >= N_periods - N_periods_avg) {

            sum_last_0 += rho_vec_0;
            sum_last_1 += rho_vec_1;
            sum_last_2 += rho_vec_2;
            sum_last_3 += rho_vec_3;

        }
        //        else if (period_idx == N_periods - 2) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
        //        }
        //        else if (period_idx == N_periods - 3) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
        //        }

    }

    // final normalization safeguard
    clamp_and_renormalize_vec16_unrolled(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
    );

    //const float inv_whole = 1.0f / float(N_steps_per_period * N_periods);
    const float inv_avg = 1.0f / float(N_steps_per_period * N_periods_avg);

    //// for avg_periods_ouput_option == "whole_last_2last_3last":
    //const size_t base = (size_t)tid * 16u;
    //// write: first 4 = whole avg, then last, 2last, 3last
    //for (int d = 0; d < 4; ++d) d_out_avg[base + d] = sum_whole[d] * inv_whole;
    //for (int d = 0; d < 4; ++d) d_out_avg[base + 4 + d] = sum_last[d] * inv_period;
    //for (int d = 0; d < 4; ++d) d_out_avg[base + 8 + d] = sum_2last[d] * inv_period;
    //for (int d = 0; d < 4; ++d) d_out_avg[base + 12 + d] = sum_3last[d] * inv_period;

    //// for avg_periods_ouput_option == "last_2last":
    //const size_t base = (size_t)tid * 8u;
    //// write: first 2 = last, 2last
    //for (int d = 0; d < 4; ++d) d_out_avg[base + d] = sum_last[d] * inv_period;
    //for (int d = 0; d < 4; ++d) d_out_avg[base + 4 + d] = sum_2last[d] * inv_period;

    // for avg_periods_ouput_option == "last":
    const size_t base = (size_t)tid * 4u;
    // write: first 1 = last
    d_out_avg[base + 0] = sum_last_0 * inv_avg;
    d_out_avg[base + 1] = sum_last_1 * inv_avg;
    d_out_avg[base + 2] = sum_last_2 * inv_avg;
    d_out_avg[base + 3] = sum_last_3 * inv_avg;

}



extern "C" __global__ void lindblad_rk4_kernel_unrolled_fsal(
    const float* __restrict__ d_eps0,
    const float* __restrict__ d_A,
    float* __restrict__ d_out_avg,

    const int N_periods_avg
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= Npoints) return;


    const float eps0 = d_eps0[tid];
    const float A = d_A[tid];

    // rho in vec16 form, initial state
    float rho_vec_0 = rho00_init; float rho_vec_1 = rho11_init;
    float rho_vec_2 = rho22_init; float rho_vec_3 = rho33_init;
    float rho_vec_4 = 0.f;  float rho_vec_5 = 0.f;  float rho_vec_6 = 0.f;
    float rho_vec_7 = 0.f;  float rho_vec_8 = 0.f;  float rho_vec_9 = 0.f;
    float rho_vec_10 = 0.f; float rho_vec_11 = 0.f; float rho_vec_12 = 0.f;
    float rho_vec_13 = 0.f; float rho_vec_14 = 0.f; float rho_vec_15 = 0.f;


    if (tid == 0) {   // only one thread prints to avoid flooding
        printf("Thread tid=0 started. Message from thread tid=0.\n");
        printf("rho00_init = %f\n", rho_vec_0);
        printf("rho11_init = %f\n", rho_vec_1);
        printf("rho22_init = %f\n", rho_vec_2);
        printf("rho33_init = %f\n", rho_vec_3);
        printf("Npoints=%d\n", Npoints);
    }

    float sum_last_0 = 0.f;
    float sum_last_1 = 0.f;
    float sum_last_2 = 0.f;
    float sum_last_3 = 0.f;
 
    // persistent between steps
    float k_saved_0 = 0.f;  float k_saved_1 = 0.f;  float k_saved_2 = 0.f;
    float k_saved_3 = 0.f;  float k_saved_4 = 0.f;  float k_saved_5 = 0.f;
    float k_saved_6 = 0.f;  float k_saved_7 = 0.f;  float k_saved_8 = 0.f;
    float k_saved_9 = 0.f;  float k_saved_10 = 0.f; float k_saved_11 = 0.f;
    float k_saved_12 = 0.f; float k_saved_13 = 0.f; float k_saved_14 = 0.f;
    float k_saved_15 = 0.f;


    // First step
    rk4_step_unrolled_v3_safe_fsal(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

        k_saved_0, k_saved_1, k_saved_2, k_saved_3,
        k_saved_4, k_saved_5, k_saved_6, k_saved_7,
        k_saved_8, k_saved_9, k_saved_10, k_saved_11,
        k_saved_12, k_saved_13, k_saved_14, k_saved_15,

        0.0f,
        eps0,
        A,
        true
    );


    // Subsequent steps
    for (int t_idx_rk4_step = 1; t_idx_rk4_step < N_steps_per_period * N_periods; ++t_idx_rk4_step) {
        const float t_step = t_idx_rk4_step * dt;

        rk4_step_unrolled_v3_safe_fsal(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

            k_saved_0, k_saved_1, k_saved_2, k_saved_3,
            k_saved_4, k_saved_5, k_saved_6, k_saved_7,
            k_saved_8, k_saved_9, k_saved_10, k_saved_11,
            k_saved_12, k_saved_13, k_saved_14, k_saved_15,

            t_step,
            eps0,
            A,
            false
        );

        // post-step stabilization
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

        const int period_idx = t_idx_rk4_step / N_steps_per_period;
        if (period_idx >= N_periods - N_periods_avg) {

            sum_last_0 += rho_vec_0;
            sum_last_1 += rho_vec_1;
            sum_last_2 += rho_vec_2;
            sum_last_3 += rho_vec_3;

        }

    }

    // final normalization safeguard
    clamp_and_renormalize_vec16_unrolled(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
    );

    //const float inv_whole = 1.0f / float(N_steps_per_period * N_periods);
    const float inv_avg = 1.0f / float(N_steps_per_period * N_periods_avg);

    const size_t base = (size_t)tid * 4u;
    // write: first 1 = last
    d_out_avg[base + 0] = sum_last_0 * inv_avg;
    d_out_avg[base + 1] = sum_last_1 * inv_avg;
    d_out_avg[base + 2] = sum_last_2 * inv_avg;
    d_out_avg[base + 3] = sum_last_3 * inv_avg;

}




extern "C" __global__ void lindblad_rk4_kernel_unrolled_ensemble_opt1(
    const float* __restrict__ d_eps0,
    const float* __restrict__ d_A,
    float* __restrict__ d_out_avg,

    const int N_periods_avg,
    const int N_samples_noise
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //if (tid == 0) { // Limit to one thread to avoid excessive output
    //    printf("\n");
    //    printf("Npoints=%d\n", Npoints);
    //    printf("\n");
    //}

    if (tid >= Npoints) return;



    const float eps0 = d_eps0[tid];
    const float A = d_A[tid];


    // Accumulators for averaged output over noise realizations
    float avg00 = 0.f;
    float avg11 = 0.f;
    float avg22 = 0.f;
    float avg33 = 0.f;


    // noise ensemble loop
    for (int j = 0; j < N_samples_noise; ++j) {

        const float eps0_noise = eps0 + c_eps_offsets[j];



        // rho in vec16 form, initial state
        float rho_vec_0 = rho00_init; float rho_vec_1 = rho11_init;
        float rho_vec_2 = rho22_init; float rho_vec_3 = rho33_init;
        float rho_vec_4 = 0.f;  float rho_vec_5 = 0.f;  float rho_vec_6 = 0.f;
        float rho_vec_7 = 0.f;  float rho_vec_8 = 0.f;  float rho_vec_9 = 0.f;
        float rho_vec_10 = 0.f; float rho_vec_11 = 0.f; float rho_vec_12 = 0.f;
        float rho_vec_13 = 0.f; float rho_vec_14 = 0.f; float rho_vec_15 = 0.f;


        //if (tid == 0) {   // only one thread prints to avoid flooding
        //    printf("Thread tid=0 started. Message from thread tid=0.\n");
        //    printf("rho00_init = %f\n", rho_vec_0);
        //    printf("rho11_init = %f\n", rho_vec_1);
        //    printf("rho22_init = %f\n", rho_vec_2);
        //    printf("rho33_init = %f\n", rho_vec_3);
        //    printf("Npoints=%d\n", Npoints);
        //}

        //float sum_whole[4] = { 0.f,0.f,0.f,0.f };

        float sum_last_0 = 0.f;
        float sum_last_1 = 0.f;
        float sum_last_2 = 0.f;
        float sum_last_3 = 0.f;

        //float sum_2last[4] = { 0.f,0.f,0.f,0.f };
        //float sum_3last[4] = { 0.f,0.f,0.f,0.f };

        //const int total_steps = N_steps_per_period * N_periods;

        for (int t_idx_rk4_step = 0; t_idx_rk4_step < N_steps_per_period * N_periods; ++t_idx_rk4_step) {
            const float t_step = t_idx_rk4_step * dt;


            /*if (tid == 0 && t_idx_rk4_step % 1000 == 0)*/
            /*if (tid == 0 && t_idx_rk4_step < 100)*/
            /*if (tid == 0)*/

            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf("before RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //}

            rk4_step_unrolled_v3_safe_ensemble(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

                t_step,
                eps0_noise,
                A
            );


            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf(" after RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //}

            // post-step stabilization
            clamp_and_renormalize_vec16_unrolled(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
            );

            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf(" after ren: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //    printf("\n");
            //}


            // accumulate populations (diagonals are rho_vec[0..3])
    //#pragma unroll
            //for (int q = 0; q < 4; q++) sum_whole[q] += rho_vec[q];

            const int period_idx = t_idx_rk4_step / N_steps_per_period;
            if (period_idx >= N_periods - N_periods_avg) {

                sum_last_0 += rho_vec_0;
                sum_last_1 += rho_vec_1;
                sum_last_2 += rho_vec_2;
                sum_last_3 += rho_vec_3;

            }
            //        else if (period_idx == N_periods - 2) {
            //#pragma unroll
            //            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
            //        }
            //        else if (period_idx == N_periods - 3) {
            //#pragma unroll
            //            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
            //        }

        }

        // final normalization safeguard
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

        //const float inv_whole = 1.0f / float(N_steps_per_period * N_periods);
        const float inv_avg = 1.0f / float(N_steps_per_period * N_periods_avg);

        avg00 += sum_last_0 * inv_avg;
        avg11 += sum_last_1 * inv_avg;
        avg22 += sum_last_2 * inv_avg;
        avg33 += sum_last_3 * inv_avg;

    }

    // average over noise realizations
    const float inv_noise = 1.0f / float(N_samples_noise);
    const size_t base = (size_t)tid * 4u;
    d_out_avg[base + 0] = avg00 * inv_noise;
    d_out_avg[base + 1] = avg11 * inv_noise;
    d_out_avg[base + 2] = avg22 * inv_noise;
    d_out_avg[base + 3] = avg33 * inv_noise;

}


extern "C" __global__ void lindblad_rk4_kernel_unrolled_ensemble_opt2(
    const float* __restrict__ d_eps0,
    const float* __restrict__ d_A,
    float* __restrict__ d_out_avg,

    const int N_periods_avg,
    const int N_samples_noise,
    const float scale_total
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //if (tid == 0) { // Limit to one thread to avoid excessive output
    //    printf("\n");
    //    printf("Npoints=%d\n", Npoints);
    //    printf("\n");
    //}

    if (tid >= Npoints) return;



    const float eps0 = d_eps0[tid];
    const float A = d_A[tid];


    // Accumulators for averaged output over noise realizations
    float avg00 = 0.f;
    float avg11 = 0.f;
    float avg22 = 0.f;
    float avg33 = 0.f;


    // noise ensemble loop
    for (int j = 0; j < N_samples_noise; ++j) {

        const float eps0_noise = eps0 + c_eps_offsets[j];



        // rho in vec16 form, initial state
        float rho_vec_0 = rho00_init; float rho_vec_1 = rho11_init;
        float rho_vec_2 = rho22_init; float rho_vec_3 = rho33_init;
        float rho_vec_4 = 0.f;  float rho_vec_5 = 0.f;  float rho_vec_6 = 0.f;
        float rho_vec_7 = 0.f;  float rho_vec_8 = 0.f;  float rho_vec_9 = 0.f;
        float rho_vec_10 = 0.f; float rho_vec_11 = 0.f; float rho_vec_12 = 0.f;
        float rho_vec_13 = 0.f; float rho_vec_14 = 0.f; float rho_vec_15 = 0.f;


        //if (tid == 0) {   // only one thread prints to avoid flooding
        //    printf("Thread tid=0 started. Message from thread tid=0.\n");
        //    printf("rho00_init = %f\n", rho_vec_0);
        //    printf("rho11_init = %f\n", rho_vec_1);
        //    printf("rho22_init = %f\n", rho_vec_2);
        //    printf("rho33_init = %f\n", rho_vec_3);
        //    printf("Npoints=%d\n", Npoints);
        //}

        //float sum_whole[4] = { 0.f,0.f,0.f,0.f };

        float sum_last_0 = 0.f;
        float sum_last_1 = 0.f;
        float sum_last_2 = 0.f;
        float sum_last_3 = 0.f;

        //float sum_2last[4] = { 0.f,0.f,0.f,0.f };
        //float sum_3last[4] = { 0.f,0.f,0.f,0.f };

        //const int total_steps = N_steps_per_period * N_periods;

        for (int t_idx_rk4_step = 0; t_idx_rk4_step < N_steps_per_period * N_periods; ++t_idx_rk4_step) {
            const float t_step = t_idx_rk4_step * dt;


            /*if (tid == 0 && t_idx_rk4_step % 1000 == 0)*/
            /*if (tid == 0 && t_idx_rk4_step < 100)*/
            /*if (tid == 0)*/

            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf("before RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //}

            rk4_step_unrolled_v3_safe_ensemble(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

                t_step,
                eps0_noise,
                A
            );


            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf(" after RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //}

            // post-step stabilization
            clamp_and_renormalize_vec16_unrolled(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
            );

            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf(" after ren: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //    printf("\n");
            //}


            // accumulate populations (diagonals are rho_vec[0..3])
    //#pragma unroll
            //for (int q = 0; q < 4; q++) sum_whole[q] += rho_vec[q];

            const int period_idx = t_idx_rk4_step / N_steps_per_period;
            if (period_idx >= N_periods - N_periods_avg) {

                sum_last_0 += rho_vec_0;
                sum_last_1 += rho_vec_1;
                sum_last_2 += rho_vec_2;
                sum_last_3 += rho_vec_3;

            }
            //        else if (period_idx == N_periods - 2) {
            //#pragma unroll
            //            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
            //        }
            //        else if (period_idx == N_periods - 3) {
            //#pragma unroll
            //            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
            //        }

        }

        // final normalization safeguard
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

        avg00 += sum_last_0;
        avg11 += sum_last_1;
        avg22 += sum_last_2;
        avg33 += sum_last_3;

    }

    // average over time steps and over noise realizations alltogether
    const size_t base = (size_t)tid * 4u;
    d_out_avg[base + 0] = avg00 * scale_total;
    d_out_avg[base + 1] = avg11 * scale_total;
    d_out_avg[base + 2] = avg22 * scale_total;
    d_out_avg[base + 3] = avg33 * scale_total;

}


extern "C" __global__ void lindblad_rk4_kernel_unrolled_ensemble_opt3(
    const float* __restrict__ d_eps0,
    const float* __restrict__ d_A,
    float* __restrict__ d_out_avg,

    const int N_periods_avg,
    const int N_samples_noise,
    const float scale_total
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //if (tid == 0) { // Limit to one thread to avoid excessive output
    //    printf("\n");
    //    printf("Npoints=%d\n", Npoints);
    //    printf("\n");
    //}

    if (tid >= Npoints) return;



    const float eps0 = d_eps0[tid];
    const float A = d_A[tid];


    // Accumulators for averaged output over noise realizations
    float avg00 = 0.f;
    float avg11 = 0.f;
    float avg22 = 0.f;
    float avg33 = 0.f;


    // noise ensemble loop
    for (int j = 0; j < N_samples_noise; ++j) {

        const float eps0_noise = eps0 + c_eps_offsets[j];

        // rho in vec16 form, initial state
        float rho_vec_0 = rho00_init; float rho_vec_1 = rho11_init;
        float rho_vec_2 = rho22_init; float rho_vec_3 = rho33_init;
        float rho_vec_4 = 0.f;  float rho_vec_5 = 0.f;  float rho_vec_6 = 0.f;
        float rho_vec_7 = 0.f;  float rho_vec_8 = 0.f;  float rho_vec_9 = 0.f;
        float rho_vec_10 = 0.f; float rho_vec_11 = 0.f; float rho_vec_12 = 0.f;
        float rho_vec_13 = 0.f; float rho_vec_14 = 0.f; float rho_vec_15 = 0.f;


        //if (tid == 0) {   // only one thread prints to avoid flooding
        //    printf("Thread tid=0 started. Message from thread tid=0.\n");
        //    printf("rho00_init = %f\n", rho_vec_0);
        //    printf("rho11_init = %f\n", rho_vec_1);
        //    printf("rho22_init = %f\n", rho_vec_2);
        //    printf("rho33_init = %f\n", rho_vec_3);
        //    printf("Npoints=%d\n", Npoints);
        //    printf("c_eps_offsets[j] = %f\n", c_eps_offsets[j]);
        //}


        //const int total_steps = N_steps_per_period * N_periods;

        for (int t_idx_rk4_step = 0; t_idx_rk4_step < N_steps_per_period * N_periods; ++t_idx_rk4_step) {
            const float t_step = t_idx_rk4_step * dt;


            /*if (tid == 0 && t_idx_rk4_step % 1000 == 0)*/
            /*if (tid == 0 && t_idx_rk4_step < 100)*/
            /*if (tid == 0)*/

            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf("before RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //}

            rk4_step_unrolled_v3_safe_ensemble(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

                t_step,
                eps0_noise,
                A
            );


            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf(" after RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //}

            // post-step stabilization
            clamp_and_renormalize_vec16_unrolled(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
            );

            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf(" after ren: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //    printf("\n");
            //}


            // accumulate populations (diagonals are rho_vec[0..3])
    //#pragma unroll
            //for (int q = 0; q < 4; q++) sum_whole[q] += rho_vec[q];

            const int period_idx = t_idx_rk4_step / N_steps_per_period;
            if (period_idx >= N_periods - N_periods_avg) {

                avg00 += rho_vec_0;
                avg11 += rho_vec_1;
                avg22 += rho_vec_2;
                avg33 += rho_vec_3;

            }
            //        else if (period_idx == N_periods - 2) {
            //#pragma unroll
            //            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
            //        }
            //        else if (period_idx == N_periods - 3) {
            //#pragma unroll
            //            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
            //        }

        }

        // final normalization safeguard
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

    }

    // average over time steps and over noise realizations alltogether
    const size_t base = (size_t)tid * 4u;
    d_out_avg[base + 0] = avg00 * scale_total;
    d_out_avg[base + 1] = avg11 * scale_total;
    d_out_avg[base + 2] = avg22 * scale_total;
    d_out_avg[base + 3] = avg33 * scale_total;

}




extern "C" __global__ void lindblad_rk4_kernel_unrolled_ensemble_opt10(
    const float* __restrict__ d_eps0,
    const float* __restrict__ d_A,
    float* __restrict__ d_out_avg,

    const int N_periods_avg,
    const int N_samples_noise,
    const float scale_total,
    const float* __restrict__ eps_offsets
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //if (tid == 0) { // Limit to one thread to avoid excessive output
    //    printf("\n");
    //    printf("Npoints=%d\n", Npoints);
    //    printf("\n");
    //}

    if (tid >= Npoints) return;



    const float eps0 = d_eps0[tid];
    const float A = d_A[tid];


    // Accumulators for averaged output over noise realizations
    float avg00 = 0.f;
    float avg11 = 0.f;
    float avg22 = 0.f;
    float avg33 = 0.f;


    // noise ensemble loop
    for (int j = 0; j < N_samples_noise; ++j) {

        const float eps0_noise = eps0 + eps_offsets[j];



        // rho in vec16 form, initial state
        float rho_vec_0 = rho00_init; float rho_vec_1 = rho11_init;
        float rho_vec_2 = rho22_init; float rho_vec_3 = rho33_init;
        float rho_vec_4 = 0.f;  float rho_vec_5 = 0.f;  float rho_vec_6 = 0.f;
        float rho_vec_7 = 0.f;  float rho_vec_8 = 0.f;  float rho_vec_9 = 0.f;
        float rho_vec_10 = 0.f; float rho_vec_11 = 0.f; float rho_vec_12 = 0.f;
        float rho_vec_13 = 0.f; float rho_vec_14 = 0.f; float rho_vec_15 = 0.f;


        //if (tid == 0) {   // only one thread prints to avoid flooding
        //    printf("Thread tid=0 started. Message from thread tid=0.\n");
        //    printf("rho00_init = %f\n", rho_vec_0);
        //    printf("rho11_init = %f\n", rho_vec_1);
        //    printf("rho22_init = %f\n", rho_vec_2);
        //    printf("rho33_init = %f\n", rho_vec_3);
        //    printf("Npoints=%d\n", Npoints);
        //}


        //const int total_steps = N_steps_per_period * N_periods;

        for (int t_idx_rk4_step = 0; t_idx_rk4_step < N_steps_per_period * N_periods; ++t_idx_rk4_step) {
            const float t_step = t_idx_rk4_step * dt;


            /*if (tid == 0 && t_idx_rk4_step % 1000 == 0)*/
            /*if (tid == 0 && t_idx_rk4_step < 100)*/
            /*if (tid == 0)*/

            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf("before RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //}

            rk4_step_unrolled_v3_safe_ensemble(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

                t_step,
                eps0_noise,
                A
            );


            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf(" after RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //}

            // post-step stabilization
            clamp_and_renormalize_vec16_unrolled(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
            );

            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf(" after ren: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //    printf("\n");
            //}


            // accumulate populations (diagonals are rho_vec[0..3])
    //#pragma unroll
            //for (int q = 0; q < 4; q++) sum_whole[q] += rho_vec[q];

            const int period_idx = t_idx_rk4_step / N_steps_per_period;
            if (period_idx >= N_periods - N_periods_avg) {

                avg00 += rho_vec_0;
                avg11 += rho_vec_1;
                avg22 += rho_vec_2;
                avg33 += rho_vec_3;

            }
            //        else if (period_idx == N_periods - 2) {
            //#pragma unroll
            //            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
            //        }
            //        else if (period_idx == N_periods - 3) {
            //#pragma unroll
            //            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
            //        }

        }

        // final normalization safeguard
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

    }

    // average over time steps and over noise realizations alltogether
    const size_t base = (size_t)tid * 4u;
    d_out_avg[base + 0] = avg00 * scale_total;
    d_out_avg[base + 1] = avg11 * scale_total;
    d_out_avg[base + 2] = avg22 * scale_total;
    d_out_avg[base + 3] = avg33 * scale_total;

}





extern "C" __global__ void lindblad_rk4_kernel_unrolled_ensemble_opt10_sequential_fsal(
    const float* __restrict__ d_eps0,
    const float* __restrict__ d_A,
    float* __restrict__ d_out_avg,

    const int N_periods_avg,
    const int N_samples_noise,
    const float scale_total,
    const float* __restrict__ eps_offsets
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= Npoints) return;



    const float eps0 = d_eps0[tid];
    const float A = d_A[tid];


    // Accumulators for averaged output over noise realizations
    float avg00 = 0.f;
    float avg11 = 0.f;
    float avg22 = 0.f;
    float avg33 = 0.f;


    // noise ensemble loop
    for (int j = 0; j < N_samples_noise; ++j) {

        const float eps0_noise = eps0 + eps_offsets[j];



        // rho in vec16 form, initial state
        float rho_vec_0 = rho00_init; float rho_vec_1 = rho11_init;
        float rho_vec_2 = rho22_init; float rho_vec_3 = rho33_init;
        float rho_vec_4 = 0.f;  float rho_vec_5 = 0.f;  float rho_vec_6 = 0.f;
        float rho_vec_7 = 0.f;  float rho_vec_8 = 0.f;  float rho_vec_9 = 0.f;
        float rho_vec_10 = 0.f; float rho_vec_11 = 0.f; float rho_vec_12 = 0.f;
        float rho_vec_13 = 0.f; float rho_vec_14 = 0.f; float rho_vec_15 = 0.f;



        // persistent between steps
        float k_saved_0 = 0.f;  float k_saved_1 = 0.f;  float k_saved_2 = 0.f;
        float k_saved_3 = 0.f;  float k_saved_4 = 0.f;  float k_saved_5 = 0.f;
        float k_saved_6 = 0.f;  float k_saved_7 = 0.f;  float k_saved_8 = 0.f;
        float k_saved_9 = 0.f;  float k_saved_10 = 0.f; float k_saved_11 = 0.f;
        float k_saved_12 = 0.f; float k_saved_13 = 0.f; float k_saved_14 = 0.f;
        float k_saved_15 = 0.f;

        // First step
        rk4_step_unrolled_v3_safe_ensemble_fsal(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

            k_saved_0, k_saved_1, k_saved_2, k_saved_3,
            k_saved_4, k_saved_5, k_saved_6, k_saved_7,
            k_saved_8, k_saved_9, k_saved_10, k_saved_11,
            k_saved_12, k_saved_13, k_saved_14, k_saved_15,

            0.0f,
            eps0_noise,
            A,
            true
        );


        // Subsequent steps
        for (int t_idx_rk4_step = 1; t_idx_rk4_step < N_steps_per_period * N_periods; ++t_idx_rk4_step) {
            const float t_step = t_idx_rk4_step * dt;

            rk4_step_unrolled_v3_safe_ensemble_fsal(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

                k_saved_0, k_saved_1, k_saved_2, k_saved_3,
                k_saved_4, k_saved_5, k_saved_6, k_saved_7,
                k_saved_8, k_saved_9, k_saved_10, k_saved_11,
                k_saved_12, k_saved_13, k_saved_14, k_saved_15,

                t_step,
                eps0_noise,
                A,
                false
            );


            // post-step stabilization
            clamp_and_renormalize_vec16_unrolled(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
            );

            const int period_idx = t_idx_rk4_step / N_steps_per_period;
            if (period_idx >= N_periods - N_periods_avg) {

                avg00 += rho_vec_0;
                avg11 += rho_vec_1;
                avg22 += rho_vec_2;
                avg33 += rho_vec_3;

            }

        }

        // final normalization safeguard
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

    }

    // average over time steps and over noise realizations alltogether
    const size_t base = (size_t)tid * 4u;
    d_out_avg[base + 0] = avg00 * scale_total;
    d_out_avg[base + 1] = avg11 * scale_total;
    d_out_avg[base + 2] = avg22 * scale_total;
    d_out_avg[base + 3] = avg33 * scale_total;

}




extern "C" __global__ void lindblad_rk4_kernel_unrolled_ensemble_opt10_parallel_fsal(
    const float* __restrict__ d_eps0,
    const float* __restrict__ d_A,
    float* __restrict__ d_out_avg,

    const int N_periods_avg,
    const int N_samples_noise,
    const float scale_total,
    const float* __restrict__ eps_offsets,
    const int total_trajectories
) {
    // Total number of parallel trajectories = Npoints * N_samples_noise
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_trajectories) return;

    // Decompose thread ID into point index and noise sample index
    const int point_idx = tid / N_samples_noise;  // which (eps0, A) pair
    const int noise_idx = tid % N_samples_noise;  // which noise realization

    // Load parameters for this point
    // const float eps0 = d_eps0[point_idx];
    const float A = d_A[point_idx];
    
    // Apply noise offset for this realization
    const float eps0_noise = d_eps0[point_idx] + eps_offsets[noise_idx];

    // Initialize density matrix (vec16 representation)
    float rho_vec_0 = rho00_init; 
    float rho_vec_1 = rho11_init;
    float rho_vec_2 = rho22_init; 
    float rho_vec_3 = rho33_init;
    float rho_vec_4 = 0.f;  float rho_vec_5 = 0.f;  
    float rho_vec_6 = 0.f;  float rho_vec_7 = 0.f;  
    float rho_vec_8 = 0.f;  float rho_vec_9 = 0.f;
    float rho_vec_10 = 0.f; float rho_vec_11 = 0.f; 
    float rho_vec_12 = 0.f; float rho_vec_13 = 0.f; 
    float rho_vec_14 = 0.f; float rho_vec_15 = 0.f;

    // Accumulators for time averaging (over last N_periods_avg periods)
    float sum_avg_0 = 0.f;
    float sum_avg_1 = 0.f;
    float sum_avg_2 = 0.f;
    float sum_avg_3 = 0.f;

    // FSAL: saved derivatives from previous step
    float k_saved_0 = 0.f;  float k_saved_1 = 0.f;  
    float k_saved_2 = 0.f;  float k_saved_3 = 0.f;  
    float k_saved_4 = 0.f;  float k_saved_5 = 0.f;
    float k_saved_6 = 0.f;  float k_saved_7 = 0.f;  
    float k_saved_8 = 0.f;  float k_saved_9 = 0.f;  
    float k_saved_10 = 0.f; float k_saved_11 = 0.f;
    float k_saved_12 = 0.f; float k_saved_13 = 0.f; 
    float k_saved_14 = 0.f; float k_saved_15 = 0.f;

    // First RK4 step (t=0, is_first_step=true)
    rk4_step_unrolled_v3_safe_ensemble_fsal(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

        k_saved_0, k_saved_1, k_saved_2, k_saved_3,
        k_saved_4, k_saved_5, k_saved_6, k_saved_7,
        k_saved_8, k_saved_9, k_saved_10, k_saved_11,
        k_saved_12, k_saved_13, k_saved_14, k_saved_15,

        0.0f,
        eps0_noise,
        A,
        true  // is_first_step
    );

    // Main time integration loop
   
    for (int t_idx_rk4_step = 1; t_idx_rk4_step < N_steps_per_period * N_periods; ++t_idx_rk4_step) {
        const float t_step = t_idx_rk4_step * dt;

        // RK4 step
        rk4_step_unrolled_v3_safe_ensemble_fsal(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

            k_saved_0, k_saved_1, k_saved_2, k_saved_3,
            k_saved_4, k_saved_5, k_saved_6, k_saved_7,
            k_saved_8, k_saved_9, k_saved_10, k_saved_11,
            k_saved_12, k_saved_13, k_saved_14, k_saved_15,

            t_step,
            eps0_noise,
            A,
            false  // is_first_step
        );

        // Post-step stabilization (enforce physical constraints)
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

        // Accumulate populations if we're in the averaging window
        const int period_idx = t_idx_rk4_step / N_steps_per_period;
        if (period_idx >= N_periods - N_periods_avg) {
            sum_avg_0 += rho_vec_0;
            sum_avg_1 += rho_vec_1;
            sum_avg_2 += rho_vec_2;
            sum_avg_3 += rho_vec_3;
        }
    }

    // Final normalization safeguard
    clamp_and_renormalize_vec16_unrolled(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
    );

    // ========================================================================
    // ATOMIC ADDITION: Average over noise ensemble for this (eps0, A) point
    // ========================================================================
    // Each of the N_samples_noise threads for this point_idx will atomically
    // add their contribution. The scale_total already includes division by
    // (N_steps_per_period * N_periods_avg * N_samples_noise)
    
    const size_t base = (size_t)point_idx * 4u;
    
    atomicAdd(&d_out_avg[base + 0], sum_avg_0 * scale_total);
    atomicAdd(&d_out_avg[base + 1], sum_avg_1 * scale_total);
    atomicAdd(&d_out_avg[base + 2], sum_avg_2 * scale_total);
    atomicAdd(&d_out_avg[base + 3], sum_avg_3 * scale_total);
}


// ============================================================================
// HOST LAUNCH CODE
// ============================================================================

// Add this to your host_branch_grid.cuh file, in the run_grid_mode function
// where you launch the ensemble kernel:

/*

    // For parallel ensemble kernel:
    if (quasi_static_ensemble_dephasing_flag) {
        
        // IMPORTANT: Zero-initialize output buffer before atomic operations
        gpuCheck(cudaMemset(d_rho_avg, 0, host_N_points * rho_avg_dim * sizeof(float)), 
                 "cudaMemset d_rho_avg");

        // Calculate total number of parallel trajectories
        const int total_trajectories = host_N_points * N_samples_noise;
        
        // Thread block configuration
        threads_per_block = 128;  // or 256, tune for your GPU
        blocks = (total_trajectories + threads_per_block - 1) / threads_per_block;

        printf("Launching PARALLEL ensemble kernel: blocks=%d threads_per_block=%d\n", 
               blocks, threads_per_block);
        printf("  Total parallel trajectories: %d (Npoints=%d × N_samples=%d)\n", 
               total_trajectories, host_N_points, N_samples_noise);
        fflush(stdout);

        const float scale_total = 1.0f / (host_N_steps_per_period * N_periods_avg * N_samples_noise);
        
        lindblad_rk4_kernel_unrolled_ensemble_opt10_parallel_fsal<<<blocks, threads_per_block>>>(
            d_eps0, 
            d_A,
            d_rho_avg,
            N_periods_avg,
            N_samples_noise,
            scale_total,
            eps_offsets
        );

        // Check for launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Parallel ensemble kernel launch failed: %s\n", 
                    cudaGetErrorString(err));
            std::abort();
        }
    }

*/




/*
extern "C" __global__ void lindblad_rk4_kernel_unrolled_warp_shfl(
    const float* __restrict__ d_eps0,
    const float* __restrict__ d_A,
    float* __restrict__ d_out_avg
) {

    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index within the entire grid
    const int warp_id = thread_id / WARP_SIZE;                   // (WARP_SIZE = 32)
    const int lane_id = thread_id % WARP_SIZE;                   // Thread's position (lane) within its warp, [0..31]

    // Each warp handles K_TRAJ_PER_WARP trajectories, divided into groups of lanes
    // LANE_GROUP_SIZE - Number of lanes per trajectory group. E.g. 16 if K_TRAJ_PER_WARP = 2
    const int group_id = lane_id / LANE_GROUP_SIZE;             // Which trajectory group (among K_TRAJ_PER_WARP per warp) this lane belongs to
    const int lane_in_group = lane_id % LANE_GROUP_SIZE;        // Lane's index inside its group (which part of trajectory work it does)

    // Calculate the global trajectory ID handled by this thread
    // warp_id * K_TRAJ_PER_WARP gives the base trajectory index for the warp
    // group_id adds the offset for the trajectory within the warp
    const int traj_id = warp_id * K_TRAJ_PER_WARP + group_id;

    if (traj_id >= Npoints) return;


    const int group_base_lane = group_id * LANE_GROUP_SIZE;  // the first lane of the group






    bool is_main_lane = (lane_in_group == 0);

    if (is_main_lane) {

        const float eps0 = d_eps0[traj_id];
        const float A = d_A[traj_id];




        //if (tid == 0) { // Limit to one thread to avoid excessive output
        //    printf("\n");
        //    printf("Npoints=%d\n", Npoints);
        //    printf("\n");
        //}


        // rho in vec16 form, initial state
        float rho_vec_0 = rho00_init;
        float rho_vec_1 = rho11_init;
        float rho_vec_2 = rho22_init;
        float rho_vec_3 = rho33_init;
        float rho_vec_4 = 0.f;
        float rho_vec_5 = 0.f;
        float rho_vec_6 = 0.f;
        float rho_vec_7 = 0.f;
        float rho_vec_8 = 0.f;
        float rho_vec_9 = 0.f;
        float rho_vec_10 = 0.f;
        float rho_vec_11 = 0.f;
        float rho_vec_12 = 0.f;
        float rho_vec_13 = 0.f;
        float rho_vec_14 = 0.f;
        float rho_vec_15 = 0.f;



        if (thread_id == 0) {   // only one thread prints to avoid flooding
            printf("Thread tid=0 started. Message from thread tid=0.\n");
            printf("rho00_init = %f\n", rho_vec_0);
            printf("rho11_init = %f\n", rho_vec_1);
            printf("rho22_init = %f\n", rho_vec_2);
            printf("rho33_init = %f\n", rho_vec_3);
            printf("Npoints=%d\n", Npoints);
        }

        //float sum_whole[4] = { 0.f,0.f,0.f,0.f };

        float sum_last_0 = 0.f;
        float sum_last_1 = 0.f;
        float sum_last_2 = 0.f;
        float sum_last_3 = 0.f;

        //float sum_2last[4] = { 0.f,0.f,0.f,0.f };
        //float sum_3last[4] = { 0.f,0.f,0.f,0.f };

        //const int total_steps = N_steps_per_period * N_periods;

        for (int t_idx_rk4_step = 0; t_idx_rk4_step < N_steps_per_period * N_periods; ++t_idx_rk4_step) {
            const float t_step = t_idx_rk4_step * dt;


            //if (tid == 0 && t_idx_rk4_step % 1000 == 0)
            //if (tid == 0 && t_idx_rk4_step < 100)
            //if (tid == 0)

            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf("before RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //}

            rk4_step_unrolled_warp_shfl_v3_safe(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

                t_step,
                eps0,
                A,

                lane_in_group,
                group_base_lane
            );


            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf(" after RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //}

            // post-step stabilization

            clamp_and_renormalize_vec16_unrolled(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
            );

            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf(" after ren: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //    printf("\n");
            //}


            // accumulate populations (diagonals are rho_vec[0..3])
    //#pragma unroll
            //for (int q = 0; q < 4; q++) sum_whole[q] += rho_vec[q];

            const int period_idx = t_idx_rk4_step / N_steps_per_period;
            if (period_idx == N_periods - 1) {

                sum_last_0 += rho_vec_0;
                sum_last_1 += rho_vec_1;
                sum_last_2 += rho_vec_2;
                sum_last_3 += rho_vec_3;

            }
            //        else if (period_idx == N_periods - 2) {
            //#pragma unroll
            //            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
            //        }
            //        else if (period_idx == N_periods - 3) {
            //#pragma unroll
            //            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
            //        }

        }

        // final normalization safeguard
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

        //const float inv_whole = 1.0f / float(N_steps_per_period * N_periods);
        const float inv_period = 1.0f / float(N_steps_per_period);

        //// for avg_periods_ouput_option == "whole_last_2last_3last":
        //const size_t base = (size_t)tid * 16u;
        //// write: first 4 = whole avg, then last, 2last, 3last
        //for (int d = 0; d < 4; ++d) d_out_avg[base + d] = sum_whole[d] * inv_whole;
        //for (int d = 0; d < 4; ++d) d_out_avg[base + 4 + d] = sum_last[d] * inv_period;
        //for (int d = 0; d < 4; ++d) d_out_avg[base + 8 + d] = sum_2last[d] * inv_period;
        //for (int d = 0; d < 4; ++d) d_out_avg[base + 12 + d] = sum_3last[d] * inv_period;

        //// for avg_periods_ouput_option == "last_2last":
        //const size_t base = (size_t)tid * 8u;
        //// write: first 2 = last, 2last
        //for (int d = 0; d < 4; ++d) d_out_avg[base + d] = sum_last[d] * inv_period;
        //for (int d = 0; d < 4; ++d) d_out_avg[base + 4 + d] = sum_2last[d] * inv_period;

        // for avg_periods_ouput_option == "last":
        const size_t base = size_t(warp_id) * 4;
        // write: first 1 = last
        d_out_avg[base + 0] = sum_last_0 * inv_period;
        d_out_avg[base + 1] = sum_last_1 * inv_period;
        d_out_avg[base + 2] = sum_last_2 * inv_period;
        d_out_avg[base + 3] = sum_last_3 * inv_period;


    }

}
*/


/*
extern "C" __global__ void lindblad_rk4_kernel_unrolled_warp_shmem(
    const float* __restrict__ d_eps0,
    const float* __restrict__ d_A,
    float* __restrict__ d_out_avg
) {

    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index within the entire grid
    const int warp_id = thread_id / WARP_SIZE;                   // (WARP_SIZE = 32)
    const int lane_id = thread_id % WARP_SIZE;                   // Thread's position (lane) within its warp, [0..31]

    // Each warp handles K_TRAJ_PER_WARP trajectories, divided into groups of lanes
    // LANE_GROUP_SIZE - Number of lanes per trajectory group. E.g. 16 if K_TRAJ_PER_WARP = 2
    const int group_id = lane_id / LANE_GROUP_SIZE;             // Which trajectory group (among K_TRAJ_PER_WARP per warp) this lane belongs to
    const int lane_in_group = lane_id % LANE_GROUP_SIZE;        // Lane's index inside its group (which part of trajectory work it does)

    // Calculate the global trajectory ID handled by this thread
    // warp_id * K_TRAJ_PER_WARP gives the base trajectory index for the warp
    // group_id adds the offset for the trajectory within the warp
    const int traj_id = warp_id * K_TRAJ_PER_WARP + group_id;

    if (traj_id >= Npoints) return;

    extern __shared__ float shm[];  // dynamically allocated shared memory





    bool is_main_lane = (lane_in_group == 0);

    if (is_main_lane) {

        const float eps0 = d_eps0[traj_id];
        const float A = d_A[traj_id];




        //if (tid == 0) { // Limit to one thread to avoid excessive output
        //    printf("\n");
        //    printf("Npoints=%d\n", Npoints);
        //    printf("\n");
        //}


        // rho in vec16 form, initial state
        float rho_vec_0 = rho00_init;
        float rho_vec_1 = rho11_init;
        float rho_vec_2 = rho22_init;
        float rho_vec_3 = rho33_init;
        float rho_vec_4 = 0.f;
        float rho_vec_5 = 0.f;
        float rho_vec_6 = 0.f;
        float rho_vec_7 = 0.f;
        float rho_vec_8 = 0.f;
        float rho_vec_9 = 0.f;
        float rho_vec_10 = 0.f;
        float rho_vec_11 = 0.f;
        float rho_vec_12 = 0.f;
        float rho_vec_13 = 0.f;
        float rho_vec_14 = 0.f;
        float rho_vec_15 = 0.f;



        if (thread_id == 0) {   // only one thread prints to avoid flooding
            printf("Thread tid=0 started. Message from thread tid=0.\n");
            printf("rho00_init = %f\n", rho_vec_0);
            printf("rho11_init = %f\n", rho_vec_1);
            printf("rho22_init = %f\n", rho_vec_2);
            printf("rho33_init = %f\n", rho_vec_3);
            printf("Npoints=%d\n", Npoints);
        }

        //float sum_whole[4] = { 0.f,0.f,0.f,0.f };

        float sum_last_0 = 0.f;
        float sum_last_1 = 0.f;
        float sum_last_2 = 0.f;
        float sum_last_3 = 0.f;

        //float sum_2last[4] = { 0.f,0.f,0.f,0.f };
        //float sum_3last[4] = { 0.f,0.f,0.f,0.f };

        //const int total_steps = N_steps_per_period * N_periods;

        for (int t_idx_rk4_step = 0; t_idx_rk4_step < N_steps_per_period * N_periods; ++t_idx_rk4_step) {
            const float t_step = t_idx_rk4_step * dt;


            //if (tid == 0 && t_idx_rk4_step % 1000 == 0)
            //if (tid == 0 && t_idx_rk4_step < 100)
            //if (tid == 0)

            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf("before RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //}

            rk4_step_unrolled_warp_shmem_v3_safe(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

                t_step,
                eps0,
                A,

                shm
            );


            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf(" after RK4: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //}

            // post-step stabilization
            clamp_and_renormalize_vec16_unrolled(
                rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
                rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
                rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
                rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
            );

            //if (tid == 0 && t_idx_rk4_step % 1000 == 0) {
            //    printf(" after ren: t_idx_rk4_step=%d, rho_vec[0..3] = %.7f, %.7f, %.7f, %.7f\n", t_idx_rk4_step, rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
            //    printf("\n");
            //}


            // accumulate populations (diagonals are rho_vec[0..3])
    //#pragma unroll
            //for (int q = 0; q < 4; q++) sum_whole[q] += rho_vec[q];

            const int period_idx = t_idx_rk4_step / N_steps_per_period;
            if (period_idx == N_periods - 1) {

                sum_last_0 += rho_vec_0;
                sum_last_1 += rho_vec_1;
                sum_last_2 += rho_vec_2;
                sum_last_3 += rho_vec_3;

            }
            //        else if (period_idx == N_periods - 2) {
            //#pragma unroll
            //            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
            //        }
            //        else if (period_idx == N_periods - 3) {
            //#pragma unroll
            //            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
            //        }

        }

        // final normalization safeguard
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

        //const float inv_whole = 1.0f / float(N_steps_per_period * N_periods);
        const float inv_period = 1.0f / float(N_steps_per_period);

        //// for avg_periods_ouput_option == "whole_last_2last_3last":
        //const size_t base = (size_t)tid * 16u;
        //// write: first 4 = whole avg, then last, 2last, 3last
        //for (int d = 0; d < 4; ++d) d_out_avg[base + d] = sum_whole[d] * inv_whole;
        //for (int d = 0; d < 4; ++d) d_out_avg[base + 4 + d] = sum_last[d] * inv_period;
        //for (int d = 0; d < 4; ++d) d_out_avg[base + 8 + d] = sum_2last[d] * inv_period;
        //for (int d = 0; d < 4; ++d) d_out_avg[base + 12 + d] = sum_3last[d] * inv_period;

        //// for avg_periods_ouput_option == "last_2last":
        //const size_t base = (size_t)tid * 8u;
        //// write: first 2 = last, 2last
        //for (int d = 0; d < 4; ++d) d_out_avg[base + d] = sum_last[d] * inv_period;
        //for (int d = 0; d < 4; ++d) d_out_avg[base + 4 + d] = sum_2last[d] * inv_period;

        // for avg_periods_ouput_option == "last":
        const size_t base = size_t(warp_id) * 4;
        // write: first 1 = last
        d_out_avg[base + 0] = sum_last_0 * inv_period;
        d_out_avg[base + 1] = sum_last_1 * inv_period;
        d_out_avg[base + 2] = sum_last_2 * inv_period;
        d_out_avg[base + 3] = sum_last_3 * inv_period;


    }

}
*/