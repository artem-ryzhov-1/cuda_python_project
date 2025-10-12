// src/kernels/lindblad_kernel_single.cuh

#pragma once

#include <cuda_runtime.h>
//#include <math.h>
//#include "cuda_intellisense_fixes.cuh"
#include "rk4/rk4_step.cuh"    // declares rk4_step_vec16_full
#include "lindblad_helpers.cuh"// inline clamp_and_renormalize_vec16
#include "constants.cuh"

/* good. only for faster compilation
extern "C" __global__ void lindblad_rk4_kernel_singlemode(

    // the single chosen point:
    float eps0_single_target,
    float A_single_target,

    float* __restrict__ d_out_avg,
    // output pho dynamics: length = total_steps * 4 (rho00..rho33)
    float* __restrict__ d_rho_dynamics,
    // time output
    float* __restrict__ d_time_dynamics,
    // epsilon (t) output
    float* __restrict__ d_eps_dynamics

) {
    // single thread only
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // rho in vec16 form, initial state
    float rho_vec[16];
#pragma unroll
    for (int i = 0; i < 16; i++) rho_vec[i] = 0.f;
    rho_vec[0] = rho00_init;
    rho_vec[1] = rho11_init;
    rho_vec[2] = rho22_init;
    rho_vec[3] = rho33_init;

    float sum_whole[4] = { 0.f,0.f,0.f,0.f };
    float sum_last[4] = { 0.f,0.f,0.f,0.f };
    float sum_2last[4] = { 0.f,0.f,0.f,0.f };
    float sum_3last[4] = { 0.f,0.f,0.f,0.f };

    const int total_steps = N_steps_per_period * N_periods;


    for (int t_idx = 0; t_idx < total_steps; ++t_idx) {
        const float t_step = t_idx * dt;
        const float eps_t_step = eps0_single_target + A_single_target * cosf(omega * t_step);


        rk4_step(
            rho_vec, t_step,
            eps0_single_target, A_single_target
        );

        // post-step stabilization
        clamp_and_renormalize_vec16(rho_vec);

        // accumulate populations (diagonals are rho_vec[0..3])
#pragma unroll
        for (int q = 0; q < 4; q++) sum_whole[q] += rho_vec[q];

        const int period_idx = t_idx / N_steps_per_period;
        if (period_idx == N_periods - 1) {
#pragma unroll
            for (int q = 0; q < 4; q++) sum_last[q] += rho_vec[q];
        }
        else if (period_idx == N_periods - 2) {
#pragma unroll
            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
        }
        else if (period_idx == N_periods - 3) {
#pragma unroll
            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
        }

        // store populations (diagonals are rho_vec[0..3])
        const int base = t_idx * 4;
        d_rho_dynamics[base + 0] = rho_vec[0];
        d_rho_dynamics[base + 1] = rho_vec[1];
        d_rho_dynamics[base + 2] = rho_vec[2];
        d_rho_dynamics[base + 3] = rho_vec[3];

        d_time_dynamics[t_idx] = t_step;
        d_eps_dynamics[t_idx] = eps_t_step;

    }

    // final normalization safeguard
    clamp_and_renormalize_vec16(rho_vec);

    const float inv_whole = 1.0f / float(total_steps);
    const float inv_period = 1.0f / float(N_steps_per_period);

    // write: first 4 = whole avg, then last, 2last, 3last
    for (int d = 0; d < 4; ++d) d_out_avg[d] = sum_whole[d] * inv_whole;
    for (int d = 0; d < 4; ++d) d_out_avg[4 + d] = sum_last[d] * inv_period;
    for (int d = 0; d < 4; ++d) d_out_avg[8 + d] = sum_2last[d] * inv_period;
    for (int d = 0; d < 4; ++d) d_out_avg[12 + d] = sum_3last[d] * inv_period;
}
*/

/* good. only for faster compilation
extern "C" __global__ void lindblad_rk4_kernel_singlemode_log(

    // the single chosen point:
    float eps0_single_target,
    float A_single_target,

    float* __restrict__ d_out_avg,
    // output pho dynamics: length = total_steps * 4 (rho00..rho33)
    float* __restrict__ d_rho_dynamics,
    // time output
    float* __restrict__ d_time_dynamics,
    // epsilon (t) output
    float* __restrict__ d_eps_dynamics,

    // log
    LogEntry* __restrict__ d_log_buffer

) {
    // single thread only
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // rho in vec16 form, initial state
    float rho_vec[16];
#pragma unroll
    for (int i = 0; i < 16; i++) rho_vec[i] = 0.f;
    rho_vec[0] = rho00_init;
    rho_vec[1] = rho11_init;
    rho_vec[2] = rho22_init;
    rho_vec[3] = rho33_init;

    float sum_whole[4] = { 0.f,0.f,0.f,0.f };
    float sum_last[4] = { 0.f,0.f,0.f,0.f };
    float sum_2last[4] = { 0.f,0.f,0.f,0.f };
    float sum_3last[4] = { 0.f,0.f,0.f,0.f };

    const int total_steps = N_steps_per_period * N_periods;


    for (int t_idx = 0; t_idx < total_steps; ++t_idx) {
        const float t_step = t_idx * dt;
        const float eps_t_step = eps0_single_target + A_single_target * cosf(omega * t_step);


        rk4_step_log(
            rho_vec, t_step,
            eps0_single_target, A_single_target,

            d_log_buffer, t_idx
        );

        // post-step stabilization
        clamp_and_renormalize_vec16(rho_vec);

        // accumulate populations (diagonals are rho_vec[0..3])
#pragma unroll
        for (int q = 0; q < 4; q++) sum_whole[q] += rho_vec[q];

        const int period_idx = t_idx / N_steps_per_period;
        if (period_idx == N_periods - 1) {
#pragma unroll
            for (int q = 0; q < 4; q++) sum_last[q] += rho_vec[q];
        }
        else if (period_idx == N_periods - 2) {
#pragma unroll
            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
        }
        else if (period_idx == N_periods - 3) {
#pragma unroll
            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
        }

        // store populations (diagonals are rho_vec[0..3])
        const int base = t_idx * 4;
        d_rho_dynamics[base + 0] = rho_vec[0];
        d_rho_dynamics[base + 1] = rho_vec[1];
        d_rho_dynamics[base + 2] = rho_vec[2];
        d_rho_dynamics[base + 3] = rho_vec[3];

        d_time_dynamics[t_idx] = t_step;
        d_eps_dynamics[t_idx] = eps_t_step;

    }

    // final normalization safeguard
    clamp_and_renormalize_vec16(rho_vec);

    const float inv_whole = 1.0f / float(total_steps);
    const float inv_period = 1.0f / float(N_steps_per_period);

    // write: first 4 = whole avg, then last, 2last, 3last
    for (int d = 0; d < 4; ++d) d_out_avg[d] = sum_whole[d] * inv_whole;
    for (int d = 0; d < 4; ++d) d_out_avg[4 + d] = sum_last[d] * inv_period;
    for (int d = 0; d < 4; ++d) d_out_avg[8 + d] = sum_2last[d] * inv_period;
    for (int d = 0; d < 4; ++d) d_out_avg[12 + d] = sum_3last[d] * inv_period;
}
*/

//////////////////////////////////////////////////////////////




extern "C" __global__ void lindblad_rk4_kernel_singlemode_unrolled(

    // the single chosen point:
    float eps0_single_target,
    float A_single_target,

    float* __restrict__ d_out_avg,
    // output pho dynamics: length = total_steps * 4 (rho00..rho33)
    float* __restrict__ d_rho_dynamics,
    // time output
    float* __restrict__ d_time_dynamics,
    // epsilon (t) output
    float* __restrict__ d_eps_dynamics

) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // single thread only
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // rho in vec16 form, initial state
    float rho_vec_0;
    float rho_vec_1;
    float rho_vec_2;
    float rho_vec_3;
    float rho_vec_4;
    float rho_vec_5;
    float rho_vec_6;
    float rho_vec_7;
    float rho_vec_8;
    float rho_vec_9;
    float rho_vec_10;
    float rho_vec_11;
    float rho_vec_12;
    float rho_vec_13;
    float rho_vec_14;
    float rho_vec_15;

    rho_vec_0 = 0.f;
    rho_vec_1 = 0.f;
    rho_vec_2 = 0.f;
    rho_vec_3 = 0.f;
    rho_vec_4 = 0.f;
    rho_vec_5 = 0.f;
    rho_vec_6 = 0.f;
    rho_vec_7 = 0.f;
    rho_vec_8 = 0.f;
    rho_vec_9 = 0.f;
    rho_vec_10 = 0.f;
    rho_vec_11 = 0.f;
    rho_vec_12 = 0.f;
    rho_vec_13 = 0.f;
    rho_vec_14 = 0.f;
    rho_vec_15 = 0.f;

    rho_vec_0 = rho00_init;
    rho_vec_1 = rho11_init;
    rho_vec_2 = rho22_init;
    rho_vec_3 = rho33_init;

    printf("Thread started. Message from thread.\n");
    printf("tid = %d\n", tid);
    printf("rho00_init = %f\n", rho_vec_0);
    printf("rho11_init = %f\n", rho_vec_1);
    printf("rho22_init = %f\n", rho_vec_2);
    printf("rho33_init = %f\n", rho_vec_3);



    //float sum_whole[4] = { 0.f,0.f,0.f,0.f };

    float sum_last_0 = 0.f;
    float sum_last_1 = 0.f;
    float sum_last_2 = 0.f;
    float sum_last_3 = 0.f;

    //float sum_2last[4] = { 0.f,0.f,0.f,0.f };
    //float sum_3last[4] = { 0.f,0.f,0.f,0.f };

    const int total_steps = N_steps_per_period * N_periods;

    for (int t_idx_rk4_step = 0; t_idx_rk4_step < total_steps; ++t_idx_rk4_step) {
        const float t_step = t_idx_rk4_step * dt;
        const float eps_t_step = eps0_single_target + A_single_target * cosf(omega * t_step);


        rk4_step_unrolled_v3_safe(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

            t_step,
            eps0_single_target,
            A_single_target
        );

        // post-step stabilization
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

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

        //        if (period_idx == N_periods - 1) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_last[q] += rho_vec[q];
        //        }
        //        else if (period_idx == N_periods - 2) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
        //        }
        //        else if (period_idx == N_periods - 3) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
        //        }

                // store populations (diagonals are rho_vec[0..3])
        const int base = t_idx_rk4_step * 4;
        d_rho_dynamics[base + 0] = rho_vec_0;
        d_rho_dynamics[base + 1] = rho_vec_1;
        d_rho_dynamics[base + 2] = rho_vec_2;
        d_rho_dynamics[base + 3] = rho_vec_3;

        d_time_dynamics[t_idx_rk4_step] = t_step;
        d_eps_dynamics[t_idx_rk4_step] = eps_t_step;

    }

    // final normalization safeguard
    clamp_and_renormalize_vec16_unrolled(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
    );

    //const float inv_whole = 1.0f / float(total_steps);
    const float inv_period = 1.0f / float(N_steps_per_period);

    //// write: first 4 = whole avg, then last, 2last, 3last
    //for (int d = 0; d < 4; ++d) d_out_avg[d] = sum_whole[d] * inv_whole;
    //for (int d = 0; d < 4; ++d) d_out_avg[4 + d] = sum_last[d] * inv_period;
    //for (int d = 0; d < 4; ++d) d_out_avg[8 + d] = sum_2last[d] * inv_period;
    //for (int d = 0; d < 4; ++d) d_out_avg[12 + d] = sum_3last[d] * inv_period;

    // for avg_periods_ouput_option == "last":

    const size_t base = (size_t)tid * 4u;
    // write: first 1 = last
    d_out_avg[base + 0] = sum_last_0 * inv_period;
    d_out_avg[base + 1] = sum_last_1 * inv_period;
    d_out_avg[base + 2] = sum_last_2 * inv_period;
    d_out_avg[base + 3] = sum_last_3 * inv_period;

    printf("tid = %d\n", tid);
    printf("base = %u\n", (unsigned int)base);

}


extern "C" __global__ void lindblad_rk4_kernel_singlemode_unrolled_log(

    // the single chosen point:
    float eps0_single_target,
    float A_single_target,

    float* __restrict__ d_out_avg,
    // output pho dynamics: length = total_steps * 4 (rho00..rho33)
    float* __restrict__ d_rho_dynamics,
    // time output
    float* __restrict__ d_time_dynamics,
    // epsilon (t) output
    float* __restrict__ d_eps_dynamics,

    // log
    LogEntry* __restrict__ d_log_buffer

) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // single thread only
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // rho in vec16 form, initial state
    float rho_vec_0;
    float rho_vec_1;
    float rho_vec_2;
    float rho_vec_3;
    float rho_vec_4;
    float rho_vec_5;
    float rho_vec_6;
    float rho_vec_7;
    float rho_vec_8;
    float rho_vec_9;
    float rho_vec_10;
    float rho_vec_11;
    float rho_vec_12;
    float rho_vec_13;
    float rho_vec_14;
    float rho_vec_15;

    rho_vec_0 = 0.f;
    rho_vec_1 = 0.f;
    rho_vec_2 = 0.f;
    rho_vec_3 = 0.f;
    rho_vec_4 = 0.f;
    rho_vec_5 = 0.f;
    rho_vec_6 = 0.f;
    rho_vec_7 = 0.f;
    rho_vec_8 = 0.f;
    rho_vec_9 = 0.f;
    rho_vec_10 = 0.f;
    rho_vec_11 = 0.f;
    rho_vec_12 = 0.f;
    rho_vec_13 = 0.f;
    rho_vec_14 = 0.f;
    rho_vec_15 = 0.f;

    rho_vec_0 = rho00_init;
    rho_vec_1 = rho11_init;
    rho_vec_2 = rho22_init;
    rho_vec_3 = rho33_init;

    printf("Thread started. Message from thread.\n");
    printf("tid = %d\n", tid);
    printf("rho00_init = %f\n", rho_vec_0);
    printf("rho11_init = %f\n", rho_vec_1);
    printf("rho22_init = %f\n", rho_vec_2);
    printf("rho33_init = %f\n", rho_vec_3);



    //float sum_whole[4] = { 0.f,0.f,0.f,0.f };

    float sum_last_0 = 0.f;
    float sum_last_1 = 0.f;
    float sum_last_2 = 0.f;
    float sum_last_3 = 0.f;

    //float sum_2last[4] = { 0.f,0.f,0.f,0.f };
    //float sum_3last[4] = { 0.f,0.f,0.f,0.f };

    const int total_steps = N_steps_per_period * N_periods;


    for (int t_idx_rk4_step = 0; t_idx_rk4_step < total_steps; ++t_idx_rk4_step) {
        const float t_step = t_idx_rk4_step * dt;
        const float eps_t_step = eps0_single_target + A_single_target * cosf(omega * t_step);


        rk4_step_unrolled_v3_safe_log(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

            t_step,
            eps0_single_target,
            A_single_target,

            d_log_buffer, t_idx_rk4_step
        );

        // post-step stabilization
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

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

        //        if (period_idx == N_periods - 1) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_last[q] += rho_vec[q];
        //        }
        //        else if (period_idx == N_periods - 2) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
        //        }
        //        else if (period_idx == N_periods - 3) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
        //        }

                // store populations (diagonals are rho_vec[0..3])
        const int base = t_idx_rk4_step * 4;
        d_rho_dynamics[base + 0] = rho_vec_0;
        d_rho_dynamics[base + 1] = rho_vec_1;
        d_rho_dynamics[base + 2] = rho_vec_2;
        d_rho_dynamics[base + 3] = rho_vec_3;

        d_time_dynamics[t_idx_rk4_step] = t_step;
        d_eps_dynamics[t_idx_rk4_step] = eps_t_step;

    }

    // final normalization safeguard
    clamp_and_renormalize_vec16_unrolled(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
    );

    //const float inv_whole = 1.0f / float(total_steps);
    const float inv_period = 1.0f / float(N_steps_per_period);

    //// write: first 4 = whole avg, then last, 2last, 3last
    //for (int d = 0; d < 4; ++d) d_out_avg[d] = sum_whole[d] * inv_whole;
    //for (int d = 0; d < 4; ++d) d_out_avg[4 + d] = sum_last[d] * inv_period;
    //for (int d = 0; d < 4; ++d) d_out_avg[8 + d] = sum_2last[d] * inv_period;
    //for (int d = 0; d < 4; ++d) d_out_avg[12 + d] = sum_3last[d] * inv_period;

    // for avg_periods_ouput_option == "last":

    const size_t base = (size_t)tid * 4u;
    // write: first 1 = last
    d_out_avg[base + 0] = sum_last_0 * inv_period;
    d_out_avg[base + 1] = sum_last_1 * inv_period;
    d_out_avg[base + 2] = sum_last_2 * inv_period;
    d_out_avg[base + 3] = sum_last_3 * inv_period;

    printf("tid = %d\n", tid);
    printf("base = %u\n", (unsigned int)base);

}



//////////////////////////////////////////////////////////////



/*
extern "C" __global__ void lindblad_rk4_kernel_singlemode_unrolled_warp(

    // the single chosen point:
    float eps0_single_target,
    float A_single_target,

    float* __restrict__ d_out_avg,
    // output pho dynamics: length = total_steps * 4 (rho00..rho33)
    float* __restrict__ d_rho_dynamics,
    // time output
    float* __restrict__ d_time_dynamics,
    // epsilon (t) output
    float* __restrict__ d_eps_dynamics

) {

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Within-warp indexing
    const int warp_id = thread_id / WARP_SIZE;
    const int lane_id = thread_id % WARP_SIZE;

    // Subdivide warp into groups handling one trajectory
    const int group_id = lane_id / LANE_GROUP_SIZE;
    //const int lane_in_group = lane_id % LANE_GROUP_SIZE;

    const int traj_id = warp_id * K_TRAJ_PER_WARP + group_id;

    // We want only one trajectory to be processed
    if (traj_id != 0) return;





    // rho in vec16 form, initial state
    float rho_vec_0;
    float rho_vec_1;
    float rho_vec_2;
    float rho_vec_3;
    float rho_vec_4;
    float rho_vec_5;
    float rho_vec_6;
    float rho_vec_7;
    float rho_vec_8;
    float rho_vec_9;
    float rho_vec_10;
    float rho_vec_11;
    float rho_vec_12;
    float rho_vec_13;
    float rho_vec_14;
    float rho_vec_15;

    rho_vec_0 = 0.f;
    rho_vec_1 = 0.f;
    rho_vec_2 = 0.f;
    rho_vec_3 = 0.f;
    rho_vec_4 = 0.f;
    rho_vec_5 = 0.f;
    rho_vec_6 = 0.f;
    rho_vec_7 = 0.f;
    rho_vec_8 = 0.f;
    rho_vec_9 = 0.f;
    rho_vec_10 = 0.f;
    rho_vec_11 = 0.f;
    rho_vec_12 = 0.f;
    rho_vec_13 = 0.f;
    rho_vec_14 = 0.f;
    rho_vec_15 = 0.f;

    rho_vec_0 = rho00_init;
    rho_vec_1 = rho11_init;
    rho_vec_2 = rho22_init;
    rho_vec_3 = rho33_init;

    printf("Thread started. Message from thread.\n");
    printf("thread_id = %d\n", thread_id);
    printf("rho00_init = %f\n", rho_vec_0);
    printf("rho11_init = %f\n", rho_vec_1);
    printf("rho22_init = %f\n", rho_vec_2);
    printf("rho33_init = %f\n", rho_vec_3);



    //float sum_whole[4] = { 0.f,0.f,0.f,0.f };

    float sum_last_0 = 0.f;
    float sum_last_1 = 0.f;
    float sum_last_2 = 0.f;
    float sum_last_3 = 0.f;

    //float sum_2last[4] = { 0.f,0.f,0.f,0.f };
    //float sum_3last[4] = { 0.f,0.f,0.f,0.f };

    const int total_steps = N_steps_per_period * N_periods;

    for (int t_idx_rk4_step = 0; t_idx_rk4_step < total_steps; ++t_idx_rk4_step) {
        const float t_step = t_idx_rk4_step * dt;
        const float eps_t_step = eps0_single_target + A_single_target * cosf(omega * t_step);


        rk4_step_unrolled_v3_safe(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

            t_step,
            eps0_single_target,
            A_single_target
        );

        // post-step stabilization
        clamp_and_renormalize_vec16_unrolled(
            rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
            rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
            rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
            rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
        );

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

        //        if (period_idx == N_periods - 1) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_last[q] += rho_vec[q];
        //        }
        //        else if (period_idx == N_periods - 2) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_2last[q] += rho_vec[q];
        //        }
        //        else if (period_idx == N_periods - 3) {
        //#pragma unroll
        //            for (int q = 0; q < 4; q++) sum_3last[q] += rho_vec[q];
        //        }

                // store populations (diagonals are rho_vec[0..3])
        const int base = t_idx_rk4_step * 4;
        d_rho_dynamics[base + 0] = rho_vec_0;
        d_rho_dynamics[base + 1] = rho_vec_1;
        d_rho_dynamics[base + 2] = rho_vec_2;
        d_rho_dynamics[base + 3] = rho_vec_3;

        d_time_dynamics[t_idx_rk4_step] = t_step;
        d_eps_dynamics[t_idx_rk4_step] = eps_t_step;

    }

    // final normalization safeguard
    clamp_and_renormalize_vec16_unrolled(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15
    );

    //const float inv_whole = 1.0f / float(total_steps);
    const float inv_period = 1.0f / float(N_steps_per_period);

    //// write: first 4 = whole avg, then last, 2last, 3last
    //for (int d = 0; d < 4; ++d) d_out_avg[d] = sum_whole[d] * inv_whole;
    //for (int d = 0; d < 4; ++d) d_out_avg[4 + d] = sum_last[d] * inv_period;
    //for (int d = 0; d < 4; ++d) d_out_avg[8 + d] = sum_2last[d] * inv_period;
    //for (int d = 0; d < 4; ++d) d_out_avg[12 + d] = sum_3last[d] * inv_period;

    // for avg_periods_ouput_option == "last":

    const size_t base = (size_t)thread_id * 4u;
    // write: first 1 = last
    d_out_avg[base + 0] = sum_last_0 * inv_period;
    d_out_avg[base + 1] = sum_last_1 * inv_period;
    d_out_avg[base + 2] = sum_last_2 * inv_period;
    d_out_avg[base + 3] = sum_last_3 * inv_period;

    printf("thread_id = %d\n", thread_id);
    printf("base = %u\n", (unsigned int)base);

}

*/





/* not yet implemented. for faster compilation
// template. not implemented
extern "C" __global__ void lindblad_rk4_kernel_singlemode_unrolled_warp_log(

    // the single chosen point:
    float eps0_single_target,
    float A_single_target,

    float* __restrict__ d_out_avg,
    // output pho dynamics: length = total_steps * 4 (rho00..rho33)
    float* __restrict__ d_rho_dynamics,
    // time output
    float* __restrict__ d_time_dynamics,
    // epsilon (t) output
    float* __restrict__ d_eps_dynamics,

    // log
    LogEntry* __restrict__ d_log_buffer

) {}
*/