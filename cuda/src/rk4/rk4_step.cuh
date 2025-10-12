// src/rk4/rk4_step.cuh

#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "constants.cuh"

#include "rk4/rk4_substep.cuh"








__device__ __forceinline__
void rk4_step(
    float rho_vec[16],
    const float t_step,

    const float eps0,
    const float A
) {

    //float H01 = pi_alpha * delta_R;
    //float H02 = pi_alpha * delta_L;
    //float H03 = 0.f;
    //float H12 = pi_alpha * delta_C;
    //float H13 = pi_alpha * delta_L;
    //float H23 = pi_alpha * delta_R;

    register float k_cur[16];
    register float acc[16];  // running weighted sum for RK4

    //float tmp;

    register float t_substep;
    register float eps_t_substep;
    //float H00, H11, H22, H33;

    // initialize accumulator to zero
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

    for (int i = 0; i < 16; ++i) k_cur[i] = 0.0f;

    // ---- stage 1 ----

    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)
    
    t_substep = t_step;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    //printf("\n");
    //printf("before RK4 stage 1: t_step = %f\n", t_step);
    //printf("before RK4 stage 1: eps_t_substep = %f\n", eps_t_substep);
    //printf("before RK4 stage 1: rho_vec[0..3] = %f, %f, %f, %f\n", rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
    //printf("before RK4 stage 1: k_cur[0..3] = %f, %f, %f, %f\n", k_cur[0], k_cur[1], k_cur[2], k_cur[3]);

    compute_drho(
        rho_vec, k_cur, eps_t_substep
    );

    //printf(" after RK4 stage 1: rho_vec[0..3] = %f, %f, %f, %f\n", rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
    //printf(" after RK4 stage 1: k_cur[0..3] = %f, %f, %f, %f\n", k_cur[0], k_cur[1], k_cur[2], k_cur[3]);


#pragma unroll
    for (int i = 0; i < 16; ++i)
        acc[i] += k_cur[i];  // weight = 1 for k1

    // ---- stage 2 ----
#pragma unroll
    for (int i = 0; i < 16; ++i)
        k_cur[i] = rho_vec[i] + 0.5f * dt * k_cur[i];  // reuse k_cur for temp


    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)
    
    t_substep = t_step + 0.5f * dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho(
        k_cur, k_cur, eps_t_substep
    );

#pragma unroll
    for (int i = 0; i < 16; ++i)
        acc[i] += 2.0f * k_cur[i];  // weight = 2 for k2

    // ---- stage 3 ----
#pragma unroll
    for (int i = 0; i < 16; ++i)
        k_cur[i] = rho_vec[i] + 0.5f * dt * k_cur[i];

    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)

    t_substep = t_step + 0.5f * dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho(
        k_cur, k_cur, eps_t_substep
    );


#pragma unroll
    for (int i = 0; i < 16; ++i)
        acc[i] += 2.0f * k_cur[i];  // weight = 2 for k3

    // ---- stage 4 ----
#pragma unroll
    for (int i = 0; i < 16; ++i)
        k_cur[i] = rho_vec[i] + dt * k_cur[i];

    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)

    t_substep = t_step + dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho(
        k_cur, k_cur, eps_t_substep
    );


#pragma unroll
    for (int i = 0; i < 16; ++i)
        acc[i] += k_cur[i];  // weight = 1 for k4

    // ---- final update ----
#pragma unroll
    for (int i = 0; i < 16; ++i)
        rho_vec[i] += (dt / 6.0f) * acc[i];
}


__device__ __forceinline__
void rk4_step_log(
    float rho_vec[16],
    const float t_step,

    const float eps0,
    const float A,

    // log
    LogEntry* __restrict__ d_log_buffer,
    const int t_idx_step
) {

    int t_idx_substep;
    int substep_num;

    //float H01 = pi_alpha * delta_R;
    //float H02 = pi_alpha * delta_L;
    //float H03 = 0.f;
    //float H12 = pi_alpha * delta_C;
    //float H13 = pi_alpha * delta_L;
    //float H23 = pi_alpha * delta_R;

    register float k_cur[16];
    register float acc[16];  // running weighted sum for RK4

    //float tmp;

    register float t_substep;
    register float eps_t_substep;
    //float H00, H11, H22, H33;

    // initialize accumulator to zero
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

    for (int i = 0; i < 16; ++i) k_cur[i] = 0.0f;

    // ---- stage 1 ----

    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)
    
    t_substep = t_step;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    //printf("\n");
    //printf("before RK4 stage 1: t_step = %f\n", t_step);
    //printf("before RK4 stage 1: eps_t_substep = %f\n", eps_t_substep);
    //printf("before RK4 stage 1: rho_vec[0..3] = %f, %f, %f, %f\n", rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
    //printf("before RK4 stage 1: k_cur[0..3] = %f, %f, %f, %f\n", k_cur[0], k_cur[1], k_cur[2], k_cur[3]);

    substep_num = 0;
    t_idx_substep = t_idx_step * 4 + substep_num;
    d_log_buffer[t_idx_substep].t_substep = t_substep;

    compute_drho_log(
        rho_vec, k_cur, eps_t_substep,
        d_log_buffer, t_idx_substep
    );

    //printf(" after RK4 stage 1: rho_vec[0..3] = %f, %f, %f, %f\n", rho_vec[0], rho_vec[1], rho_vec[2], rho_vec[3]);
    //printf(" after RK4 stage 1: k_cur[0..3] = %f, %f, %f, %f\n", k_cur[0], k_cur[1], k_cur[2], k_cur[3]);


#pragma unroll
    for (int i = 0; i < 16; ++i)
        acc[i] += k_cur[i];  // weight = 1 for k1

    // ---- stage 2 ----
#pragma unroll
    for (int i = 0; i < 16; ++i)
        k_cur[i] = rho_vec[i] + 0.5f * dt * k_cur[i];  // reuse k_cur for temp


    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)

    t_substep = t_step + 0.5f * dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    substep_num = 1;
    t_idx_substep = t_idx_step * 4 + substep_num;
    d_log_buffer[t_idx_substep].t_substep = t_substep;

    compute_drho_log(
        k_cur, k_cur, eps_t_substep,
        d_log_buffer, t_idx_substep
    );

#pragma unroll
    for (int i = 0; i < 16; ++i)
        acc[i] += 2.0f * k_cur[i];  // weight = 2 for k2

    // ---- stage 3 ----
#pragma unroll
    for (int i = 0; i < 16; ++i)
        k_cur[i] = rho_vec[i] + 0.5f * dt * k_cur[i];

    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)

    t_substep = t_step + 0.5f * dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    substep_num = 2;
    t_idx_substep = t_idx_step * 4 + substep_num;
    d_log_buffer[t_idx_substep].t_substep = t_substep;

    compute_drho_log(
        k_cur, k_cur, eps_t_substep,
        d_log_buffer, t_idx_substep
    );


#pragma unroll
    for (int i = 0; i < 16; ++i)
        acc[i] += 2.0f * k_cur[i];  // weight = 2 for k3

    // ---- stage 4 ----
#pragma unroll
    for (int i = 0; i < 16; ++i)
        k_cur[i] = rho_vec[i] + dt * k_cur[i];

    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)

    t_substep = t_step + dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    substep_num = 3;
    t_idx_substep = t_idx_step * 4 + substep_num;
    d_log_buffer[t_idx_substep].t_substep = t_substep;

    compute_drho_log(
        k_cur, k_cur, eps_t_substep,
        d_log_buffer, t_idx_substep
    );


#pragma unroll
    for (int i = 0; i < 16; ++i)
        acc[i] += k_cur[i];  // weight = 1 for k4

    // ---- final update ----
#pragma unroll
    for (int i = 0; i < 16; ++i)
        rho_vec[i] += (dt / 6.0f) * acc[i];
}





__device__ __forceinline__
void rk4_step_unrolled(
    float& rho_vec_0, float& rho_vec_1, float& rho_vec_2, float& rho_vec_3,
    float& rho_vec_4, float& rho_vec_5, float& rho_vec_6, float& rho_vec_7,
    float& rho_vec_8, float& rho_vec_9, float& rho_vec_10, float& rho_vec_11,
    float& rho_vec_12, float& rho_vec_13, float& rho_vec_14, float& rho_vec_15,

    const float t_step,
    const float eps0,
    const float A
)
{

    //float H01 = pi_alpha * delta_R;
    //float H02 = pi_alpha * delta_L;
    //float H03 = 0.f;
    //float H12 = pi_alpha * delta_C;
    //float H13 = pi_alpha * delta_L;
    //float H23 = pi_alpha * delta_R;

    // derivative buffers
    register float k_cur_0;
    register float k_cur_1;
    register float k_cur_2;
    register float k_cur_3;
    register float k_cur_4;
    register float k_cur_5;
    register float k_cur_6;
    register float k_cur_7;
    register float k_cur_8;
    register float k_cur_9;
    register float k_cur_10;
    register float k_cur_11;
    register float k_cur_12;
    register float k_cur_13;
    register float k_cur_14;
    register float k_cur_15;

    // accumulators // running weighted sum for RK4
    register float acc_0;
    register float acc_1;
    register float acc_2;
    register float acc_3;
    register float acc_4;
    register float acc_5;
    register float acc_6;
    register float acc_7;
    register float acc_8;
    register float acc_9;
    register float acc_10;
    register float acc_11;
    register float acc_12;
    register float acc_13;
    register float acc_14;
    register float acc_15;
      

    //float tmp;

    register float t_substep;
    register float eps_t_substep;
    //float H00, H11, H22, H33;

    // initialize accumulator to zero

    acc_0 = 0.0f;
    acc_1 = 0.0f;
    acc_2 = 0.0f;
    acc_3 = 0.0f;
    acc_4 = 0.0f;
    acc_5 = 0.0f;
    acc_6 = 0.0f;
    acc_7 = 0.0f;
    acc_8 = 0.0f;
    acc_9 = 0.0f;
    acc_10 = 0.0f;
    acc_11 = 0.0f;
    acc_12 = 0.0f;
    acc_13 = 0.0f;
    acc_14 = 0.0f;
    acc_15 = 0.0f;

    k_cur_0 = 0.0f;
    k_cur_1 = 0.0f;
    k_cur_2 = 0.0f;
    k_cur_3 = 0.0f;
    k_cur_4 = 0.0f;
    k_cur_5 = 0.0f;
    k_cur_6 = 0.0f;
    k_cur_7 = 0.0f;
    k_cur_8 = 0.0f;
    k_cur_9 = 0.0f;
    k_cur_10 = 0.0f;
    k_cur_11 = 0.0f;
    k_cur_12 = 0.0f;
    k_cur_13 = 0.0f;
    k_cur_14 = 0.0f;
    k_cur_15 = 0.0f;

    // ---- stage 1 ----

    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)
    
    t_substep = t_step;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    //printf("\n");
    //printf("before RK4 stage 1: t_step = %f\n", t_step);
    //printf("before RK4 stage 1: eps_t_substep = %f\n", eps_t_substep);
    //printf("before RK4 stage 1: rho_vec[0..3] = %f, %f, %f, %f\n", rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3);
    //printf("before RK4 stage 1: k_cur[0..3] = %f, %f, %f, %f\n", k_cur_0, k_cur_1, k_cur_2, k_cur_3);

    compute_drho_unrolled(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    //printf(" after RK4 stage 1: rho_vec[0..3] = %f, %f, %f, %f\n", rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3);
    //printf(" after RK4 stage 1: k_cur[0..3] = %f, %f, %f, %f\n", k_cur_0, k_cur_1, k_cur_2, k_cur_3);


    // weight = 1 for k1
    acc_0 += k_cur_0;
    acc_1 += k_cur_1;
    acc_2 += k_cur_2;
    acc_3 += k_cur_3;
    acc_4 += k_cur_4;
    acc_5 += k_cur_5;
    acc_6 += k_cur_6;
    acc_7 += k_cur_7;
    acc_8 += k_cur_8;
    acc_9 += k_cur_9;
    acc_10 += k_cur_10;
    acc_11 += k_cur_11;
    acc_12 += k_cur_12;
    acc_13 += k_cur_13;
    acc_14 += k_cur_14;
    acc_15 += k_cur_15;


    // ---- stage 2 ----

    // reuse k_cur for temp
    k_cur_0 = rho_vec_0 + 0.5f * dt * k_cur_0;
    k_cur_1 = rho_vec_1 + 0.5f * dt * k_cur_1;
    k_cur_2 = rho_vec_2 + 0.5f * dt * k_cur_2;
    k_cur_3 = rho_vec_3 + 0.5f * dt * k_cur_3;
    k_cur_4 = rho_vec_4 + 0.5f * dt * k_cur_4;
    k_cur_5 = rho_vec_5 + 0.5f * dt * k_cur_5;
    k_cur_6 = rho_vec_6 + 0.5f * dt * k_cur_6;
    k_cur_7 = rho_vec_7 + 0.5f * dt * k_cur_7;
    k_cur_8 = rho_vec_8 + 0.5f * dt * k_cur_8;
    k_cur_9 = rho_vec_9 + 0.5f * dt * k_cur_9;
    k_cur_10 = rho_vec_10 + 0.5f * dt * k_cur_10;
    k_cur_11 = rho_vec_11 + 0.5f * dt * k_cur_11;
    k_cur_12 = rho_vec_12 + 0.5f * dt * k_cur_12;
    k_cur_13 = rho_vec_13 + 0.5f * dt * k_cur_13;
    k_cur_14 = rho_vec_14 + 0.5f * dt * k_cur_14;
    k_cur_15 = rho_vec_15 + 0.5f * dt * k_cur_15;
    


    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)
    
    t_substep = t_step + 0.5f * dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho_unrolled(
        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 2 for k2
    acc_0 += 2.0f * k_cur_0;
    acc_1 += 2.0f * k_cur_1;
    acc_2 += 2.0f * k_cur_2;
    acc_3 += 2.0f * k_cur_3;
    acc_4 += 2.0f * k_cur_4;
    acc_5 += 2.0f * k_cur_5;
    acc_6 += 2.0f * k_cur_6;
    acc_7 += 2.0f * k_cur_7;
    acc_8 += 2.0f * k_cur_8;
    acc_9 += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10;
    acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12;
    acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14;
    acc_15 += 2.0f * k_cur_15;
    

    // ---- stage 3 ----

    k_cur_0 = rho_vec_0 + 0.5f * dt * k_cur_0;
    k_cur_1 = rho_vec_1 + 0.5f * dt * k_cur_1;
    k_cur_2 = rho_vec_2 + 0.5f * dt * k_cur_2;
    k_cur_3 = rho_vec_3 + 0.5f * dt * k_cur_3;
    k_cur_4 = rho_vec_4 + 0.5f * dt * k_cur_4;
    k_cur_5 = rho_vec_5 + 0.5f * dt * k_cur_5;
    k_cur_6 = rho_vec_6 + 0.5f * dt * k_cur_6;
    k_cur_7 = rho_vec_7 + 0.5f * dt * k_cur_7;
    k_cur_8 = rho_vec_8 + 0.5f * dt * k_cur_8;
    k_cur_9 = rho_vec_9 + 0.5f * dt * k_cur_9;
    k_cur_10 = rho_vec_10 + 0.5f * dt * k_cur_10;
    k_cur_11 = rho_vec_11 + 0.5f * dt * k_cur_11;
    k_cur_12 = rho_vec_12 + 0.5f * dt * k_cur_12;
    k_cur_13 = rho_vec_13 + 0.5f * dt * k_cur_13;
    k_cur_14 = rho_vec_14 + 0.5f * dt * k_cur_14;
    k_cur_15 = rho_vec_15 + 0.5f * dt * k_cur_15;


    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)

    t_substep = t_step + 0.5f * dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho_unrolled(
        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );


    // weight = 2 for k3
    acc_0 += 2.0f * k_cur_0;
    acc_1 += 2.0f * k_cur_1;
    acc_2 += 2.0f * k_cur_2;
    acc_3 += 2.0f * k_cur_3;
    acc_4 += 2.0f * k_cur_4;
    acc_5 += 2.0f * k_cur_5;
    acc_6 += 2.0f * k_cur_6;
    acc_7 += 2.0f * k_cur_7;
    acc_8 += 2.0f * k_cur_8;
    acc_9 += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10;
    acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12;
    acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14;
    acc_15 += 2.0f * k_cur_15;
    

    // ---- stage 4 ----

    k_cur_0 = rho_vec_0 + dt * k_cur_0;
    k_cur_1 = rho_vec_1 + dt * k_cur_1;
    k_cur_2 = rho_vec_2 + dt * k_cur_2;
    k_cur_3 = rho_vec_3 + dt * k_cur_3;
    k_cur_4 = rho_vec_4 + dt * k_cur_4;
    k_cur_5 = rho_vec_5 + dt * k_cur_5;
    k_cur_6 = rho_vec_6 + dt * k_cur_6;
    k_cur_7 = rho_vec_7 + dt * k_cur_7;
    k_cur_8 = rho_vec_8 + dt * k_cur_8;
    k_cur_9 = rho_vec_9 + dt * k_cur_9;
    k_cur_10 = rho_vec_10 + dt * k_cur_10;
    k_cur_11 = rho_vec_11 + dt * k_cur_11;
    k_cur_12 = rho_vec_12 + dt * k_cur_12;
    k_cur_13 = rho_vec_13 + dt * k_cur_13;
    k_cur_14 = rho_vec_14 + dt * k_cur_14;
    k_cur_15 = rho_vec_15 + dt * k_cur_15;


    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)

    t_substep = t_step + dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho_unrolled(
        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );


    // weight = 1 for k4
    acc_0 += k_cur_0;
    acc_1 += k_cur_1;
    acc_2 += k_cur_2;
    acc_3 += k_cur_3;
    acc_4 += k_cur_4;
    acc_5 += k_cur_5;
    acc_6 += k_cur_6;
    acc_7 += k_cur_7;
    acc_8 += k_cur_8;
    acc_9 += k_cur_9;
    acc_10 += k_cur_10;
    acc_11 += k_cur_11;
    acc_12 += k_cur_12;
    acc_13 += k_cur_13;
    acc_14 += k_cur_14;
    acc_15 += k_cur_15;
    

    // ---- final update ----

    rho_vec_0 += (dt / 6.0f) * acc_0;
    rho_vec_1 += (dt / 6.0f) * acc_1;
    rho_vec_2 += (dt / 6.0f) * acc_2;
    rho_vec_3 += (dt / 6.0f) * acc_3;
    rho_vec_4 += (dt / 6.0f) * acc_4;
    rho_vec_5 += (dt / 6.0f) * acc_5;
    rho_vec_6 += (dt / 6.0f) * acc_6;
    rho_vec_7 += (dt / 6.0f) * acc_7;
    rho_vec_8 += (dt / 6.0f) * acc_8;
    rho_vec_9 += (dt / 6.0f) * acc_9;
    rho_vec_10 += (dt / 6.0f) * acc_10;
    rho_vec_11 += (dt / 6.0f) * acc_11;
    rho_vec_12 += (dt / 6.0f) * acc_12;
    rho_vec_13 += (dt / 6.0f) * acc_13;
    rho_vec_14 += (dt / 6.0f) * acc_14;
    rho_vec_15 += (dt / 6.0f) * acc_15;

}




__device__ __forceinline__
void rk4_step_unrolled_log(
    float& rho_vec_0, float& rho_vec_1, float& rho_vec_2, float& rho_vec_3,
    float& rho_vec_4, float& rho_vec_5, float& rho_vec_6, float& rho_vec_7,
    float& rho_vec_8, float& rho_vec_9, float& rho_vec_10, float& rho_vec_11,
    float& rho_vec_12, float& rho_vec_13, float& rho_vec_14, float& rho_vec_15,

    const float t_step,
    const float eps0,
    const float A,

    // log
    LogEntry* __restrict__ d_log_buffer,
    const int t_idx_step
)
{

    int t_idx_substep;
    int substep_num;

    //float H01 = pi_alpha * delta_R;
    //float H02 = pi_alpha * delta_L;
    //float H03 = 0.f;
    //float H12 = pi_alpha * delta_C;
    //float H13 = pi_alpha * delta_L;
    //float H23 = pi_alpha * delta_R;

    // derivative buffers
    register float k_cur_0;
    register float k_cur_1;
    register float k_cur_2;
    register float k_cur_3;
    register float k_cur_4;
    register float k_cur_5;
    register float k_cur_6;
    register float k_cur_7;
    register float k_cur_8;
    register float k_cur_9;
    register float k_cur_10;
    register float k_cur_11;
    register float k_cur_12;
    register float k_cur_13;
    register float k_cur_14;
    register float k_cur_15;

    // accumulators // running weighted sum for RK4
    register float acc_0;
    register float acc_1;
    register float acc_2;
    register float acc_3;
    register float acc_4;
    register float acc_5;
    register float acc_6;
    register float acc_7;
    register float acc_8;
    register float acc_9;
    register float acc_10;
    register float acc_11;
    register float acc_12;
    register float acc_13;
    register float acc_14;
    register float acc_15;


    //float tmp;

    register float t_substep;
    register float eps_t_substep;
    //float H00, H11, H22, H33;

    // initialize accumulator to zero

    acc_0 = 0.0f;
    acc_1 = 0.0f;
    acc_2 = 0.0f;
    acc_3 = 0.0f;
    acc_4 = 0.0f;
    acc_5 = 0.0f;
    acc_6 = 0.0f;
    acc_7 = 0.0f;
    acc_8 = 0.0f;
    acc_9 = 0.0f;
    acc_10 = 0.0f;
    acc_11 = 0.0f;
    acc_12 = 0.0f;
    acc_13 = 0.0f;
    acc_14 = 0.0f;
    acc_15 = 0.0f;

    k_cur_0 = 0.0f;
    k_cur_1 = 0.0f;
    k_cur_2 = 0.0f;
    k_cur_3 = 0.0f;
    k_cur_4 = 0.0f;
    k_cur_5 = 0.0f;
    k_cur_6 = 0.0f;
    k_cur_7 = 0.0f;
    k_cur_8 = 0.0f;
    k_cur_9 = 0.0f;
    k_cur_10 = 0.0f;
    k_cur_11 = 0.0f;
    k_cur_12 = 0.0f;
    k_cur_13 = 0.0f;
    k_cur_14 = 0.0f;
    k_cur_15 = 0.0f;

    // ---- stage 1 ----

    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)

    t_substep = t_step;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    //printf("\n");
    //printf("before RK4 stage 1: t_step = %f\n", t_step);
    //printf("before RK4 stage 1: eps_t_substep = %f\n", eps_t_substep);
    //printf("before RK4 stage 1: rho_vec[0..3] = %f, %f, %f, %f\n", rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3);
    //printf("before RK4 stage 1: k_cur[0..3] = %f, %f, %f, %f\n", k_cur_0, k_cur_1, k_cur_2, k_cur_3);

    substep_num = 0;
    t_idx_substep = t_idx_step * 4 + substep_num;

    compute_drho_unrolled_log(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep,
        t_idx_step, substep_num, t_step, t_substep
    );

    //printf(" after RK4 stage 1: rho_vec[0..3] = %f, %f, %f, %f\n", rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3);
    //printf(" after RK4 stage 1: k_cur[0..3] = %f, %f, %f, %f\n", k_cur_0, k_cur_1, k_cur_2, k_cur_3);


    // weight = 1 for k1
    acc_0 += k_cur_0;
    acc_1 += k_cur_1;
    acc_2 += k_cur_2;
    acc_3 += k_cur_3;
    acc_4 += k_cur_4;
    acc_5 += k_cur_5;
    acc_6 += k_cur_6;
    acc_7 += k_cur_7;
    acc_8 += k_cur_8;
    acc_9 += k_cur_9;
    acc_10 += k_cur_10;
    acc_11 += k_cur_11;
    acc_12 += k_cur_12;
    acc_13 += k_cur_13;
    acc_14 += k_cur_14;
    acc_15 += k_cur_15;


    // ---- stage 2 ----

    // reuse k_cur for temp
    k_cur_0 = rho_vec_0 + 0.5f * dt * k_cur_0;
    k_cur_1 = rho_vec_1 + 0.5f * dt * k_cur_1;
    k_cur_2 = rho_vec_2 + 0.5f * dt * k_cur_2;
    k_cur_3 = rho_vec_3 + 0.5f * dt * k_cur_3;
    k_cur_4 = rho_vec_4 + 0.5f * dt * k_cur_4;
    k_cur_5 = rho_vec_5 + 0.5f * dt * k_cur_5;
    k_cur_6 = rho_vec_6 + 0.5f * dt * k_cur_6;
    k_cur_7 = rho_vec_7 + 0.5f * dt * k_cur_7;
    k_cur_8 = rho_vec_8 + 0.5f * dt * k_cur_8;
    k_cur_9 = rho_vec_9 + 0.5f * dt * k_cur_9;
    k_cur_10 = rho_vec_10 + 0.5f * dt * k_cur_10;
    k_cur_11 = rho_vec_11 + 0.5f * dt * k_cur_11;
    k_cur_12 = rho_vec_12 + 0.5f * dt * k_cur_12;
    k_cur_13 = rho_vec_13 + 0.5f * dt * k_cur_13;
    k_cur_14 = rho_vec_14 + 0.5f * dt * k_cur_14;
    k_cur_15 = rho_vec_15 + 0.5f * dt * k_cur_15;



    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)

    t_substep = t_step + 0.5f * dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    substep_num = 1;
    t_idx_substep = t_idx_step * 4 + substep_num;

    compute_drho_unrolled_log(
        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep,
        t_idx_step, substep_num, t_step, t_substep
    );

    // weight = 2 for k2
    acc_0 += 2.0f * k_cur_0;
    acc_1 += 2.0f * k_cur_1;
    acc_2 += 2.0f * k_cur_2;
    acc_3 += 2.0f * k_cur_3;
    acc_4 += 2.0f * k_cur_4;
    acc_5 += 2.0f * k_cur_5;
    acc_6 += 2.0f * k_cur_6;
    acc_7 += 2.0f * k_cur_7;
    acc_8 += 2.0f * k_cur_8;
    acc_9 += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10;
    acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12;
    acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14;
    acc_15 += 2.0f * k_cur_15;


    // ---- stage 3 ----

    k_cur_0 = rho_vec_0 + 0.5f * dt * k_cur_0;
    k_cur_1 = rho_vec_1 + 0.5f * dt * k_cur_1;
    k_cur_2 = rho_vec_2 + 0.5f * dt * k_cur_2;
    k_cur_3 = rho_vec_3 + 0.5f * dt * k_cur_3;
    k_cur_4 = rho_vec_4 + 0.5f * dt * k_cur_4;
    k_cur_5 = rho_vec_5 + 0.5f * dt * k_cur_5;
    k_cur_6 = rho_vec_6 + 0.5f * dt * k_cur_6;
    k_cur_7 = rho_vec_7 + 0.5f * dt * k_cur_7;
    k_cur_8 = rho_vec_8 + 0.5f * dt * k_cur_8;
    k_cur_9 = rho_vec_9 + 0.5f * dt * k_cur_9;
    k_cur_10 = rho_vec_10 + 0.5f * dt * k_cur_10;
    k_cur_11 = rho_vec_11 + 0.5f * dt * k_cur_11;
    k_cur_12 = rho_vec_12 + 0.5f * dt * k_cur_12;
    k_cur_13 = rho_vec_13 + 0.5f * dt * k_cur_13;
    k_cur_14 = rho_vec_14 + 0.5f * dt * k_cur_14;
    k_cur_15 = rho_vec_15 + 0.5f * dt * k_cur_15;


    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)

    t_substep = t_step + 0.5f * dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    substep_num = 2;
    t_idx_substep = t_idx_step * 4 + substep_num;

    compute_drho_unrolled_log(
        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep,
        t_idx_step, substep_num, t_step, t_substep
    );


    // weight = 2 for k3
    acc_0 += 2.0f * k_cur_0;
    acc_1 += 2.0f * k_cur_1;
    acc_2 += 2.0f * k_cur_2;
    acc_3 += 2.0f * k_cur_3;
    acc_4 += 2.0f * k_cur_4;
    acc_5 += 2.0f * k_cur_5;
    acc_6 += 2.0f * k_cur_6;
    acc_7 += 2.0f * k_cur_7;
    acc_8 += 2.0f * k_cur_8;
    acc_9 += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10;
    acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12;
    acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14;
    acc_15 += 2.0f * k_cur_15;


    // ---- stage 4 ----

    k_cur_0 = rho_vec_0 + dt * k_cur_0;
    k_cur_1 = rho_vec_1 + dt * k_cur_1;
    k_cur_2 = rho_vec_2 + dt * k_cur_2;
    k_cur_3 = rho_vec_3 + dt * k_cur_3;
    k_cur_4 = rho_vec_4 + dt * k_cur_4;
    k_cur_5 = rho_vec_5 + dt * k_cur_5;
    k_cur_6 = rho_vec_6 + dt * k_cur_6;
    k_cur_7 = rho_vec_7 + dt * k_cur_7;
    k_cur_8 = rho_vec_8 + dt * k_cur_8;
    k_cur_9 = rho_vec_9 + dt * k_cur_9;
    k_cur_10 = rho_vec_10 + dt * k_cur_10;
    k_cur_11 = rho_vec_11 + dt * k_cur_11;
    k_cur_12 = rho_vec_12 + dt * k_cur_12;
    k_cur_13 = rho_vec_13 + dt * k_cur_13;
    k_cur_14 = rho_vec_14 + dt * k_cur_14;
    k_cur_15 = rho_vec_15 + dt * k_cur_15;


    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)

    t_substep = t_step + dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    substep_num = 3;
    t_idx_substep = t_idx_step * 4 + substep_num;

    compute_drho_unrolled_log(
        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep,
        t_idx_step, substep_num, t_step, t_substep
    );


    // weight = 1 for k4
    acc_0 += k_cur_0;
    acc_1 += k_cur_1;
    acc_2 += k_cur_2;
    acc_3 += k_cur_3;
    acc_4 += k_cur_4;
    acc_5 += k_cur_5;
    acc_6 += k_cur_6;
    acc_7 += k_cur_7;
    acc_8 += k_cur_8;
    acc_9 += k_cur_9;
    acc_10 += k_cur_10;
    acc_11 += k_cur_11;
    acc_12 += k_cur_12;
    acc_13 += k_cur_13;
    acc_14 += k_cur_14;
    acc_15 += k_cur_15;


    // ---- final update ----

    rho_vec_0 += (dt / 6.0f) * acc_0;
    rho_vec_1 += (dt / 6.0f) * acc_1;
    rho_vec_2 += (dt / 6.0f) * acc_2;
    rho_vec_3 += (dt / 6.0f) * acc_3;
    rho_vec_4 += (dt / 6.0f) * acc_4;
    rho_vec_5 += (dt / 6.0f) * acc_5;
    rho_vec_6 += (dt / 6.0f) * acc_6;
    rho_vec_7 += (dt / 6.0f) * acc_7;
    rho_vec_8 += (dt / 6.0f) * acc_8;
    rho_vec_9 += (dt / 6.0f) * acc_9;
    rho_vec_10 += (dt / 6.0f) * acc_10;
    rho_vec_11 += (dt / 6.0f) * acc_11;
    rho_vec_12 += (dt / 6.0f) * acc_12;
    rho_vec_13 += (dt / 6.0f) * acc_13;
    rho_vec_14 += (dt / 6.0f) * acc_14;
    rho_vec_15 += (dt / 6.0f) * acc_15;

}






__device__ __forceinline__
void rk4_step_unrolled_v2_safe(
    float& rho_vec_0,  float& rho_vec_1,  float& rho_vec_2,  float& rho_vec_3,
    float& rho_vec_4,  float& rho_vec_5,  float& rho_vec_6,  float& rho_vec_7,
    float& rho_vec_8,  float& rho_vec_9,  float& rho_vec_10, float& rho_vec_11,
    float& rho_vec_12, float& rho_vec_13, float& rho_vec_14, float& rho_vec_15,

    const float t_step,
    const float eps0,
    const float A
) {
    // derivative buffers
    register float k_cur_0, k_cur_1, k_cur_2, k_cur_3;
    register float k_cur_4, k_cur_5, k_cur_6, k_cur_7;
    register float k_cur_8, k_cur_9, k_cur_10, k_cur_11;
    register float k_cur_12, k_cur_13, k_cur_14, k_cur_15;

    // accumulators
    register float acc_0 = 0.0f,  acc_1 = 0.0f,  acc_2 = 0.0f,  acc_3 = 0.0f;
    register float acc_4 = 0.0f,  acc_5 = 0.0f,  acc_6 = 0.0f,  acc_7 = 0.0f;
    register float acc_8 = 0.0f,  acc_9 = 0.0f,  acc_10 = 0.0f, acc_11 = 0.0f;
    register float acc_12 = 0.0f, acc_13 = 0.0f, acc_14 = 0.0f, acc_15 = 0.0f;

    register float t_substep;
    register float eps_t_substep;

    // ---------- stage 1 ----------

    t_substep = t_step;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho_unrolled(
        rho_vec_0,  rho_vec_1,  rho_vec_2,  rho_vec_3,
        rho_vec_4,  rho_vec_5,  rho_vec_6,  rho_vec_7,
        rho_vec_8,  rho_vec_9,  rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

        k_cur_0,  k_cur_1,  k_cur_2,  k_cur_3,
        k_cur_4,  k_cur_5,  k_cur_6,  k_cur_7,
        k_cur_8,  k_cur_9,  k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 1 for k1
    acc_0  += k_cur_0;  acc_1  += k_cur_1;  acc_2  += k_cur_2;  acc_3  += k_cur_3;
    acc_4  += k_cur_4;  acc_5  += k_cur_5;  acc_6  += k_cur_6;  acc_7  += k_cur_7;
    acc_8  += k_cur_8;  acc_9  += k_cur_9;  acc_10 += k_cur_10; acc_11 += k_cur_11;
    acc_12 += k_cur_12; acc_13 += k_cur_13; acc_14 += k_cur_14; acc_15 += k_cur_15;

    // ---------- stage 2 ----------

    t_substep = t_step + 0.5f * dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho_unrolled(
        rho_vec_0  + 0.5f * dt * k_cur_0,  rho_vec_1  + 0.5f * dt * k_cur_1,
        rho_vec_2  + 0.5f * dt * k_cur_2,  rho_vec_3  + 0.5f * dt * k_cur_3,
        rho_vec_4  + 0.5f * dt * k_cur_4,  rho_vec_5  + 0.5f * dt * k_cur_5,
        rho_vec_6  + 0.5f * dt * k_cur_6,  rho_vec_7  + 0.5f * dt * k_cur_7,
        rho_vec_8  + 0.5f * dt * k_cur_8,  rho_vec_9  + 0.5f * dt * k_cur_9,
        rho_vec_10 + 0.5f * dt * k_cur_10, rho_vec_11 + 0.5f * dt * k_cur_11,
        rho_vec_12 + 0.5f * dt * k_cur_12, rho_vec_13 + 0.5f * dt * k_cur_13,
        rho_vec_14 + 0.5f * dt * k_cur_14, rho_vec_15 + 0.5f * dt * k_cur_15,

        k_cur_0,  k_cur_1,  k_cur_2,  k_cur_3,
        k_cur_4,  k_cur_5,  k_cur_6,  k_cur_7,
        k_cur_8,  k_cur_9,  k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 2 for k2
    acc_0  += 2.0f * k_cur_0;  acc_1  += 2.0f * k_cur_1;
    acc_2  += 2.0f * k_cur_2;  acc_3  += 2.0f * k_cur_3;
    acc_4  += 2.0f * k_cur_4;  acc_5  += 2.0f * k_cur_5;
    acc_6  += 2.0f * k_cur_6;  acc_7  += 2.0f * k_cur_7;
    acc_8  += 2.0f * k_cur_8;  acc_9  += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10; acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12; acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14; acc_15 += 2.0f * k_cur_15;

    // ---------- stage 3 ----------
    compute_drho_unrolled(
        rho_vec_0  + 0.5f * dt * k_cur_0,  rho_vec_1  + 0.5f * dt * k_cur_1,
        rho_vec_2  + 0.5f * dt * k_cur_2,  rho_vec_3  + 0.5f * dt * k_cur_3,
        rho_vec_4  + 0.5f * dt * k_cur_4,  rho_vec_5  + 0.5f * dt * k_cur_5,
        rho_vec_6  + 0.5f * dt * k_cur_6,  rho_vec_7  + 0.5f * dt * k_cur_7,
        rho_vec_8  + 0.5f * dt * k_cur_8,  rho_vec_9  + 0.5f * dt * k_cur_9,
        rho_vec_10 + 0.5f * dt * k_cur_10, rho_vec_11 + 0.5f * dt * k_cur_11,
        rho_vec_12 + 0.5f * dt * k_cur_12, rho_vec_13 + 0.5f * dt * k_cur_13,
        rho_vec_14 + 0.5f * dt * k_cur_14, rho_vec_15 + 0.5f * dt * k_cur_15,

        k_cur_0,  k_cur_1,  k_cur_2,  k_cur_3,
        k_cur_4,  k_cur_5,  k_cur_6,  k_cur_7,
        k_cur_8,  k_cur_9,  k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 2 for k3
    acc_0  += 2.0f * k_cur_0;  acc_1  += 2.0f * k_cur_1;
    acc_2  += 2.0f * k_cur_2;  acc_3  += 2.0f * k_cur_3;
    acc_4  += 2.0f * k_cur_4;  acc_5  += 2.0f * k_cur_5;
    acc_6  += 2.0f * k_cur_6;  acc_7  += 2.0f * k_cur_7;
    acc_8  += 2.0f * k_cur_8;  acc_9  += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10; acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12; acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14; acc_15 += 2.0f * k_cur_15;

    // ---------- stage 4 ----------
    t_substep = t_step + dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho_unrolled(
        rho_vec_0  + dt * k_cur_0,  rho_vec_1  + dt * k_cur_1,
        rho_vec_2  + dt * k_cur_2,  rho_vec_3  + dt * k_cur_3,
        rho_vec_4  + dt * k_cur_4,  rho_vec_5  + dt * k_cur_5,
        rho_vec_6  + dt * k_cur_6,  rho_vec_7  + dt * k_cur_7,
        rho_vec_8  + dt * k_cur_8,  rho_vec_9  + dt * k_cur_9,
        rho_vec_10 + dt * k_cur_10, rho_vec_11 + dt * k_cur_11,
        rho_vec_12 + dt * k_cur_12, rho_vec_13 + dt * k_cur_13,
        rho_vec_14 + dt * k_cur_14, rho_vec_15 + dt * k_cur_15,

        k_cur_0,  k_cur_1,  k_cur_2,  k_cur_3,
        k_cur_4,  k_cur_5,  k_cur_6,  k_cur_7,
        k_cur_8,  k_cur_9,  k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 1 for k4
    acc_0  += k_cur_0;  acc_1  += k_cur_1;  acc_2  += k_cur_2;  acc_3  += k_cur_3;
    acc_4  += k_cur_4;  acc_5  += k_cur_5;  acc_6  += k_cur_6;  acc_7  += k_cur_7;
    acc_8  += k_cur_8;  acc_9  += k_cur_9;  acc_10 += k_cur_10; acc_11 += k_cur_11;
    acc_12 += k_cur_12; acc_13 += k_cur_13; acc_14 += k_cur_14; acc_15 += k_cur_15;

    // ---------- final RK4 update ----------
    const float coeff = dt / 6.0f;

    rho_vec_0  += coeff * acc_0;  rho_vec_1  += coeff * acc_1;
    rho_vec_2  += coeff * acc_2;  rho_vec_3  += coeff * acc_3;
    rho_vec_4  += coeff * acc_4;  rho_vec_5  += coeff * acc_5;
    rho_vec_6  += coeff * acc_6;  rho_vec_7  += coeff * acc_7;
    rho_vec_8  += coeff * acc_8;  rho_vec_9  += coeff * acc_9;
    rho_vec_10 += coeff * acc_10; rho_vec_11 += coeff * acc_11;
    rho_vec_12 += coeff * acc_12; rho_vec_13 += coeff * acc_13;
    rho_vec_14 += coeff * acc_14; rho_vec_15 += coeff * acc_15;

}



__device__ __forceinline__
void rk4_step_unrolled_v2_safe_log(
    float& rho_vec_0,  float& rho_vec_1,  float& rho_vec_2,  float& rho_vec_3,
    float& rho_vec_4,  float& rho_vec_5,  float& rho_vec_6,  float& rho_vec_7,
    float& rho_vec_8,  float& rho_vec_9,  float& rho_vec_10, float& rho_vec_11,
    float& rho_vec_12, float& rho_vec_13, float& rho_vec_14, float& rho_vec_15,

    const float t_step,
    const float eps0,
    const float A,

    // log
    LogEntry* __restrict__ d_log_buffer,
    const int t_idx_step
) {

    int t_idx_substep;
    int substep_num;

    // derivative buffers
    register float k_cur_0,  k_cur_1,  k_cur_2,  k_cur_3;
    register float k_cur_4,  k_cur_5,  k_cur_6,  k_cur_7;
    register float k_cur_8,  k_cur_9,  k_cur_10, k_cur_11;
    register float k_cur_12, k_cur_13, k_cur_14, k_cur_15;

    // accumulators
    register float acc_0 = 0.0f,  acc_1 = 0.0f,  acc_2 = 0.0f,  acc_3 = 0.0f;
    register float acc_4 = 0.0f,  acc_5 = 0.0f,  acc_6 = 0.0f,  acc_7 = 0.0f;
    register float acc_8 = 0.0f,  acc_9 = 0.0f,  acc_10 = 0.0f, acc_11 = 0.0f;
    register float acc_12 = 0.0f, acc_13 = 0.0f, acc_14 = 0.0f, acc_15 = 0.0f;

    register float t_substep;
    register float eps_t_substep;

    // ---------- stage 1 ----------

    t_substep = t_step;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    substep_num = 0;
    t_idx_substep = t_idx_step * 4 + substep_num;

    compute_drho_unrolled_log(
        rho_vec_0,  rho_vec_1,  rho_vec_2,  rho_vec_3,
        rho_vec_4,  rho_vec_5,  rho_vec_6,  rho_vec_7,
        rho_vec_8,  rho_vec_9,  rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

        k_cur_0,  k_cur_1,  k_cur_2,  k_cur_3,
        k_cur_4,  k_cur_5,  k_cur_6,  k_cur_7,
        k_cur_8,  k_cur_9,  k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep,
        t_idx_step, substep_num, t_step, t_substep
    );

    // weight = 1 for k1
    acc_0 += k_cur_0;   acc_1 += k_cur_1;   acc_2 += k_cur_2;   acc_3 += k_cur_3;
    acc_4 += k_cur_4;   acc_5 += k_cur_5;   acc_6 += k_cur_6;   acc_7 += k_cur_7;
    acc_8 += k_cur_8;   acc_9 += k_cur_9;   acc_10 += k_cur_10; acc_11 += k_cur_11;
    acc_12 += k_cur_12; acc_13 += k_cur_13; acc_14 += k_cur_14; acc_15 += k_cur_15;

    // ---------- stage 2 ----------

    t_substep = t_step + 0.5f * dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    substep_num = 1;
    t_idx_substep = t_idx_step * 4 + substep_num;

    compute_drho_unrolled_log(
        rho_vec_0  + 0.5f * dt * k_cur_0,  rho_vec_1  + 0.5f * dt * k_cur_1,
        rho_vec_2  + 0.5f * dt * k_cur_2,  rho_vec_3  + 0.5f * dt * k_cur_3,
        rho_vec_4  + 0.5f * dt * k_cur_4,  rho_vec_5  + 0.5f * dt * k_cur_5,
        rho_vec_6  + 0.5f * dt * k_cur_6,  rho_vec_7  + 0.5f * dt * k_cur_7,
        rho_vec_8  + 0.5f * dt * k_cur_8,  rho_vec_9  + 0.5f * dt * k_cur_9,
        rho_vec_10 + 0.5f * dt * k_cur_10, rho_vec_11 + 0.5f * dt * k_cur_11,
        rho_vec_12 + 0.5f * dt * k_cur_12, rho_vec_13 + 0.5f * dt * k_cur_13,
        rho_vec_14 + 0.5f * dt * k_cur_14, rho_vec_15 + 0.5f * dt * k_cur_15,

        k_cur_0,  k_cur_1,  k_cur_2,  k_cur_3,
        k_cur_4,  k_cur_5,  k_cur_6,  k_cur_7,
        k_cur_8,  k_cur_9,  k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep,
        t_idx_step, substep_num, t_step, t_substep
    );

    // weight = 2 for k2
    acc_0  += 2.0f * k_cur_0;  acc_1  += 2.0f * k_cur_1;
    acc_2  += 2.0f * k_cur_2;  acc_3  += 2.0f * k_cur_3;
    acc_4  += 2.0f * k_cur_4;  acc_5  += 2.0f * k_cur_5;
    acc_6  += 2.0f * k_cur_6;  acc_7  += 2.0f * k_cur_7;
    acc_8  += 2.0f * k_cur_8;  acc_9  += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10; acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12; acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14; acc_15 += 2.0f * k_cur_15;

    // ---------- stage 3 ----------

    substep_num = 2;
    t_idx_substep = t_idx_step * 4 + substep_num;

    compute_drho_unrolled_log(
        rho_vec_0  + 0.5f * dt * k_cur_0,  rho_vec_1  + 0.5f * dt * k_cur_1,
        rho_vec_2  + 0.5f * dt * k_cur_2,  rho_vec_3  + 0.5f * dt * k_cur_3,
        rho_vec_4  + 0.5f * dt * k_cur_4,  rho_vec_5  + 0.5f * dt * k_cur_5,
        rho_vec_6  + 0.5f * dt * k_cur_6,  rho_vec_7  + 0.5f * dt * k_cur_7,
        rho_vec_8  + 0.5f * dt * k_cur_8,  rho_vec_9  + 0.5f * dt * k_cur_9,
        rho_vec_10 + 0.5f * dt * k_cur_10, rho_vec_11 + 0.5f * dt * k_cur_11,
        rho_vec_12 + 0.5f * dt * k_cur_12, rho_vec_13 + 0.5f * dt * k_cur_13,
        rho_vec_14 + 0.5f * dt * k_cur_14, rho_vec_15 + 0.5f * dt * k_cur_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep,
        t_idx_step, substep_num, t_step, t_substep
    );

    // weight = 2 for k3
    acc_0  += 2.0f * k_cur_0;  acc_1  += 2.0f * k_cur_1;
    acc_2  += 2.0f * k_cur_2;  acc_3  += 2.0f * k_cur_3;
    acc_4  += 2.0f * k_cur_4;  acc_5  += 2.0f * k_cur_5;
    acc_6  += 2.0f * k_cur_6;  acc_7  += 2.0f * k_cur_7;
    acc_8  += 2.0f * k_cur_8;  acc_9  += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10; acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12; acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14; acc_15 += 2.0f * k_cur_15;

    // ---------- stage 4 ----------
    t_substep = t_step + dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    substep_num = 3;
    t_idx_substep = t_idx_step * 4 + substep_num;

    compute_drho_unrolled_log(
        rho_vec_0  + dt * k_cur_0,  rho_vec_1  + dt * k_cur_1,
        rho_vec_2  + dt * k_cur_2,  rho_vec_3  + dt * k_cur_3,
        rho_vec_4  + dt * k_cur_4,  rho_vec_5  + dt * k_cur_5,
        rho_vec_6  + dt * k_cur_6,  rho_vec_7  + dt * k_cur_7,
        rho_vec_8  + dt * k_cur_8,  rho_vec_9  + dt * k_cur_9,
        rho_vec_10 + dt * k_cur_10, rho_vec_11 + dt * k_cur_11,
        rho_vec_12 + dt * k_cur_12, rho_vec_13 + dt * k_cur_13,
        rho_vec_14 + dt * k_cur_14, rho_vec_15 + dt * k_cur_15,

        k_cur_0,  k_cur_1,  k_cur_2,  k_cur_3,
        k_cur_4,  k_cur_5,  k_cur_6,  k_cur_7,
        k_cur_8,  k_cur_9,  k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep,
        t_idx_step, substep_num, t_step, t_substep
    );

    // weight = 1 for k4
    acc_0  += k_cur_0;  acc_1  += k_cur_1;  acc_2  += k_cur_2;  acc_3  += k_cur_3;
    acc_4  += k_cur_4;  acc_5  += k_cur_5;  acc_6  += k_cur_6;  acc_7  += k_cur_7;
    acc_8  += k_cur_8;  acc_9  += k_cur_9;  acc_10 += k_cur_10; acc_11 += k_cur_11;
    acc_12 += k_cur_12; acc_13 += k_cur_13; acc_14 += k_cur_14; acc_15 += k_cur_15;

    // ---------- final RK4 update ----------
    const float coeff = dt / 6.0f;

    rho_vec_0  += coeff * acc_0;  rho_vec_1  += coeff * acc_1;
    rho_vec_2  += coeff * acc_2;  rho_vec_3  += coeff * acc_3;
    rho_vec_4  += coeff * acc_4;  rho_vec_5  += coeff * acc_5;
    rho_vec_6  += coeff * acc_6;  rho_vec_7  += coeff * acc_7;
    rho_vec_8  += coeff * acc_8;  rho_vec_9  += coeff * acc_9;
    rho_vec_10 += coeff * acc_10; rho_vec_11 += coeff * acc_11;
    rho_vec_12 += coeff * acc_12; rho_vec_13 += coeff * acc_13;
    rho_vec_14 += coeff * acc_14; rho_vec_15 += coeff * acc_15;

}



__device__ __forceinline__
void rk4_step_unrolled_v3_safe(
    float& rho_vec_0, float& rho_vec_1, float& rho_vec_2, float& rho_vec_3,
    float& rho_vec_4, float& rho_vec_5, float& rho_vec_6, float& rho_vec_7,
    float& rho_vec_8, float& rho_vec_9, float& rho_vec_10, float& rho_vec_11,
    float& rho_vec_12, float& rho_vec_13, float& rho_vec_14, float& rho_vec_15,

    const float t_step,
    const float eps0,
    const float A
) {
    // derivative buffers
    register float k_cur_0,  k_cur_1,  k_cur_2,  k_cur_3;
    register float k_cur_4,  k_cur_5,  k_cur_6,  k_cur_7;
    register float k_cur_8,  k_cur_9,  k_cur_10, k_cur_11;
    register float k_cur_12, k_cur_13, k_cur_14, k_cur_15;

    // accumulators
    register float acc_0 = 0.0f,  acc_1 = 0.0f,  acc_2 = 0.0f,  acc_3 = 0.0f;
    register float acc_4 = 0.0f,  acc_5 = 0.0f,  acc_6 = 0.0f,  acc_7 = 0.0f;
    register float acc_8 = 0.0f,  acc_9 = 0.0f,  acc_10 = 0.0f, acc_11 = 0.0f;
    register float acc_12 = 0.0f, acc_13 = 0.0f, acc_14 = 0.0f, acc_15 = 0.0f;

    register float t_substep;
    register float eps_t_substep;

    // ---------- stage 1 ----------

    t_substep = t_step;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho_unrolled(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 1 for k1
    acc_0  += k_cur_0;  acc_1  += k_cur_1;  acc_2  += k_cur_2;  acc_3  += k_cur_3;
    acc_4  += k_cur_4;  acc_5  += k_cur_5;  acc_6  += k_cur_6;  acc_7  += k_cur_7;
    acc_8  += k_cur_8;  acc_9  += k_cur_9;  acc_10 += k_cur_10; acc_11 += k_cur_11;
    acc_12 += k_cur_12; acc_13 += k_cur_13; acc_14 += k_cur_14; acc_15 += k_cur_15;

    // ---------- stage 2 ----------

    t_substep = t_step + 0.5f * dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    float rho_tmp_0  = rho_vec_0  + 0.5f * dt * k_cur_0;
    float rho_tmp_1  = rho_vec_1  + 0.5f * dt * k_cur_1;
    float rho_tmp_2  = rho_vec_2  + 0.5f * dt * k_cur_2;
    float rho_tmp_3  = rho_vec_3  + 0.5f * dt * k_cur_3;
    float rho_tmp_4  = rho_vec_4  + 0.5f * dt * k_cur_4;
    float rho_tmp_5  = rho_vec_5  + 0.5f * dt * k_cur_5;
    float rho_tmp_6  = rho_vec_6  + 0.5f * dt * k_cur_6;
    float rho_tmp_7  = rho_vec_7  + 0.5f * dt * k_cur_7;
    float rho_tmp_8  = rho_vec_8  + 0.5f * dt * k_cur_8;
    float rho_tmp_9  = rho_vec_9  + 0.5f * dt * k_cur_9;
    float rho_tmp_10 = rho_vec_10 + 0.5f * dt * k_cur_10;
    float rho_tmp_11 = rho_vec_11 + 0.5f * dt * k_cur_11;
    float rho_tmp_12 = rho_vec_12 + 0.5f * dt * k_cur_12;
    float rho_tmp_13 = rho_vec_13 + 0.5f * dt * k_cur_13;
    float rho_tmp_14 = rho_vec_14 + 0.5f * dt * k_cur_14;
    float rho_tmp_15 = rho_vec_15 + 0.5f * dt * k_cur_15;

    compute_drho_unrolled(
        rho_tmp_0,  rho_tmp_1,  rho_tmp_2,  rho_tmp_3,
        rho_tmp_4,  rho_tmp_5,  rho_tmp_6,  rho_tmp_7,
        rho_tmp_8,  rho_tmp_9,  rho_tmp_10, rho_tmp_11,
        rho_tmp_12, rho_tmp_13, rho_tmp_14, rho_tmp_15,

        k_cur_0,  k_cur_1,  k_cur_2,  k_cur_3,
        k_cur_4,  k_cur_5,  k_cur_6,  k_cur_7,
        k_cur_8,  k_cur_9,  k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 2 for k2
    acc_0  += 2.0f * k_cur_0;  acc_1  += 2.0f * k_cur_1;
    acc_2  += 2.0f * k_cur_2;  acc_3  += 2.0f * k_cur_3;
    acc_4  += 2.0f * k_cur_4;  acc_5  += 2.0f * k_cur_5;
    acc_6  += 2.0f * k_cur_6;  acc_7  += 2.0f * k_cur_7;
    acc_8  += 2.0f * k_cur_8;  acc_9  += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10; acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12; acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14; acc_15 += 2.0f * k_cur_15;

    // ---------- stage 3 ----------

    rho_tmp_0  = rho_vec_0  + 0.5f * dt * k_cur_0;
    rho_tmp_1  = rho_vec_1  + 0.5f * dt * k_cur_1;
    rho_tmp_2  = rho_vec_2  + 0.5f * dt * k_cur_2;
    rho_tmp_3  = rho_vec_3  + 0.5f * dt * k_cur_3;
    rho_tmp_4  = rho_vec_4  + 0.5f * dt * k_cur_4;
    rho_tmp_5  = rho_vec_5  + 0.5f * dt * k_cur_5;
    rho_tmp_6  = rho_vec_6  + 0.5f * dt * k_cur_6;
    rho_tmp_7  = rho_vec_7  + 0.5f * dt * k_cur_7;
    rho_tmp_8  = rho_vec_8  + 0.5f * dt * k_cur_8;
    rho_tmp_9  = rho_vec_9  + 0.5f * dt * k_cur_9;
    rho_tmp_10 = rho_vec_10 + 0.5f * dt * k_cur_10;
    rho_tmp_11 = rho_vec_11 + 0.5f * dt * k_cur_11;
    rho_tmp_12 = rho_vec_12 + 0.5f * dt * k_cur_12;
    rho_tmp_13 = rho_vec_13 + 0.5f * dt * k_cur_13;
    rho_tmp_14 = rho_vec_14 + 0.5f * dt * k_cur_14;
    rho_tmp_15 = rho_vec_15 + 0.5f * dt * k_cur_15;

    compute_drho_unrolled(
        rho_tmp_0,  rho_tmp_1,  rho_tmp_2,  rho_tmp_3,
        rho_tmp_4,  rho_tmp_5,  rho_tmp_6,  rho_tmp_7,
        rho_tmp_8,  rho_tmp_9,  rho_tmp_10, rho_tmp_11,
        rho_tmp_12, rho_tmp_13, rho_tmp_14, rho_tmp_15,

        k_cur_0,  k_cur_1,  k_cur_2,  k_cur_3,
        k_cur_4,  k_cur_5,  k_cur_6,  k_cur_7,
        k_cur_8,  k_cur_9,  k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 2 for k3
    acc_0  += 2.0f * k_cur_0;  acc_1  += 2.0f * k_cur_1;
    acc_2  += 2.0f * k_cur_2;  acc_3  += 2.0f * k_cur_3;
    acc_4  += 2.0f * k_cur_4;  acc_5  += 2.0f * k_cur_5;
    acc_6  += 2.0f * k_cur_6;  acc_7  += 2.0f * k_cur_7;
    acc_8  += 2.0f * k_cur_8;  acc_9  += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10; acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12; acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14; acc_15 += 2.0f * k_cur_15;

    // ---------- stage 4 ----------
    t_substep = t_step + dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    rho_tmp_0  = rho_vec_0  + dt * k_cur_0;
    rho_tmp_1  = rho_vec_1  + dt * k_cur_1;
    rho_tmp_2  = rho_vec_2  + dt * k_cur_2;
    rho_tmp_3  = rho_vec_3  + dt * k_cur_3;
    rho_tmp_4  = rho_vec_4  + dt * k_cur_4;
    rho_tmp_5  = rho_vec_5  + dt * k_cur_5;
    rho_tmp_6  = rho_vec_6  + dt * k_cur_6;
    rho_tmp_7  = rho_vec_7  + dt * k_cur_7;
    rho_tmp_8  = rho_vec_8  + dt * k_cur_8;
    rho_tmp_9  = rho_vec_9  + dt * k_cur_9;
    rho_tmp_10 = rho_vec_10 + dt * k_cur_10;
    rho_tmp_11 = rho_vec_11 + dt * k_cur_11;
    rho_tmp_12 = rho_vec_12 + dt * k_cur_12;
    rho_tmp_13 = rho_vec_13 + dt * k_cur_13;
    rho_tmp_14 = rho_vec_14 + dt * k_cur_14;
    rho_tmp_15 = rho_vec_15 + dt * k_cur_15;

    compute_drho_unrolled(
        rho_tmp_0,  rho_tmp_1,  rho_tmp_2,  rho_tmp_3,
        rho_tmp_4,  rho_tmp_5,  rho_tmp_6,  rho_tmp_7,
        rho_tmp_8,  rho_tmp_9,  rho_tmp_10, rho_tmp_11,
        rho_tmp_12, rho_tmp_13, rho_tmp_14, rho_tmp_15,

        k_cur_0,  k_cur_1,  k_cur_2,  k_cur_3,
        k_cur_4,  k_cur_5,  k_cur_6,  k_cur_7,
        k_cur_8,  k_cur_9,  k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 1 for k4
    acc_0  += k_cur_0;  acc_1  += k_cur_1;  acc_2  += k_cur_2;  acc_3  += k_cur_3;
    acc_4  += k_cur_4;  acc_5  += k_cur_5;  acc_6  += k_cur_6;  acc_7  += k_cur_7;
    acc_8  += k_cur_8;  acc_9  += k_cur_9;  acc_10 += k_cur_10; acc_11 += k_cur_11;
    acc_12 += k_cur_12; acc_13 += k_cur_13; acc_14 += k_cur_14; acc_15 += k_cur_15;

    // ---------- final RK4 update ----------
    const float coeff = dt / 6.0f;

    rho_vec_0  += coeff * acc_0;  rho_vec_1  += coeff * acc_1;
    rho_vec_2  += coeff * acc_2;  rho_vec_3  += coeff * acc_3;
    rho_vec_4  += coeff * acc_4;  rho_vec_5  += coeff * acc_5;
    rho_vec_6  += coeff * acc_6;  rho_vec_7  += coeff * acc_7;
    rho_vec_8  += coeff * acc_8;  rho_vec_9  += coeff * acc_9;
    rho_vec_10 += coeff * acc_10; rho_vec_11 += coeff * acc_11;
    rho_vec_12 += coeff * acc_12; rho_vec_13 += coeff * acc_13;
    rho_vec_14 += coeff * acc_14; rho_vec_15 += coeff * acc_15;

}



__device__ __forceinline__
void rk4_step_unrolled_v3_safe_log(
    float& rho_vec_0, float& rho_vec_1, float& rho_vec_2, float& rho_vec_3,
    float& rho_vec_4, float& rho_vec_5, float& rho_vec_6, float& rho_vec_7,
    float& rho_vec_8, float& rho_vec_9, float& rho_vec_10, float& rho_vec_11,
    float& rho_vec_12, float& rho_vec_13, float& rho_vec_14, float& rho_vec_15,

    const float t_step,
    const float eps0,
    const float A,
    
    // log
    LogEntry* __restrict__ d_log_buffer,
    const int t_idx_step
) {

    int t_idx_substep;
    int substep_num;

    // derivative buffers
    register float k_cur_0, k_cur_1, k_cur_2, k_cur_3;
    register float k_cur_4, k_cur_5, k_cur_6, k_cur_7;
    register float k_cur_8, k_cur_9, k_cur_10, k_cur_11;
    register float k_cur_12, k_cur_13, k_cur_14, k_cur_15;

    // accumulators
    register float acc_0 = 0.0f, acc_1 = 0.0f, acc_2 = 0.0f, acc_3 = 0.0f;
    register float acc_4 = 0.0f, acc_5 = 0.0f, acc_6 = 0.0f, acc_7 = 0.0f;
    register float acc_8 = 0.0f, acc_9 = 0.0f, acc_10 = 0.0f, acc_11 = 0.0f;
    register float acc_12 = 0.0f, acc_13 = 0.0f, acc_14 = 0.0f, acc_15 = 0.0f;

    register float t_substep;
    register float eps_t_substep;

    // ---------- stage 1 ----------

    t_substep = t_step;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    substep_num = 0;
    t_idx_substep = t_idx_step * 4 + substep_num;


    compute_drho_unrolled_log(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep,
        t_idx_step, substep_num, t_step, t_substep
    );

    // weight = 1 for k1
    acc_0 += k_cur_0;  acc_1 += k_cur_1;  acc_2 += k_cur_2;  acc_3 += k_cur_3;
    acc_4 += k_cur_4;  acc_5 += k_cur_5;  acc_6 += k_cur_6;  acc_7 += k_cur_7;
    acc_8 += k_cur_8;  acc_9 += k_cur_9;  acc_10 += k_cur_10; acc_11 += k_cur_11;
    acc_12 += k_cur_12; acc_13 += k_cur_13; acc_14 += k_cur_14; acc_15 += k_cur_15;

    // ---------- stage 2 ----------

    t_substep = t_step + 0.5f * dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    float rho_tmp_0 = rho_vec_0 + 0.5f * dt * k_cur_0;
    float rho_tmp_1 = rho_vec_1 + 0.5f * dt * k_cur_1;
    float rho_tmp_2 = rho_vec_2 + 0.5f * dt * k_cur_2;
    float rho_tmp_3 = rho_vec_3 + 0.5f * dt * k_cur_3;
    float rho_tmp_4 = rho_vec_4 + 0.5f * dt * k_cur_4;
    float rho_tmp_5 = rho_vec_5 + 0.5f * dt * k_cur_5;
    float rho_tmp_6 = rho_vec_6 + 0.5f * dt * k_cur_6;
    float rho_tmp_7 = rho_vec_7 + 0.5f * dt * k_cur_7;
    float rho_tmp_8 = rho_vec_8 + 0.5f * dt * k_cur_8;
    float rho_tmp_9 = rho_vec_9 + 0.5f * dt * k_cur_9;
    float rho_tmp_10 = rho_vec_10 + 0.5f * dt * k_cur_10;
    float rho_tmp_11 = rho_vec_11 + 0.5f * dt * k_cur_11;
    float rho_tmp_12 = rho_vec_12 + 0.5f * dt * k_cur_12;
    float rho_tmp_13 = rho_vec_13 + 0.5f * dt * k_cur_13;
    float rho_tmp_14 = rho_vec_14 + 0.5f * dt * k_cur_14;
    float rho_tmp_15 = rho_vec_15 + 0.5f * dt * k_cur_15;

    substep_num = 1;
    t_idx_substep = t_idx_step * 4 + substep_num;

    compute_drho_unrolled_log(
        rho_tmp_0, rho_tmp_1, rho_tmp_2, rho_tmp_3,
        rho_tmp_4, rho_tmp_5, rho_tmp_6, rho_tmp_7,
        rho_tmp_8, rho_tmp_9, rho_tmp_10, rho_tmp_11,
        rho_tmp_12, rho_tmp_13, rho_tmp_14, rho_tmp_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep,
        t_idx_step, substep_num, t_step, t_substep
    );

    // weight = 2 for k2
    acc_0 += 2.0f * k_cur_0;  acc_1 += 2.0f * k_cur_1;
    acc_2 += 2.0f * k_cur_2;  acc_3 += 2.0f * k_cur_3;
    acc_4 += 2.0f * k_cur_4;  acc_5 += 2.0f * k_cur_5;
    acc_6 += 2.0f * k_cur_6;  acc_7 += 2.0f * k_cur_7;
    acc_8 += 2.0f * k_cur_8;  acc_9 += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10; acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12; acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14; acc_15 += 2.0f * k_cur_15;

    // ---------- stage 3 ----------

    rho_tmp_0 = rho_vec_0 + 0.5f * dt * k_cur_0;
    rho_tmp_1 = rho_vec_1 + 0.5f * dt * k_cur_1;
    rho_tmp_2 = rho_vec_2 + 0.5f * dt * k_cur_2;
    rho_tmp_3 = rho_vec_3 + 0.5f * dt * k_cur_3;
    rho_tmp_4 = rho_vec_4 + 0.5f * dt * k_cur_4;
    rho_tmp_5 = rho_vec_5 + 0.5f * dt * k_cur_5;
    rho_tmp_6 = rho_vec_6 + 0.5f * dt * k_cur_6;
    rho_tmp_7 = rho_vec_7 + 0.5f * dt * k_cur_7;
    rho_tmp_8 = rho_vec_8 + 0.5f * dt * k_cur_8;
    rho_tmp_9 = rho_vec_9 + 0.5f * dt * k_cur_9;
    rho_tmp_10 = rho_vec_10 + 0.5f * dt * k_cur_10;
    rho_tmp_11 = rho_vec_11 + 0.5f * dt * k_cur_11;
    rho_tmp_12 = rho_vec_12 + 0.5f * dt * k_cur_12;
    rho_tmp_13 = rho_vec_13 + 0.5f * dt * k_cur_13;
    rho_tmp_14 = rho_vec_14 + 0.5f * dt * k_cur_14;
    rho_tmp_15 = rho_vec_15 + 0.5f * dt * k_cur_15;

    substep_num = 2;
    t_idx_substep = t_idx_step * 4 + substep_num;

    compute_drho_unrolled_log(
        rho_tmp_0, rho_tmp_1, rho_tmp_2, rho_tmp_3,
        rho_tmp_4, rho_tmp_5, rho_tmp_6, rho_tmp_7,
        rho_tmp_8, rho_tmp_9, rho_tmp_10, rho_tmp_11,
        rho_tmp_12, rho_tmp_13, rho_tmp_14, rho_tmp_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep,
        t_idx_step, substep_num, t_step, t_substep
    );

    // weight = 2 for k3
    acc_0 += 2.0f * k_cur_0;  acc_1 += 2.0f * k_cur_1;
    acc_2 += 2.0f * k_cur_2;  acc_3 += 2.0f * k_cur_3;
    acc_4 += 2.0f * k_cur_4;  acc_5 += 2.0f * k_cur_5;
    acc_6 += 2.0f * k_cur_6;  acc_7 += 2.0f * k_cur_7;
    acc_8 += 2.0f * k_cur_8;  acc_9 += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10; acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12; acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14; acc_15 += 2.0f * k_cur_15;

    // ---------- stage 4 ----------
    t_substep = t_step + dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    rho_tmp_0 = rho_vec_0 + dt * k_cur_0;
    rho_tmp_1 = rho_vec_1 + dt * k_cur_1;
    rho_tmp_2 = rho_vec_2 + dt * k_cur_2;
    rho_tmp_3 = rho_vec_3 + dt * k_cur_3;
    rho_tmp_4 = rho_vec_4 + dt * k_cur_4;
    rho_tmp_5 = rho_vec_5 + dt * k_cur_5;
    rho_tmp_6 = rho_vec_6 + dt * k_cur_6;
    rho_tmp_7 = rho_vec_7 + dt * k_cur_7;
    rho_tmp_8 = rho_vec_8 + dt * k_cur_8;
    rho_tmp_9 = rho_vec_9 + dt * k_cur_9;
    rho_tmp_10 = rho_vec_10 + dt * k_cur_10;
    rho_tmp_11 = rho_vec_11 + dt * k_cur_11;
    rho_tmp_12 = rho_vec_12 + dt * k_cur_12;
    rho_tmp_13 = rho_vec_13 + dt * k_cur_13;
    rho_tmp_14 = rho_vec_14 + dt * k_cur_14;
    rho_tmp_15 = rho_vec_15 + dt * k_cur_15;

    substep_num = 3;
    t_idx_substep = t_idx_step * 4 + substep_num;

    compute_drho_unrolled_log(
        rho_tmp_0, rho_tmp_1, rho_tmp_2, rho_tmp_3,
        rho_tmp_4, rho_tmp_5, rho_tmp_6, rho_tmp_7,
        rho_tmp_8, rho_tmp_9, rho_tmp_10, rho_tmp_11,
        rho_tmp_12, rho_tmp_13, rho_tmp_14, rho_tmp_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep,
        t_idx_step, substep_num, t_step, t_substep
    );

    // weight = 1 for k4
    acc_0 += k_cur_0;  acc_1 += k_cur_1;  acc_2 += k_cur_2;  acc_3 += k_cur_3;
    acc_4 += k_cur_4;  acc_5 += k_cur_5;  acc_6 += k_cur_6;  acc_7 += k_cur_7;
    acc_8 += k_cur_8;  acc_9 += k_cur_9;  acc_10 += k_cur_10; acc_11 += k_cur_11;
    acc_12 += k_cur_12; acc_13 += k_cur_13; acc_14 += k_cur_14; acc_15 += k_cur_15;

    // ---------- final RK4 update ----------
    const float coeff = dt / 6.0f;

    rho_vec_0 += coeff * acc_0;  rho_vec_1 += coeff * acc_1;
    rho_vec_2 += coeff * acc_2;  rho_vec_3 += coeff * acc_3;
    rho_vec_4 += coeff * acc_4;  rho_vec_5 += coeff * acc_5;
    rho_vec_6 += coeff * acc_6;  rho_vec_7 += coeff * acc_7;
    rho_vec_8 += coeff * acc_8;  rho_vec_9 += coeff * acc_9;
    rho_vec_10 += coeff * acc_10; rho_vec_11 += coeff * acc_11;
    rho_vec_12 += coeff * acc_12; rho_vec_13 += coeff * acc_13;
    rho_vec_14 += coeff * acc_14; rho_vec_15 += coeff * acc_15;



}



__device__ __forceinline__
void rk4_step_unrolled_v3a_safe(
    float& rho_vec_0, float& rho_vec_1, float& rho_vec_2, float& rho_vec_3,
    float& rho_vec_4, float& rho_vec_5, float& rho_vec_6, float& rho_vec_7,
    float& rho_vec_8, float& rho_vec_9, float& rho_vec_10, float& rho_vec_11,
    float& rho_vec_12, float& rho_vec_13, float& rho_vec_14, float& rho_vec_15,

    const float t_step,
    const float eps0,
    const float A
) {
    // derivative buffers
    register float k_cur_0, k_cur_1, k_cur_2, k_cur_3;
    register float k_cur_4, k_cur_5, k_cur_6, k_cur_7;
    register float k_cur_8, k_cur_9, k_cur_10, k_cur_11;
    register float k_cur_12, k_cur_13, k_cur_14, k_cur_15;

    // accumulators
    register float acc_0, acc_1, acc_2, acc_3;
    register float acc_4, acc_5, acc_6, acc_7;
    register float acc_8, acc_9, acc_10, acc_11;
    register float acc_12, acc_13, acc_14, acc_15;

    register float t_substep;
    register float eps_t_substep;

    const register float dt_half = 0.5f * dt;

    // ---------- stage 1 ----------

    t_substep = t_step;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho_unrolled(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 1 for k1
    acc_0 = k_cur_0;  acc_1 = k_cur_1;  acc_2 = k_cur_2;  acc_3 = k_cur_3;
    acc_4 = k_cur_4;  acc_5 = k_cur_5;  acc_6 = k_cur_6;  acc_7 = k_cur_7;
    acc_8 = k_cur_8;  acc_9 = k_cur_9;  acc_10 = k_cur_10; acc_11 = k_cur_11;
    acc_12 = k_cur_12; acc_13 = k_cur_13; acc_14 = k_cur_14; acc_15 = k_cur_15;

    // ---------- stage 2 ----------

    t_substep = t_step + dt_half;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    float rho_tmp_0 = rho_vec_0 + dt_half * k_cur_0;
    float rho_tmp_1 = rho_vec_1 + dt_half * k_cur_1;
    float rho_tmp_2 = rho_vec_2 + dt_half * k_cur_2;
    float rho_tmp_3 = rho_vec_3 + dt_half * k_cur_3;
    float rho_tmp_4 = rho_vec_4 + dt_half * k_cur_4;
    float rho_tmp_5 = rho_vec_5 + dt_half * k_cur_5;
    float rho_tmp_6 = rho_vec_6 + dt_half * k_cur_6;
    float rho_tmp_7 = rho_vec_7 + dt_half * k_cur_7;
    float rho_tmp_8 = rho_vec_8 + dt_half * k_cur_8;
    float rho_tmp_9 = rho_vec_9 + dt_half * k_cur_9;
    float rho_tmp_10 = rho_vec_10 + dt_half * k_cur_10;
    float rho_tmp_11 = rho_vec_11 + dt_half * k_cur_11;
    float rho_tmp_12 = rho_vec_12 + dt_half * k_cur_12;
    float rho_tmp_13 = rho_vec_13 + dt_half * k_cur_13;
    float rho_tmp_14 = rho_vec_14 + dt_half * k_cur_14;
    float rho_tmp_15 = rho_vec_15 + dt_half * k_cur_15;

    compute_drho_unrolled(
        rho_tmp_0, rho_tmp_1, rho_tmp_2, rho_tmp_3,
        rho_tmp_4, rho_tmp_5, rho_tmp_6, rho_tmp_7,
        rho_tmp_8, rho_tmp_9, rho_tmp_10, rho_tmp_11,
        rho_tmp_12, rho_tmp_13, rho_tmp_14, rho_tmp_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 2 for k2
    acc_0 += 2.0f * k_cur_0;  acc_1 += 2.0f * k_cur_1;
    acc_2 += 2.0f * k_cur_2;  acc_3 += 2.0f * k_cur_3;
    acc_4 += 2.0f * k_cur_4;  acc_5 += 2.0f * k_cur_5;
    acc_6 += 2.0f * k_cur_6;  acc_7 += 2.0f * k_cur_7;
    acc_8 += 2.0f * k_cur_8;  acc_9 += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10; acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12; acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14; acc_15 += 2.0f * k_cur_15;

    // ---------- stage 3 ----------

    rho_tmp_0 = rho_vec_0 + dt_half * k_cur_0;
    rho_tmp_1 = rho_vec_1 + dt_half * k_cur_1;
    rho_tmp_2 = rho_vec_2 + dt_half * k_cur_2;
    rho_tmp_3 = rho_vec_3 + dt_half * k_cur_3;
    rho_tmp_4 = rho_vec_4 + dt_half * k_cur_4;
    rho_tmp_5 = rho_vec_5 + dt_half * k_cur_5;
    rho_tmp_6 = rho_vec_6 + dt_half * k_cur_6;
    rho_tmp_7 = rho_vec_7 + dt_half * k_cur_7;
    rho_tmp_8 = rho_vec_8 + dt_half * k_cur_8;
    rho_tmp_9 = rho_vec_9 + dt_half * k_cur_9;
    rho_tmp_10 = rho_vec_10 + dt_half * k_cur_10;
    rho_tmp_11 = rho_vec_11 + dt_half * k_cur_11;
    rho_tmp_12 = rho_vec_12 + dt_half * k_cur_12;
    rho_tmp_13 = rho_vec_13 + dt_half * k_cur_13;
    rho_tmp_14 = rho_vec_14 + dt_half * k_cur_14;
    rho_tmp_15 = rho_vec_15 + dt_half * k_cur_15;

    compute_drho_unrolled(
        rho_tmp_0, rho_tmp_1, rho_tmp_2, rho_tmp_3,
        rho_tmp_4, rho_tmp_5, rho_tmp_6, rho_tmp_7,
        rho_tmp_8, rho_tmp_9, rho_tmp_10, rho_tmp_11,
        rho_tmp_12, rho_tmp_13, rho_tmp_14, rho_tmp_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 2 for k3
    acc_0 += 2.0f * k_cur_0;  acc_1 += 2.0f * k_cur_1;
    acc_2 += 2.0f * k_cur_2;  acc_3 += 2.0f * k_cur_3;
    acc_4 += 2.0f * k_cur_4;  acc_5 += 2.0f * k_cur_5;
    acc_6 += 2.0f * k_cur_6;  acc_7 += 2.0f * k_cur_7;
    acc_8 += 2.0f * k_cur_8;  acc_9 += 2.0f * k_cur_9;
    acc_10 += 2.0f * k_cur_10; acc_11 += 2.0f * k_cur_11;
    acc_12 += 2.0f * k_cur_12; acc_13 += 2.0f * k_cur_13;
    acc_14 += 2.0f * k_cur_14; acc_15 += 2.0f * k_cur_15;

    // ---------- stage 4 ----------
    t_substep = t_step + dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    rho_tmp_0 = rho_vec_0 + dt * k_cur_0;
    rho_tmp_1 = rho_vec_1 + dt * k_cur_1;
    rho_tmp_2 = rho_vec_2 + dt * k_cur_2;
    rho_tmp_3 = rho_vec_3 + dt * k_cur_3;
    rho_tmp_4 = rho_vec_4 + dt * k_cur_4;
    rho_tmp_5 = rho_vec_5 + dt * k_cur_5;
    rho_tmp_6 = rho_vec_6 + dt * k_cur_6;
    rho_tmp_7 = rho_vec_7 + dt * k_cur_7;
    rho_tmp_8 = rho_vec_8 + dt * k_cur_8;
    rho_tmp_9 = rho_vec_9 + dt * k_cur_9;
    rho_tmp_10 = rho_vec_10 + dt * k_cur_10;
    rho_tmp_11 = rho_vec_11 + dt * k_cur_11;
    rho_tmp_12 = rho_vec_12 + dt * k_cur_12;
    rho_tmp_13 = rho_vec_13 + dt * k_cur_13;
    rho_tmp_14 = rho_vec_14 + dt * k_cur_14;
    rho_tmp_15 = rho_vec_15 + dt * k_cur_15;

    compute_drho_unrolled(
        rho_tmp_0, rho_tmp_1, rho_tmp_2, rho_tmp_3,
        rho_tmp_4, rho_tmp_5, rho_tmp_6, rho_tmp_7,
        rho_tmp_8, rho_tmp_9, rho_tmp_10, rho_tmp_11,
        rho_tmp_12, rho_tmp_13, rho_tmp_14, rho_tmp_15,

        k_cur_0, k_cur_1, k_cur_2, k_cur_3,
        k_cur_4, k_cur_5, k_cur_6, k_cur_7,
        k_cur_8, k_cur_9, k_cur_10, k_cur_11,
        k_cur_12, k_cur_13, k_cur_14, k_cur_15,

        eps_t_substep
    );

    // weight = 1 for k4
    acc_0 += k_cur_0;  acc_1 += k_cur_1;  acc_2 += k_cur_2;  acc_3 += k_cur_3;
    acc_4 += k_cur_4;  acc_5 += k_cur_5;  acc_6 += k_cur_6;  acc_7 += k_cur_7;
    acc_8 += k_cur_8;  acc_9 += k_cur_9;  acc_10 += k_cur_10; acc_11 += k_cur_11;
    acc_12 += k_cur_12; acc_13 += k_cur_13; acc_14 += k_cur_14; acc_15 += k_cur_15;

    // ---------- final RK4 update ----------
    const float coeff = dt / 6.0f;

    rho_vec_0 += coeff * acc_0;  rho_vec_1 += coeff * acc_1;
    rho_vec_2 += coeff * acc_2;  rho_vec_3 += coeff * acc_3;
    rho_vec_4 += coeff * acc_4;  rho_vec_5 += coeff * acc_5;
    rho_vec_6 += coeff * acc_6;  rho_vec_7 += coeff * acc_7;
    rho_vec_8 += coeff * acc_8;  rho_vec_9 += coeff * acc_9;
    rho_vec_10 += coeff * acc_10; rho_vec_11 += coeff * acc_11;
    rho_vec_12 += coeff * acc_12; rho_vec_13 += coeff * acc_13;
    rho_vec_14 += coeff * acc_14; rho_vec_15 += coeff * acc_15;

}




__device__ __forceinline__
void rk4_step_unrolled_v4(
    float& rho_vec_0, float& rho_vec_1, float& rho_vec_2, float& rho_vec_3,
    float& rho_vec_4, float& rho_vec_5, float& rho_vec_6, float& rho_vec_7,
    float& rho_vec_8, float& rho_vec_9, float& rho_vec_10, float& rho_vec_11,
    float& rho_vec_12, float& rho_vec_13, float& rho_vec_14, float& rho_vec_15,

    const float t_step,
    const float eps0,
    const float A
) {
    // Use only 8 temporary registers instead of 16
    // We'll reuse these for rho_tmp in each stage
    register float tmp_0, tmp_1, tmp_2, tmp_3;
    register float tmp_4, tmp_5, tmp_6, tmp_7;

    // Accumulator for weighted k values (k1 + 2*k2 + 2*k3 + k4)
    register float acc_0 = 0.0f, acc_1 = 0.0f, acc_2 = 0.0f, acc_3 = 0.0f;
    register float acc_4 = 0.0f, acc_5 = 0.0f, acc_6 = 0.0f, acc_7 = 0.0f;
    register float acc_8 = 0.0f, acc_9 = 0.0f, acc_10 = 0.0f, acc_11 = 0.0f;
    register float acc_12 = 0.0f, acc_13 = 0.0f, acc_14 = 0.0f, acc_15 = 0.0f;

    register float t_substep, eps_t_substep;
    const register float dt_half = 0.5f * dt;

    // ---------- Stage 1: k1 ----------
    t_substep = t_step;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    // Compute k1 directly into tmp variables (using them as k_cur)
    compute_drho_unrolled(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,
        tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7,
        acc_8, acc_9, acc_10, acc_11, acc_12, acc_13, acc_14, acc_15,
        eps_t_substep
    );

    // Accumulate k1 with weight 1, immediately overwrite tmp with rho + 0.5*dt*k1
    acc_0 = tmp_0; tmp_0 = rho_vec_0 + dt_half * tmp_0;
    acc_1 = tmp_1; tmp_1 = rho_vec_1 + dt_half * tmp_1;
    acc_2 = tmp_2; tmp_2 = rho_vec_2 + dt_half * tmp_2;
    acc_3 = tmp_3; tmp_3 = rho_vec_3 + dt_half * tmp_3;
    acc_4 = tmp_4; tmp_4 = rho_vec_4 + dt_half * tmp_4;
    acc_5 = tmp_5; tmp_5 = rho_vec_5 + dt_half * tmp_5;
    acc_6 = tmp_6; tmp_6 = rho_vec_6 + dt_half * tmp_6;
    acc_7 = tmp_7; tmp_7 = rho_vec_7 + dt_half * tmp_7;
    // acc_8-15 already have k1 values

    // ---------- Stage 2: k2 ----------
    t_substep = t_step + dt_half;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    // Reuse same registers for the next 8 temporaries
    register float tmp_8, tmp_9, tmp_10, tmp_11, tmp_12, tmp_13, tmp_14, tmp_15;
    tmp_8 = rho_vec_8 + dt_half * acc_8;
    tmp_9 = rho_vec_9 + dt_half * acc_9;
    tmp_10 = rho_vec_10 + dt_half * acc_10;
    tmp_11 = rho_vec_11 + dt_half * acc_11;
    tmp_12 = rho_vec_12 + dt_half * acc_12;
    tmp_13 = rho_vec_13 + dt_half * acc_13;
    tmp_14 = rho_vec_14 + dt_half * acc_14;
    tmp_15 = rho_vec_15 + dt_half * acc_15;

    compute_drho_unrolled(
        tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11, tmp_12, tmp_13, tmp_14, tmp_15,
        tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11, tmp_12, tmp_13, tmp_14, tmp_15,
        eps_t_substep
    );

    // Accumulate k2 with weight 2, prepare for stage 3
    acc_0 += 2.0f * tmp_0; tmp_0 = rho_vec_0 + dt_half * tmp_0;
    acc_1 += 2.0f * tmp_1; tmp_1 = rho_vec_1 + dt_half * tmp_1;
    acc_2 += 2.0f * tmp_2; tmp_2 = rho_vec_2 + dt_half * tmp_2;
    acc_3 += 2.0f * tmp_3; tmp_3 = rho_vec_3 + dt_half * tmp_3;
    acc_4 += 2.0f * tmp_4; tmp_4 = rho_vec_4 + dt_half * tmp_4;
    acc_5 += 2.0f * tmp_5; tmp_5 = rho_vec_5 + dt_half * tmp_5;
    acc_6 += 2.0f * tmp_6; tmp_6 = rho_vec_6 + dt_half * tmp_6;
    acc_7 += 2.0f * tmp_7; tmp_7 = rho_vec_7 + dt_half * tmp_7;
    acc_8 += 2.0f * tmp_8; tmp_8 = rho_vec_8 + dt_half * tmp_8;
    acc_9 += 2.0f * tmp_9; tmp_9 = rho_vec_9 + dt_half * tmp_9;
    acc_10 += 2.0f * tmp_10; tmp_10 = rho_vec_10 + dt_half * tmp_10;
    acc_11 += 2.0f * tmp_11; tmp_11 = rho_vec_11 + dt_half * tmp_11;
    acc_12 += 2.0f * tmp_12; tmp_12 = rho_vec_12 + dt_half * tmp_12;
    acc_13 += 2.0f * tmp_13; tmp_13 = rho_vec_13 + dt_half * tmp_13;
    acc_14 += 2.0f * tmp_14; tmp_14 = rho_vec_14 + dt_half * tmp_14;
    acc_15 += 2.0f * tmp_15; tmp_15 = rho_vec_15 + dt_half * tmp_15;

    // ---------- Stage 3: k3 ----------
    // t_substep stays the same (t_step + dt_half)

    compute_drho_unrolled(
        tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11, tmp_12, tmp_13, tmp_14, tmp_15,
        tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11, tmp_12, tmp_13, tmp_14, tmp_15,
        eps_t_substep
    );

    // Accumulate k3 with weight 2, prepare for stage 4
    acc_0 += 2.0f * tmp_0; tmp_0 = rho_vec_0 + dt * tmp_0;
    acc_1 += 2.0f * tmp_1; tmp_1 = rho_vec_1 + dt * tmp_1;
    acc_2 += 2.0f * tmp_2; tmp_2 = rho_vec_2 + dt * tmp_2;
    acc_3 += 2.0f * tmp_3; tmp_3 = rho_vec_3 + dt * tmp_3;
    acc_4 += 2.0f * tmp_4; tmp_4 = rho_vec_4 + dt * tmp_4;
    acc_5 += 2.0f * tmp_5; tmp_5 = rho_vec_5 + dt * tmp_5;
    acc_6 += 2.0f * tmp_6; tmp_6 = rho_vec_6 + dt * tmp_6;
    acc_7 += 2.0f * tmp_7; tmp_7 = rho_vec_7 + dt * tmp_7;
    acc_8 += 2.0f * tmp_8; tmp_8 = rho_vec_8 + dt * tmp_8;
    acc_9 += 2.0f * tmp_9; tmp_9 = rho_vec_9 + dt * tmp_9;
    acc_10 += 2.0f * tmp_10; tmp_10 = rho_vec_10 + dt * tmp_10;
    acc_11 += 2.0f * tmp_11; tmp_11 = rho_vec_11 + dt * tmp_11;
    acc_12 += 2.0f * tmp_12; tmp_12 = rho_vec_12 + dt * tmp_12;
    acc_13 += 2.0f * tmp_13; tmp_13 = rho_vec_13 + dt * tmp_13;
    acc_14 += 2.0f * tmp_14; tmp_14 = rho_vec_14 + dt * tmp_14;
    acc_15 += 2.0f * tmp_15; tmp_15 = rho_vec_15 + dt * tmp_15;

    // ---------- Stage 4: k4 ----------
    t_substep = t_step + dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho_unrolled(
        tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11, tmp_12, tmp_13, tmp_14, tmp_15,
        tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11, tmp_12, tmp_13, tmp_14, tmp_15,
        eps_t_substep
    );

    // Final accumulation and update
    const float coeff = dt / 6.0f;

    rho_vec_0 += coeff * (acc_0 + tmp_0);
    rho_vec_1 += coeff * (acc_1 + tmp_1);
    rho_vec_2 += coeff * (acc_2 + tmp_2);
    rho_vec_3 += coeff * (acc_3 + tmp_3);
    rho_vec_4 += coeff * (acc_4 + tmp_4);
    rho_vec_5 += coeff * (acc_5 + tmp_5);
    rho_vec_6 += coeff * (acc_6 + tmp_6);
    rho_vec_7 += coeff * (acc_7 + tmp_7);
    rho_vec_8 += coeff * (acc_8 + tmp_8);
    rho_vec_9 += coeff * (acc_9 + tmp_9);
    rho_vec_10 += coeff * (acc_10 + tmp_10);
    rho_vec_11 += coeff * (acc_11 + tmp_11);
    rho_vec_12 += coeff * (acc_12 + tmp_12);
    rho_vec_13 += coeff * (acc_13 + tmp_13);
    rho_vec_14 += coeff * (acc_14 + tmp_14);
    rho_vec_15 += coeff * (acc_15 + tmp_15);
}


__device__ __forceinline__
void rk4_step_unrolled_v5(
    float& rho_vec_0, float& rho_vec_1, float& rho_vec_2, float& rho_vec_3,
    float& rho_vec_4, float& rho_vec_5, float& rho_vec_6, float& rho_vec_7,
    float& rho_vec_8, float& rho_vec_9, float& rho_vec_10, float& rho_vec_11,
    float& rho_vec_12, float& rho_vec_13, float& rho_vec_14, float& rho_vec_15,

    const float t_step,
    const float eps0,
    const float A
) {
    // One set of temporaries reused throughout (k_cur, rho_tmp, etc.)
    float tmp_0, tmp_1, tmp_2, tmp_3;
    float tmp_4, tmp_5, tmp_6, tmp_7;
    float tmp_8, tmp_9, tmp_10, tmp_11;
    float tmp_12, tmp_13, tmp_14, tmp_15;

    // Accumulators for final result (k1 + 2*k2 + 2*k3 + k4)
    float acc_0 = 0.0f, acc_1 = 0.0f, acc_2 = 0.0f, acc_3 = 0.0f;
    float acc_4 = 0.0f, acc_5 = 0.0f, acc_6 = 0.0f, acc_7 = 0.0f;
    float acc_8 = 0.0f, acc_9 = 0.0f, acc_10 = 0.0f, acc_11 = 0.0f;
    float acc_12 = 0.0f, acc_13 = 0.0f, acc_14 = 0.0f, acc_15 = 0.0f;

    const float dt_half = 0.5f * dt;
    float t_substep, eps_t_substep;

    // ---------- Stage 1: k1 ----------
    t_substep = t_step;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho_unrolled(
        rho_vec_0, rho_vec_1, rho_vec_2, rho_vec_3,
        rho_vec_4, rho_vec_5, rho_vec_6, rho_vec_7,
        rho_vec_8, rho_vec_9, rho_vec_10, rho_vec_11,
        rho_vec_12, rho_vec_13, rho_vec_14, rho_vec_15,

        tmp_0, tmp_1, tmp_2, tmp_3,
        tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11,
        tmp_12, tmp_13, tmp_14, tmp_15,

        eps_t_substep
    );

    // Store k1 in acc, also compute rho + dt/2 * k1 in-place in tmp
    acc_0 = tmp_0; tmp_0 = rho_vec_0 + dt_half * tmp_0;
    acc_1 = tmp_1; tmp_1 = rho_vec_1 + dt_half * tmp_1;
    acc_2 = tmp_2; tmp_2 = rho_vec_2 + dt_half * tmp_2;
    acc_3 = tmp_3; tmp_3 = rho_vec_3 + dt_half * tmp_3;
    acc_4 = tmp_4; tmp_4 = rho_vec_4 + dt_half * tmp_4;
    acc_5 = tmp_5; tmp_5 = rho_vec_5 + dt_half * tmp_5;
    acc_6 = tmp_6; tmp_6 = rho_vec_6 + dt_half * tmp_6;
    acc_7 = tmp_7; tmp_7 = rho_vec_7 + dt_half * tmp_7;
    acc_8 = tmp_8; tmp_8 = rho_vec_8 + dt_half * tmp_8;
    acc_9 = tmp_9; tmp_9 = rho_vec_9 + dt_half * tmp_9;
    acc_10 = tmp_10; tmp_10 = rho_vec_10 + dt_half * tmp_10;
    acc_11 = tmp_11; tmp_11 = rho_vec_11 + dt_half * tmp_11;
    acc_12 = tmp_12; tmp_12 = rho_vec_12 + dt_half * tmp_12;
    acc_13 = tmp_13; tmp_13 = rho_vec_13 + dt_half * tmp_13;
    acc_14 = tmp_14; tmp_14 = rho_vec_14 + dt_half * tmp_14;
    acc_15 = tmp_15; tmp_15 = rho_vec_15 + dt_half * tmp_15;

    // ---------- Stage 2: k2 ----------
    t_substep = t_step + dt_half;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho_unrolled(
        tmp_0, tmp_1, tmp_2, tmp_3,
        tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11,
        tmp_12, tmp_13, tmp_14, tmp_15,

        tmp_0, tmp_1, tmp_2, tmp_3,
        tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11,
        tmp_12, tmp_13, tmp_14, tmp_15,

        eps_t_substep
    );

    acc_0 += 2.0f * tmp_0; tmp_0 = rho_vec_0 + dt_half * tmp_0;
    acc_1 += 2.0f * tmp_1; tmp_1 = rho_vec_1 + dt_half * tmp_1;
    acc_2 += 2.0f * tmp_2; tmp_2 = rho_vec_2 + dt_half * tmp_2;
    acc_3 += 2.0f * tmp_3; tmp_3 = rho_vec_3 + dt_half * tmp_3;
    acc_4 += 2.0f * tmp_4; tmp_4 = rho_vec_4 + dt_half * tmp_4;
    acc_5 += 2.0f * tmp_5; tmp_5 = rho_vec_5 + dt_half * tmp_5;
    acc_6 += 2.0f * tmp_6; tmp_6 = rho_vec_6 + dt_half * tmp_6;
    acc_7 += 2.0f * tmp_7; tmp_7 = rho_vec_7 + dt_half * tmp_7;
    acc_8 += 2.0f * tmp_8; tmp_8 = rho_vec_8 + dt_half * tmp_8;
    acc_9 += 2.0f * tmp_9; tmp_9 = rho_vec_9 + dt_half * tmp_9;
    acc_10 += 2.0f * tmp_10; tmp_10 = rho_vec_10 + dt_half * tmp_10;
    acc_11 += 2.0f * tmp_11; tmp_11 = rho_vec_11 + dt_half * tmp_11;
    acc_12 += 2.0f * tmp_12; tmp_12 = rho_vec_12 + dt_half * tmp_12;
    acc_13 += 2.0f * tmp_13; tmp_13 = rho_vec_13 + dt_half * tmp_13;
    acc_14 += 2.0f * tmp_14; tmp_14 = rho_vec_14 + dt_half * tmp_14;
    acc_15 += 2.0f * tmp_15; tmp_15 = rho_vec_15 + dt_half * tmp_15;

    // ---------- Stage 3: k3 ----------
    compute_drho_unrolled(
        tmp_0, tmp_1, tmp_2, tmp_3,
        tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11,
        tmp_12, tmp_13, tmp_14, tmp_15,

        tmp_0, tmp_1, tmp_2, tmp_3,
        tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11,
        tmp_12, tmp_13, tmp_14, tmp_15,

        eps_t_substep
    );

    acc_0 += 2.0f * tmp_0; tmp_0 = rho_vec_0 + dt * tmp_0;
    acc_1 += 2.0f * tmp_1; tmp_1 = rho_vec_1 + dt * tmp_1;
    acc_2 += 2.0f * tmp_2; tmp_2 = rho_vec_2 + dt * tmp_2;
    acc_3 += 2.0f * tmp_3; tmp_3 = rho_vec_3 + dt * tmp_3;
    acc_4 += 2.0f * tmp_4; tmp_4 = rho_vec_4 + dt * tmp_4;
    acc_5 += 2.0f * tmp_5; tmp_5 = rho_vec_5 + dt * tmp_5;
    acc_6 += 2.0f * tmp_6; tmp_6 = rho_vec_6 + dt * tmp_6;
    acc_7 += 2.0f * tmp_7; tmp_7 = rho_vec_7 + dt * tmp_7;
    acc_8 += 2.0f * tmp_8; tmp_8 = rho_vec_8 + dt * tmp_8;
    acc_9 += 2.0f * tmp_9; tmp_9 = rho_vec_9 + dt * tmp_9;
    acc_10 += 2.0f * tmp_10; tmp_10 = rho_vec_10 + dt * tmp_10;
    acc_11 += 2.0f * tmp_11; tmp_11 = rho_vec_11 + dt * tmp_11;
    acc_12 += 2.0f * tmp_12; tmp_12 = rho_vec_12 + dt * tmp_12;
    acc_13 += 2.0f * tmp_13; tmp_13 = rho_vec_13 + dt * tmp_13;
    acc_14 += 2.0f * tmp_14; tmp_14 = rho_vec_14 + dt * tmp_14;
    acc_15 += 2.0f * tmp_15; tmp_15 = rho_vec_15 + dt * tmp_15;

    // ---------- Stage 4: k4 ----------
    t_substep = t_step + dt;
    eps_t_substep = eps0 + A * cosf(omega * t_substep);

    compute_drho_unrolled(
        tmp_0, tmp_1, tmp_2, tmp_3,
        tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11,
        tmp_12, tmp_13, tmp_14, tmp_15,

        tmp_0, tmp_1, tmp_2, tmp_3,
        tmp_4, tmp_5, tmp_6, tmp_7,
        tmp_8, tmp_9, tmp_10, tmp_11,
        tmp_12, tmp_13, tmp_14, tmp_15,

        eps_t_substep
    );

    // Final update: rho += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    const float coeff = dt / 6.0f;

    rho_vec_0 += coeff * (acc_0 + tmp_0);
    rho_vec_1 += coeff * (acc_1 + tmp_1);
    rho_vec_2 += coeff * (acc_2 + tmp_2);
    rho_vec_3 += coeff * (acc_3 + tmp_3);
    rho_vec_4 += coeff * (acc_4 + tmp_4);
    rho_vec_5 += coeff * (acc_5 + tmp_5);
    rho_vec_6 += coeff * (acc_6 + tmp_6);
    rho_vec_7 += coeff * (acc_7 + tmp_7);
    rho_vec_8 += coeff * (acc_8 + tmp_8);
    rho_vec_9 += coeff * (acc_9 + tmp_9);
    rho_vec_10 += coeff * (acc_10 + tmp_10);
    rho_vec_11 += coeff * (acc_11 + tmp_11);
    rho_vec_12 += coeff * (acc_12 + tmp_12);
    rho_vec_13 += coeff * (acc_13 + tmp_13);
    rho_vec_14 += coeff * (acc_14 + tmp_14);
    rho_vec_15 += coeff * (acc_15 + tmp_15);
}



