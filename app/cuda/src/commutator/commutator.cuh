////////////////////////////////////////
// app/cuda/src/commutator/commutator.cuh
////////////////////////////////////////

#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "constants.cuh"

/*
// probably wrong sing
static __device__ __forceinline__
void commutator_v2_unrolled(
    const float rho_in_0, const float rho_in_1, const float rho_in_2,
    const float rho_in_3, const float rho_in_4, const float rho_in_5,
    const float rho_in_6, const float rho_in_7, const float rho_in_8,
    const float rho_in_9, const float rho_in_10, const float rho_in_11,
    const float rho_in_12, const float rho_in_13, const float rho_in_14,
    const float rho_in_15,

    float& drho_out_0, float& drho_out_1, float& drho_out_2,
    float& drho_out_3, float& drho_out_4, float& drho_out_5,
    float& drho_out_6, float& drho_out_7, float& drho_out_8,
    float& drho_out_9, float& drho_out_10, float& drho_out_11,
    float& drho_out_12, float& drho_out_13, float& drho_out_14,
    float& drho_out_15,

    const float eps_t_substep
)
{

    drho_out_0 = 0.f;
    drho_out_1 = 0.f;
    drho_out_2 = 0.f;
    drho_out_3 = 0.f;
    drho_out_4 = 0.f;
    drho_out_5 = 0.f;
    drho_out_6 = 0.f;
    drho_out_7 = 0.f;
    drho_out_8 = 0.f;
    drho_out_9 = 0.f;
    drho_out_10 = 0.f;
    drho_out_11 = 0.f;
    drho_out_12 = 0.f;
    drho_out_13 = 0.f;
    drho_out_14 = 0.f;
    drho_out_15 = 0.f;


    // --- load drho_out into registers (accumulators) ---
    register float tmp;
    //const float H03 = 0.0f;

    // --- Commutator contribution (register-accumulated, one-term-per-line) ---

    // Diagonals
    tmp = 0.0f;
    tmp += -2.f * pi_alpha_delta_R * rho_in_5;
    tmp += -2.f * pi_alpha_delta_L * rho_in_7;
    //tmp += -2.f * H03 * rho_in_9;
    drho_out_0 = tmp;


    tmp = 0.0f;
    tmp += -2.f * -pi_alpha_delta_R * rho_in_5;
    tmp += -2.f * pi_alpha_delta_C * rho_in_11;
    tmp += -2.f * pi_alpha_delta_L * rho_in_13;
    drho_out_1 = tmp;

    tmp = 0.0f;
    tmp += -2.f * -pi_alpha_delta_L * rho_in_7;
    tmp += -2.f * -pi_alpha_delta_C * rho_in_11;
    tmp += -2.f * pi_alpha_delta_R * rho_in_15;
    drho_out_2 = tmp;

    tmp = 0.0f;
    //tmp += -2.f * -H03 * rho_in_9;
    tmp += -2.f * -pi_alpha_delta_L * rho_in_13;
    tmp += -2.f * -pi_alpha_delta_R * rho_in_15;
    drho_out_3 = tmp;

    // (0,1)
    tmp = 0.0f;
    tmp += pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in_5;
    tmp += -pi_alpha * eps_t_substep * rho_in_5;
    tmp += -pi_alpha_delta_L * rho_in_11;
    tmp += -pi_alpha_delta_C * rho_in_7;
    //tmp += -H03 * rho_in_13;
    tmp += -pi_alpha_delta_L * rho_in_9;
    drho_out_4 = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in_4;
    tmp += pi_alpha_delta_R * rho_in_0;
    tmp += -pi_alpha_delta_R * rho_in_1;
    tmp += -pi_alpha_delta_L * rho_in_10;
    //tmp += -H03 * rho_in_12;
    tmp += pi_alpha * eps_t_substep * rho_in_4;
    tmp += pi_alpha_delta_C * rho_in_6;
    tmp += pi_alpha_delta_L * rho_in_8;
    drho_out_5 = tmp;

    // (0,2)
    tmp = 0.0f;
    tmp += pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in_7;
    tmp += -(-pi_alpha * eps_t_substep) * rho_in_7;
    tmp += pi_alpha_delta_R * rho_in_11;
    tmp += -pi_alpha_delta_C * rho_in_5;
    //tmp += -H03 * rho_in_15;
    tmp += -pi_alpha_delta_R * rho_in_9;
    drho_out_6 = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in_6;
    tmp += -pi_alpha_delta_R * rho_in_10;
    tmp += pi_alpha_delta_L * rho_in_0;
    tmp += -pi_alpha_delta_L * rho_in_2;
    //tmp += -H03 * rho_in_14;
    tmp += -pi_alpha * eps_t_substep * rho_in_6;
    tmp += pi_alpha_delta_C * rho_in_4;
    tmp += pi_alpha_delta_R * rho_in_8;
    drho_out_7 = tmp;

    // (0,3)
    tmp = 0.0f;
    tmp += pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in_9;
    tmp += -pi_alpha * (B * eps_t_substep - one_div_m) * rho_in_9;
    tmp += pi_alpha_delta_R * rho_in_13;
    tmp += pi_alpha_delta_L * rho_in_15;
    tmp += -pi_alpha_delta_L * rho_in_5;
    tmp += -pi_alpha_delta_R * rho_in_7;
    drho_out_8 = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in_8;
    tmp += -pi_alpha_delta_R * rho_in_12;
    tmp += -pi_alpha_delta_L * rho_in_14;
    //tmp += H03 * rho_in_0;
    //tmp += -H03 * rho_in_3;
    tmp += pi_alpha * (B * eps_t_substep - one_div_m) * rho_in_8;
    tmp += pi_alpha_delta_L * rho_in_4;
    tmp += pi_alpha_delta_R * rho_in_6;
    drho_out_9 = tmp;

    // (1,2)
    tmp = 0.0f;
    tmp += pi_alpha * eps_t_substep * rho_in_11;
    tmp += -(-pi_alpha * eps_t_substep) * rho_in_11;
    tmp += pi_alpha_delta_R * rho_in_7;
    tmp += pi_alpha_delta_L * rho_in_5;
    tmp += -pi_alpha_delta_L * rho_in_15;
    tmp += -pi_alpha_delta_R * rho_in_13;
    drho_out_10 = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * eps_t_substep * rho_in_10;
    tmp += -pi_alpha_delta_R * rho_in_6;
    tmp += pi_alpha_delta_C * rho_in_1;
    tmp += -pi_alpha_delta_C * rho_in_2;
    tmp += -pi_alpha_delta_L * rho_in_14;
    tmp += -pi_alpha * eps_t_substep * rho_in_10;
    tmp += pi_alpha_delta_L * rho_in_4;
    tmp += pi_alpha_delta_R * rho_in_12;
    drho_out_11 = tmp;

    // (1,3)
    tmp = 0.0f;
    tmp += pi_alpha * eps_t_substep * rho_in_13;
    tmp += -pi_alpha * (B * eps_t_substep - one_div_m) * rho_in_13;
    tmp += pi_alpha_delta_R * rho_in_9;
    //tmp += H03 * rho_in_5;
    tmp += pi_alpha_delta_C * rho_in_15;
    tmp += -pi_alpha_delta_R * rho_in_11;
    drho_out_12 = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * eps_t_substep * rho_in_12;
    tmp += -pi_alpha_delta_C * rho_in_14;
    tmp += -pi_alpha_delta_R * rho_in_8;
    //tmp += H03 * rho_in_4;
    tmp += pi_alpha_delta_L * rho_in_1;
    tmp += -pi_alpha_delta_L * rho_in_3;
    tmp += pi_alpha * (B * eps_t_substep - one_div_m) * rho_in_12;
    tmp += pi_alpha_delta_R * rho_in_10;
    drho_out_13 = tmp;

    // (2,3)
    tmp = 0.0f;
    tmp += -pi_alpha * eps_t_substep * rho_in_15;
    tmp += -pi_alpha * (B * eps_t_substep - one_div_m) * rho_in_15;
    tmp += pi_alpha_delta_L * rho_in_9;
    //tmp += H03 * rho_in_7;
    tmp += pi_alpha_delta_C * rho_in_13;
    tmp += pi_alpha_delta_L * rho_in_11;
    drho_out_14 = tmp;

    tmp = 0.0f;
    tmp += -(-pi_alpha * eps_t_substep) * rho_in_14;
    tmp += -pi_alpha_delta_C * rho_in_12;
    tmp += -pi_alpha_delta_L * rho_in_8;
    //tmp += H03 * rho_in_6;
    tmp += pi_alpha_delta_R * rho_in_2;
    tmp += -pi_alpha_delta_R * rho_in_3;
    tmp += pi_alpha * (B * eps_t_substep - one_div_m) * rho_in_14;
    tmp += pi_alpha_delta_L * rho_in_10;
    drho_out_15 = tmp;


}
*/

/*
// probably wrong sign
static __device__ __forceinline__
void commutator_v2_unrolled_log(
    const float rho_in_0, const float rho_in_1, const float rho_in_2,
    const float rho_in_3, const float rho_in_4, const float rho_in_5,
    const float rho_in_6, const float rho_in_7, const float rho_in_8,
    const float rho_in_9, const float rho_in_10, const float rho_in_11,
    const float rho_in_12, const float rho_in_13, const float rho_in_14,
    const float rho_in_15,

    float& drho_out_0, float& drho_out_1, float& drho_out_2,
    float& drho_out_3, float& drho_out_4, float& drho_out_5,
    float& drho_out_6, float& drho_out_7, float& drho_out_8,
    float& drho_out_9, float& drho_out_10, float& drho_out_11,
    float& drho_out_12, float& drho_out_13, float& drho_out_14,
    float& drho_out_15,

    const float eps_t_substep,

    // log
    LogEntry* __restrict__ d_log_buffer,
    const int t_idx_substep
)
{

    drho_out_0 = 0.f;
    drho_out_1 = 0.f;
    drho_out_2 = 0.f;
    drho_out_3 = 0.f;
    drho_out_4 = 0.f;
    drho_out_5 = 0.f;
    drho_out_6 = 0.f;
    drho_out_7 = 0.f;
    drho_out_8 = 0.f;
    drho_out_9 = 0.f;
    drho_out_10 = 0.f;
    drho_out_11 = 0.f;
    drho_out_12 = 0.f;
    drho_out_13 = 0.f;
    drho_out_14 = 0.f;
    drho_out_15 = 0.f;


    // --- load drho_out into registers (accumulators) ---
    register float tmp;
    //const float H03 = 0.0f;

    // --- Commutator contribution (register-accumulated, one-term-per-line) ---

    // Diagonals
    tmp = 0.0f;
    tmp += -2.f * pi_alpha_delta_R * rho_in_5;
    tmp += -2.f * pi_alpha_delta_L * rho_in_7;
    //tmp += -2.f * H03 * rho_in_9;
    drho_out_0 = tmp;


    tmp = 0.0f;
    tmp += -2.f * -pi_alpha_delta_R * rho_in_5;
    tmp += -2.f * pi_alpha_delta_C * rho_in_11;
    tmp += -2.f * pi_alpha_delta_L * rho_in_13;
    drho_out_1 = tmp;

    tmp = 0.0f;
    tmp += -2.f * -pi_alpha_delta_L * rho_in_7;
    tmp += -2.f * -pi_alpha_delta_C * rho_in_11;
    tmp += -2.f * pi_alpha_delta_R * rho_in_15;
    drho_out_2 = tmp;

    tmp = 0.0f;
    //tmp += -2.f * -H03 * rho_in_9;
    tmp += -2.f * -pi_alpha_delta_L * rho_in_13;
    tmp += -2.f * -pi_alpha_delta_R * rho_in_15;
    drho_out_3 = tmp;

    // (0,1)
    tmp = 0.0f;
    tmp += pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in_5;
    tmp += -pi_alpha * eps_t_substep * rho_in_5;
    tmp += -pi_alpha_delta_L * rho_in_11;
    tmp += -pi_alpha_delta_C * rho_in_7;
    //tmp += -H03 * rho_in_13;
    tmp += -pi_alpha_delta_L * rho_in_9;
    drho_out_4 = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in_4;
    tmp += pi_alpha_delta_R * rho_in_0;
    tmp += -pi_alpha_delta_R * rho_in_1;
    tmp += -pi_alpha_delta_L * rho_in_10;
    //tmp += -H03 * rho_in_12;
    tmp += pi_alpha * eps_t_substep * rho_in_4;
    tmp += pi_alpha_delta_C * rho_in_6;
    tmp += pi_alpha_delta_L * rho_in_8;
    drho_out_5 = tmp;

    // (0,2)
    tmp = 0.0f;
    tmp += pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in_7;
    tmp += -(-pi_alpha * eps_t_substep) * rho_in_7;
    tmp += pi_alpha_delta_R * rho_in_11;
    tmp += -pi_alpha_delta_C * rho_in_5;
    //tmp += -H03 * rho_in_15;
    tmp += -pi_alpha_delta_R * rho_in_9;
    drho_out_6 = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in_6;
    tmp += -pi_alpha_delta_R * rho_in_10;
    tmp += pi_alpha_delta_L * rho_in_0;
    tmp += -pi_alpha_delta_L * rho_in_2;
    //tmp += -H03 * rho_in_14;
    tmp += -pi_alpha * eps_t_substep * rho_in_6;
    tmp += pi_alpha_delta_C * rho_in_4;
    tmp += pi_alpha_delta_R * rho_in_8;
    drho_out_7 = tmp;

    // (0,3)
    tmp = 0.0f;
    tmp += pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in_9;
    tmp += -pi_alpha * (B * eps_t_substep - one_div_m) * rho_in_9;
    tmp += pi_alpha_delta_R * rho_in_13;
    tmp += pi_alpha_delta_L * rho_in_15;
    tmp += -pi_alpha_delta_L * rho_in_5;
    tmp += -pi_alpha_delta_R * rho_in_7;
    drho_out_8 = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in_8;
    tmp += -pi_alpha_delta_R * rho_in_12;
    tmp += -pi_alpha_delta_L * rho_in_14;
    //tmp += H03 * rho_in_0;
    //tmp += -H03 * rho_in_3;
    tmp += pi_alpha * (B * eps_t_substep - one_div_m) * rho_in_8;
    tmp += pi_alpha_delta_L * rho_in_4;
    tmp += pi_alpha_delta_R * rho_in_6;
    drho_out_9 = tmp;

    // (1,2)
    tmp = 0.0f;
    tmp += pi_alpha * eps_t_substep * rho_in_11;
    tmp += -(-pi_alpha * eps_t_substep) * rho_in_11;
    tmp += pi_alpha_delta_R * rho_in_7;
    tmp += pi_alpha_delta_L * rho_in_5;
    tmp += -pi_alpha_delta_L * rho_in_15;
    tmp += -pi_alpha_delta_R * rho_in_13;
    drho_out_10 = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * eps_t_substep * rho_in_10;
    tmp += -pi_alpha_delta_R * rho_in_6;
    tmp += pi_alpha_delta_C * rho_in_1;
    tmp += -pi_alpha_delta_C * rho_in_2;
    tmp += -pi_alpha_delta_L * rho_in_14;
    tmp += -pi_alpha * eps_t_substep * rho_in_10;
    tmp += pi_alpha_delta_L * rho_in_4;
    tmp += pi_alpha_delta_R * rho_in_12;
    drho_out_11 = tmp;

    // (1,3)
    tmp = 0.0f;
    tmp += pi_alpha * eps_t_substep * rho_in_13;
    tmp += -pi_alpha * (B * eps_t_substep - one_div_m) * rho_in_13;
    tmp += pi_alpha_delta_R * rho_in_9;
    //tmp += H03 * rho_in_5;
    tmp += pi_alpha_delta_C * rho_in_15;
    tmp += -pi_alpha_delta_R * rho_in_11;
    drho_out_12 = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * eps_t_substep * rho_in_12;
    tmp += -pi_alpha_delta_C * rho_in_14;
    tmp += -pi_alpha_delta_R * rho_in_8;
    //tmp += H03 * rho_in_4;
    tmp += pi_alpha_delta_L * rho_in_1;
    tmp += -pi_alpha_delta_L * rho_in_3;
    tmp += pi_alpha * (B * eps_t_substep - one_div_m) * rho_in_12;
    tmp += pi_alpha_delta_R * rho_in_10;
    drho_out_13 = tmp;

    // (2,3)
    tmp = 0.0f;
    tmp += -pi_alpha * eps_t_substep * rho_in_15;
    tmp += -pi_alpha * (B * eps_t_substep - one_div_m) * rho_in_15;
    tmp += pi_alpha_delta_L * rho_in_9;
    //tmp += H03 * rho_in_7;
    tmp += pi_alpha_delta_C * rho_in_13;
    tmp += pi_alpha_delta_L * rho_in_11;
    drho_out_14 = tmp;

    tmp = 0.0f;
    tmp += -(-pi_alpha * eps_t_substep) * rho_in_14;
    tmp += -pi_alpha_delta_C * rho_in_12;
    tmp += -pi_alpha_delta_L * rho_in_8;
    //tmp += H03 * rho_in_6;
    tmp += pi_alpha_delta_R * rho_in_2;
    tmp += -pi_alpha_delta_R * rho_in_3;
    tmp += pi_alpha * (B * eps_t_substep - one_div_m) * rho_in_14;
    tmp += pi_alpha_delta_L * rho_in_10;
    drho_out_15 = tmp;


    d_log_buffer[t_idx_substep].drho_out_comm_0 = drho_out_0;
    d_log_buffer[t_idx_substep].drho_out_comm_1 = drho_out_1;
    d_log_buffer[t_idx_substep].drho_out_comm_2 = drho_out_2;
    d_log_buffer[t_idx_substep].drho_out_comm_3 = drho_out_3;
    d_log_buffer[t_idx_substep].drho_out_comm_4 = drho_out_4;
    d_log_buffer[t_idx_substep].drho_out_comm_5 = drho_out_5;
    d_log_buffer[t_idx_substep].drho_out_comm_6 = drho_out_6;
    d_log_buffer[t_idx_substep].drho_out_comm_7 = drho_out_7;
    d_log_buffer[t_idx_substep].drho_out_comm_8 = drho_out_8;
    d_log_buffer[t_idx_substep].drho_out_comm_9 = drho_out_9;
    d_log_buffer[t_idx_substep].drho_out_comm_10 = drho_out_10;
    d_log_buffer[t_idx_substep].drho_out_comm_11 = drho_out_11;
    d_log_buffer[t_idx_substep].drho_out_comm_12 = drho_out_12;
    d_log_buffer[t_idx_substep].drho_out_comm_13 = drho_out_13;
    d_log_buffer[t_idx_substep].drho_out_comm_14 = drho_out_14;
    d_log_buffer[t_idx_substep].drho_out_comm_15 = drho_out_15;

}
*/



// optimized v2 for the case when delta_L and delta_r = 0
// fixed sign
static __device__ __forceinline__
void commutator_v3_unrolled(
    const float rho_in_0, const float rho_in_1, const float rho_in_2,
    const float rho_in_3, const float rho_in_4, const float rho_in_5,
    const float rho_in_6, const float rho_in_7, const float rho_in_8,
    const float rho_in_9, const float rho_in_10, const float rho_in_11,
    const float rho_in_12, const float rho_in_13, const float rho_in_14,
    const float rho_in_15,

    float& drho_out_0, float& drho_out_1, float& drho_out_2,
    float& drho_out_3, float& drho_out_4, float& drho_out_5,
    float& drho_out_6, float& drho_out_7, float& drho_out_8,
    float& drho_out_9, float& drho_out_10, float& drho_out_11,
    float& drho_out_12, float& drho_out_13, float& drho_out_14,
    float& drho_out_15,

    const float eps_t_substep
)
{

    //drho_out_0 = 0.f;
    //drho_out_1 = 0.f;
    //drho_out_2 = 0.f;
    //drho_out_3 = 0.f;
    //drho_out_4 = 0.f;
    //drho_out_5 = 0.f;
    //drho_out_6 = 0.f;
    //drho_out_7 = 0.f;
    //drho_out_8 = 0.f;
    //drho_out_9 = 0.f;
    //drho_out_10 = 0.f;
    //drho_out_11 = 0.f;
    //drho_out_12 = 0.f;
    //drho_out_13 = 0.f;
    //drho_out_14 = 0.f;
    //drho_out_15 = 0.f;



    // --- load drho_out into registers (accumulators) ---
    register float tmp;
    //const float H03 = 0.0f;

    // --- Commutator contribution (register-accumulated, one-term-per-line) ---

    drho_out_0 = 0.0f;

    tmp = delta_C * rho_in_11;

    drho_out_1 = tmp;

    drho_out_2 = -tmp;

    drho_out_3 = 0.0f;

    // (0,1)
    tmp = 0.0f;
    tmp = B * eps_t_substep;
    tmp += one_div_m;
    tmp += eps_t_substep;
    tmp *= rho_in_5;

    tmp += delta_C * rho_in_7;
    tmp *= 0.5f;
    drho_out_4 = tmp;

    tmp = 0.0f;
    tmp = -B * eps_t_substep;
    tmp += -one_div_m;
    tmp += -eps_t_substep;
    tmp *= rho_in_4;

    tmp += -delta_C * rho_in_6;
    tmp *= 0.5f;
    drho_out_5 = tmp;

    // (0,2)
    tmp = 0.0f;
    tmp = B * eps_t_substep;
    tmp += one_div_m;
    tmp += -eps_t_substep;
    tmp *= rho_in_7;

    tmp += delta_C * rho_in_5;
    tmp *= 0.5f;
    drho_out_6 = tmp;

    tmp = 0.0f;
    tmp = -B * eps_t_substep;
    tmp += -one_div_m;
    tmp += eps_t_substep;
    tmp *= rho_in_6;

    tmp += -delta_C * rho_in_4;
    tmp *= 0.5f;
    drho_out_7 = tmp;

    // (0,3)
    drho_out_8 = B * eps_t_substep * rho_in_9;

    drho_out_9 = -B * eps_t_substep * rho_in_8;

    // (1,2)
    drho_out_10 = -eps_t_substep * rho_in_11;

    tmp = 0.0f;
    tmp = -delta_C * rho_in_1;
    tmp += delta_C * rho_in_2;
    tmp += 2 * eps_t_substep * rho_in_10;
    tmp *= 0.5f;
    drho_out_11 = tmp;

    // (1,3)
    tmp = 0.0f;
    tmp = -eps_t_substep;
    tmp += B * eps_t_substep;
    tmp += -one_div_m;
    tmp *= rho_in_13;

    tmp += -delta_C * rho_in_15;
    tmp *= 0.5f;
    drho_out_12 = tmp;

    tmp = 0.0f;
    tmp = eps_t_substep;
    tmp += -B * eps_t_substep;
    tmp += one_div_m;
    tmp *= rho_in_12;

    tmp += delta_C * rho_in_14;
    tmp *= 0.5f;
    drho_out_13 = tmp;

    // (2,3)
    tmp = 0.0f;
    tmp = eps_t_substep;
    tmp += B * eps_t_substep;
    tmp += -one_div_m;
    tmp *= rho_in_15;

    tmp += -delta_C * rho_in_13;
    tmp *= 0.5f;
    drho_out_14 = tmp;


    tmp = 0.0f;
    tmp = -B * eps_t_substep;
    tmp += -eps_t_substep;
    tmp += one_div_m;
    tmp *= rho_in_14;

    tmp += delta_C * rho_in_12;
    tmp *= 0.5f;
    drho_out_15 = tmp;



}


// optimized v2 for the case when delta_L and delta_r = 0
// fixed sign
static __device__ __forceinline__
void commutator_v3_unrolled_log(
    const float rho_in_0, const float rho_in_1, const float rho_in_2,
    const float rho_in_3, const float rho_in_4, const float rho_in_5,
    const float rho_in_6, const float rho_in_7, const float rho_in_8,
    const float rho_in_9, const float rho_in_10, const float rho_in_11,
    const float rho_in_12, const float rho_in_13, const float rho_in_14,
    const float rho_in_15,

    float& drho_out_0, float& drho_out_1, float& drho_out_2,
    float& drho_out_3, float& drho_out_4, float& drho_out_5,
    float& drho_out_6, float& drho_out_7, float& drho_out_8,
    float& drho_out_9, float& drho_out_10, float& drho_out_11,
    float& drho_out_12, float& drho_out_13, float& drho_out_14,
    float& drho_out_15,

    const float eps_t_substep,

    // log
    LogEntry* __restrict__ d_log_buffer,
    const int t_idx_substep
)
{

    //drho_out_0 = 0.f;
    //drho_out_1 = 0.f;
    //drho_out_2 = 0.f;
    //drho_out_3 = 0.f;
    //drho_out_4 = 0.f;
    //drho_out_5 = 0.f;
    //drho_out_6 = 0.f;
    //drho_out_7 = 0.f;
    //drho_out_8 = 0.f;
    //drho_out_9 = 0.f;
    //drho_out_10 = 0.f;
    //drho_out_11 = 0.f;
    //drho_out_12 = 0.f;
    //drho_out_13 = 0.f;
    //drho_out_14 = 0.f;
    //drho_out_15 = 0.f;



    // --- load drho_out into registers (accumulators) ---
    register float tmp;
    //const float H03 = 0.0f;

    // --- Commutator contribution (register-accumulated, one-term-per-line) ---

    drho_out_0 = 0.0f;

    tmp = delta_C * rho_in_11;

    drho_out_1 = tmp;

    drho_out_2 = -tmp;

    drho_out_3 = 0.0f;

    // (0,1)
    tmp = 0.0f;
    tmp = B * eps_t_substep;
    tmp += one_div_m;
    tmp += eps_t_substep;
    tmp *= rho_in_5;

    tmp += delta_C * rho_in_7;
    tmp *= 0.5f;
    drho_out_4 = tmp;

    tmp = 0.0f;
    tmp = -B * eps_t_substep;
    tmp += -one_div_m;
    tmp += -eps_t_substep;
    tmp *= rho_in_4;

    tmp += -delta_C * rho_in_6;
    tmp *= 0.5f;
    drho_out_5 = tmp;

    // (0,2)
    tmp = 0.0f;
    tmp = B * eps_t_substep;
    tmp += one_div_m;
    tmp += -eps_t_substep;
    tmp *= rho_in_7;

    tmp += delta_C * rho_in_5;
    tmp *= 0.5f;
    drho_out_6 = tmp;

    tmp = 0.0f;
    tmp = -B * eps_t_substep;
    tmp += -one_div_m;
    tmp += eps_t_substep;
    tmp *= rho_in_6;

    tmp += -delta_C * rho_in_4;
    tmp *= 0.5f;
    drho_out_7 = tmp;

    // (0,3)
    drho_out_8 = B * eps_t_substep * rho_in_9;

    drho_out_9 = -B * eps_t_substep * rho_in_8;

    // (1,2)
    drho_out_10 = -eps_t_substep * rho_in_11;

    tmp = 0.0f;
    tmp = -delta_C * rho_in_1;
    tmp += delta_C * rho_in_2;
    tmp += 2 * eps_t_substep * rho_in_10;
    tmp *= 0.5f;
    drho_out_11 = tmp;

    // (1,3)
    tmp = 0.0f;
    tmp = -eps_t_substep;
    tmp += B * eps_t_substep;
    tmp += -one_div_m;
    tmp *= rho_in_13;

    tmp += -delta_C * rho_in_15;
    tmp *= 0.5f;
    drho_out_12 = tmp;

    tmp = 0.0f;
    tmp = eps_t_substep;
    tmp += -B * eps_t_substep;
    tmp += one_div_m;
    tmp *= rho_in_12;

    tmp += delta_C * rho_in_14;
    tmp *= 0.5f;
    drho_out_13 = tmp;

    // (2,3)
    tmp = 0.0f;
    tmp = eps_t_substep;
    tmp += B * eps_t_substep;
    tmp += -one_div_m;
    tmp *= rho_in_15;

    tmp += -delta_C * rho_in_13;
    tmp *= 0.5f;
    drho_out_14 = tmp;


    tmp = 0.0f;
    tmp = -B * eps_t_substep;
    tmp += -eps_t_substep;
    tmp += one_div_m;
    tmp *= rho_in_14;

    tmp += delta_C * rho_in_12;
    tmp *= 0.5f;
    drho_out_15 = tmp;






    d_log_buffer[t_idx_substep].drho_out_comm_0 = drho_out_0;
    d_log_buffer[t_idx_substep].drho_out_comm_1 = drho_out_1;
    d_log_buffer[t_idx_substep].drho_out_comm_2 = drho_out_2;
    d_log_buffer[t_idx_substep].drho_out_comm_3 = drho_out_3;
    d_log_buffer[t_idx_substep].drho_out_comm_4 = drho_out_4;
    d_log_buffer[t_idx_substep].drho_out_comm_5 = drho_out_5;
    d_log_buffer[t_idx_substep].drho_out_comm_6 = drho_out_6;
    d_log_buffer[t_idx_substep].drho_out_comm_7 = drho_out_7;
    d_log_buffer[t_idx_substep].drho_out_comm_8 = drho_out_8;
    d_log_buffer[t_idx_substep].drho_out_comm_9 = drho_out_9;
    d_log_buffer[t_idx_substep].drho_out_comm_10 = drho_out_10;
    d_log_buffer[t_idx_substep].drho_out_comm_11 = drho_out_11;
    d_log_buffer[t_idx_substep].drho_out_comm_12 = drho_out_12;
    d_log_buffer[t_idx_substep].drho_out_comm_13 = drho_out_13;
    d_log_buffer[t_idx_substep].drho_out_comm_14 = drho_out_14;
    d_log_buffer[t_idx_substep].drho_out_comm_15 = drho_out_15;

}


/*
__device__ __forceinline__
void commutator_v4(
    const float rho_in[16],
    float drho_out[16],

    const float eps_t_substep
)
{

    for (int i = 0; i < 16; i++) drho_out[i] = 0.f;

    // --- Hamiltonian ---
    const float H00 = pi_alpha * (-B * eps_t_substep - one_div_m);
    const float H01 = pi_alpha_delta_R;
    const float H02 = pi_alpha_delta_L;
    const float H03 = 0.f;
    const float H11 = pi_alpha * eps_t_substep;
    const float H12 = pi_alpha_delta_C;
    const float H13 = pi_alpha_delta_L;
    const float H22 = -pi_alpha * eps_t_substep;
    const float H23 = pi_alpha_delta_R;
    const float H33 = pi_alpha * (B * eps_t_substep - one_div_m);



    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)
    //float eps_t_substep = eps0 + A * cosf(omega * t_substep);
    
    //const float local_H00 = pi_alpha * (-B * eps_t_substep - one_div_m);
    //const float local_H33 = pi_alpha * (B * eps_t_substep - one_div_m);

    //const float local_pi_alpha_eps_t_substep = pi_alpha * eps_t_substep;

    //const float local_pi_alpha_delta_R = pi_alpha_delta_R;
    //const float local_pi_alpha_delta_L = pi_alpha_delta_L;
    //const float local_pi_alpha_delta_C = pi_alpha_delta_C;



    // --- load drho_out into registers (accumulators) ---
    register float tmp;

    //const float H03 = 0.0f;

    // --- Commutator contribution (register-accumulated, one-term-per-line) ---


// Diagonals
    tmp = 0.0f;
    tmp += -2.f * H01 * rho_in[5];
    tmp += -2.f * H02 * rho_in[7];
    tmp += -2.f * H03 * rho_in[9];
    drho_out[0] += tmp;

    tmp = 0.0f;
    tmp += -2.f * -H01 * rho_in[5];
    tmp += -2.f * H12 * rho_in[11];
    tmp += -2.f * H13 * rho_in[13];
    drho_out[1] += tmp;

    tmp = 0.0f;
    tmp += -2.f * -H02 * rho_in[7];
    tmp += -2.f * -H12 * rho_in[11];
    tmp += -2.f * H23 * rho_in[15];
    drho_out[2] += tmp;

    tmp = 0.0f;
    tmp += -2.f * -H03 * rho_in[9];
    tmp += -2.f * -H13 * rho_in[13];
    tmp += -2.f * -H23 * rho_in[15];
    drho_out[3] += tmp;


    // (0,1)
    tmp = 0.0f;
    tmp += (H00 - H11) * rho_in[5];
    tmp += -H02 * rho_in[11];
    tmp += -H12 * rho_in[7];
    tmp += -H03 * rho_in[13];
    tmp += -H13 * rho_in[9];
    drho_out[4] += tmp;

    tmp = 0.0f;
    tmp += -H00 * rho_in[4];
    tmp += H01 * rho_in[0];
    tmp += -H01 * rho_in[1];
    tmp += -H02 * rho_in[10];
    tmp += -H03 * rho_in[12];
    tmp += H11 * rho_in[4];
    tmp += H12 * rho_in[6];
    tmp += H13 * rho_in[8];
    drho_out[5] += tmp;


    // (0,2)
    tmp = 0.0f;
    tmp += (H00 - H22) * rho_in[7];
    tmp += H01 * rho_in[11];
    tmp += -H12 * rho_in[5];
    tmp += -H03 * rho_in[15];
    tmp += -H23 * rho_in[9];
    drho_out[6] += tmp;

    tmp = 0.0f;
    tmp += -H00 * rho_in[6];
    tmp += -H01 * rho_in[10];
    tmp += H02 * rho_in[0];
    tmp += -H02 * rho_in[2];
    tmp += -H03 * rho_in[14];
    tmp += H22 * rho_in[6];
    tmp += H12 * rho_in[4];
    tmp += H23 * rho_in[8];
    drho_out[7] += tmp;


    // (0,3)
    tmp = 0.0f;
    tmp += (H00 - H33) * rho_in[9];
    tmp += H01 * rho_in[13];
    tmp += H02 * rho_in[15];
    tmp += -H13 * rho_in[5];
    tmp += -H23 * rho_in[7];
    drho_out[8] += tmp;

    tmp = 0.0f;
    tmp += -H00 * rho_in[8];
    tmp += -H01 * rho_in[12];
    tmp += -H02 * rho_in[14];
    tmp += H03 * rho_in[0];
    tmp += -H03 * rho_in[3];
    tmp += H33 * rho_in[8];
    tmp += H13 * rho_in[4];
    tmp += H23 * rho_in[6];
    drho_out[9] += tmp;


    // (1,2)
    tmp = 0.0f;
    tmp += (H11 - H22) * rho_in[11];
    tmp += H01 * rho_in[7];
    tmp += H02 * rho_in[5];
    tmp += -H13 * rho_in[15];
    tmp += -H23 * rho_in[13];
    drho_out[10] += tmp;

    tmp = 0.0f;
    tmp += -H11 * rho_in[10];
    tmp += -H01 * rho_in[6];
    tmp += H12 * rho_in[1];
    tmp += -H12 * rho_in[2];
    tmp += -H13 * rho_in[14];
    tmp += H22 * rho_in[10];
    tmp += H02 * rho_in[4];
    tmp += H23 * rho_in[12];
    drho_out[11] += tmp;


    // (1,3)
    tmp = 0.0f;
    tmp += (H11 - H33) * rho_in[13];
    tmp += H01 * rho_in[9];
    tmp += H03 * rho_in[5];
    tmp += H12 * rho_in[15];
    tmp += -H23 * rho_in[11];
    drho_out[12] += tmp;

    tmp = 0.0f;
    tmp += -H11 * rho_in[12];
    tmp += -H12 * rho_in[14];
    tmp += -H01 * rho_in[8];
    tmp += H03 * rho_in[4];
    tmp += H13 * rho_in[1];
    tmp += -H13 * rho_in[3];
    tmp += H33 * rho_in[12];
    tmp += H23 * rho_in[10];
    drho_out[13] += tmp;


    // (2,3)
    tmp = 0.0f;
    tmp += (H22 - H33) * rho_in[15];
    tmp += H02 * rho_in[9];
    tmp += H03 * rho_in[7];
    tmp += H12 * rho_in[13];
    tmp += H13 * rho_in[11];
    drho_out[14] += tmp;

    tmp = 0.0f;
    tmp += -H22 * rho_in[14];
    tmp += -H12 * rho_in[12];
    tmp += -H02 * rho_in[8];
    tmp += H03 * rho_in[6];
    tmp += H23 * rho_in[2];
    tmp += -H23 * rho_in[3];
    tmp += H33 * rho_in[14];
    tmp += H13 * rho_in[10];
    drho_out[15] += tmp;

}
*/



/*
__device__ __forceinline__
void commutator_v3(
    const float rho_in[16],
    float drho_out[16],

    const float eps_t_substep
)
{

    for (int i = 0; i < 16; i++) drho_out[i] = 0.f;

    // compute time-dependent diagonals in registers (if needed by commutator/dissipator)
    //float eps_t_substep = eps0 + A * cosf(omega * t_substep);
    const float local_H00 = pi_alpha * (-B * eps_t_substep - one_div_m);
    const float local_H33 = pi_alpha * (B * eps_t_substep - one_div_m);

    const float local_pi_alpha_eps_t_substep = pi_alpha * eps_t_substep;

    const float local_pi_alpha_delta_R = pi_alpha_delta_R;
    const float local_pi_alpha_delta_L = pi_alpha_delta_L;
    const float local_pi_alpha_delta_C = pi_alpha_delta_C;



    // --- load drho_out into registers (accumulators) ---
    register float tmp;

    //const float H03 = 0.0f;

    // --- Commutator contribution (register-accumulated, one-term-per-line) ---

    // Diagonals
    tmp = 0.0f;
    tmp += -2.f * local_pi_alpha_delta_R * rho_in[5];
    tmp += -2.f * local_pi_alpha_delta_L * rho_in[7];
    //tmp += -2.f * H03 * rho_in[9];
    drho_out[0] = tmp;


    tmp = 0.0f;
    tmp += -2.f * -local_pi_alpha_delta_R * rho_in[5];
    tmp += -2.f * local_pi_alpha_delta_C * rho_in[11];
    tmp += -2.f * local_pi_alpha_delta_L * rho_in[13];
    drho_out[1] = tmp;

    tmp = 0.0f;
    tmp += -2.f * -local_pi_alpha_delta_L * rho_in[7];
    tmp += -2.f * -local_pi_alpha_delta_C * rho_in[11];
    tmp += -2.f * local_pi_alpha_delta_R * rho_in[15];
    drho_out[2] = tmp;

    tmp = 0.0f;
    //tmp += -2.f * -H03 * rho_in[9];
    tmp += -2.f * -local_pi_alpha_delta_L * rho_in[13];
    tmp += -2.f * -local_pi_alpha_delta_R * rho_in[15];
    drho_out[3] = tmp;

    // (0,1)
    tmp = 0.0f;
    //tmp += pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in[5];
    tmp += local_H00 * rho_in[5];
    tmp += -local_pi_alpha_eps_t_substep * rho_in[5];
    tmp += -local_pi_alpha_delta_L * rho_in[11];
    tmp += -local_pi_alpha_delta_C * rho_in[7];
    //tmp += -H03 * rho_in[13];
    tmp += -local_pi_alpha_delta_L * rho_in[9];
    drho_out[4] = tmp;

    tmp = 0.0f;
    //tmp += -pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in[4];
    tmp += -local_H00 * rho_in[4];
    tmp += local_pi_alpha_delta_R * rho_in[0];
    tmp += -local_pi_alpha_delta_R * rho_in[1];
    tmp += -local_pi_alpha_delta_L * rho_in[10];
    //tmp += -H03 * rho_in[12];
    tmp += local_pi_alpha_eps_t_substep * rho_in[4];
    tmp += local_pi_alpha_delta_C * rho_in[6];
    tmp += local_pi_alpha_delta_L * rho_in[8];
    drho_out[5] = tmp;

    // (0,2)
    tmp = 0.0f;
    //tmp += pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in[7];
    tmp += local_H00 * rho_in[7];
    tmp += local_pi_alpha_eps_t_substep * rho_in[7];
    tmp += local_pi_alpha_delta_R * rho_in[11];
    tmp += -local_pi_alpha_delta_C * rho_in[5];
    //tmp += -H03 * rho_in[15];
    tmp += -local_pi_alpha_delta_R * rho_in[9];
    drho_out[6] = tmp;

    tmp = 0.0f;
    //tmp += -pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in[6];
    tmp += -local_H00 * rho_in[6];
    tmp += -local_pi_alpha_delta_R * rho_in[10];
    tmp += local_pi_alpha_delta_L * rho_in[0];
    tmp += -local_pi_alpha_delta_L * rho_in[2];
    //tmp += -H03 * rho_in[14];
    tmp += -local_pi_alpha_eps_t_substep * rho_in[6];
    tmp += local_pi_alpha_delta_C * rho_in[4];
    tmp += local_pi_alpha_delta_R * rho_in[8];
    drho_out[7] = tmp;

    // (0,3)
    tmp = 0.0f;
    //tmp += pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in[9];
    tmp += local_H00 * rho_in[9];
    //tmp += -pi_alpha * (B * eps_t_substep - one_div_m) * rho_in[9];
    tmp += -local_H33 * rho_in[9];
    tmp += local_pi_alpha_delta_R * rho_in[13];
    tmp += local_pi_alpha_delta_L * rho_in[15];
    tmp += -local_pi_alpha_delta_L * rho_in[5];
    tmp += -local_pi_alpha_delta_R * rho_in[7];
    drho_out[8] = tmp;

    tmp = 0.0f;
    //tmp += -pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in[8];
    tmp += -local_H00 * rho_in[8];
    tmp += -local_pi_alpha_delta_R * rho_in[12];
    tmp += -local_pi_alpha_delta_L * rho_in[14];
    //tmp += H03 * rho_in[0];
    //tmp += -H03 * rho_in[3];
    //tmp += pi_alpha * (B * eps_t_substep - one_div_m) * rho_in[8];
    tmp += local_H33 * rho_in[8];
    tmp += local_pi_alpha_delta_L * rho_in[4];
    tmp += local_pi_alpha_delta_R * rho_in[6];
    drho_out[9] = tmp;

    // (1,2)
    tmp = 0.0f;
    tmp += local_pi_alpha_eps_t_substep * rho_in[11];
    tmp += local_pi_alpha_eps_t_substep * rho_in[11];
    tmp += local_pi_alpha_delta_R * rho_in[7];
    tmp += local_pi_alpha_delta_L * rho_in[5];
    tmp += -local_pi_alpha_delta_L * rho_in[15];
    tmp += -local_pi_alpha_delta_R * rho_in[13];
    drho_out[10] = tmp;

    tmp = 0.0f;
    tmp += -local_pi_alpha_eps_t_substep * rho_in[10];
    tmp += -local_pi_alpha_delta_R * rho_in[6];
    tmp += local_pi_alpha_delta_C * rho_in[1];
    tmp += -local_pi_alpha_delta_C * rho_in[2];
    tmp += -local_pi_alpha_delta_L * rho_in[14];
    tmp += -local_pi_alpha_eps_t_substep * rho_in[10];
    tmp += local_pi_alpha_delta_L * rho_in[4];
    tmp += local_pi_alpha_delta_R * rho_in[12];
    drho_out[11] = tmp;

    // (1,3)
    tmp = 0.0f;
    tmp += local_pi_alpha_eps_t_substep * rho_in[13];
    //tmp += -pi_alpha * (B * eps_t_substep - one_div_m) * rho_in[13];
    tmp += -local_H33 * rho_in[13];
    tmp += local_pi_alpha_delta_R * rho_in[9];
    //tmp += H03 * rho_in[5];
    tmp += local_pi_alpha_delta_C * rho_in[15];
    tmp += -local_pi_alpha_delta_R * rho_in[11];
    drho_out[12] = tmp;

    tmp = 0.0f;
    tmp += -local_pi_alpha_eps_t_substep * rho_in[12];
    tmp += -local_pi_alpha_delta_C * rho_in[14];
    tmp += -local_pi_alpha_delta_R * rho_in[8];
    //tmp += H03 * rho_in[4];
    tmp += local_pi_alpha_delta_L * rho_in[1];
    tmp += -local_pi_alpha_delta_L * rho_in[3];
    //tmp += pi_alpha * (B * eps_t_substep - one_div_m) * rho_in[12];
    tmp += local_H33 * rho_in[12];
    tmp += local_pi_alpha_delta_R * rho_in[10];
    drho_out[13] = tmp;

    // (2,3)
    tmp = 0.0f;
    tmp += -local_pi_alpha_eps_t_substep * rho_in[15];
    //tmp += -pi_alpha * (B * eps_t_substep - one_div_m) * rho_in[15];
    tmp += -local_H33 * rho_in[15];
    tmp += local_pi_alpha_delta_L * rho_in[9];
    //tmp += H03 * rho_in[7];
    tmp += local_pi_alpha_delta_C * rho_in[13];
    tmp += local_pi_alpha_delta_L * rho_in[11];
    drho_out[14] = tmp;

    tmp = 0.0f;
    tmp += local_pi_alpha_eps_t_substep * rho_in[14];
    tmp += -local_pi_alpha_delta_C * rho_in[12];
    tmp += -local_pi_alpha_delta_L * rho_in[8];
    //tmp += H03 * rho_in[6];
    tmp += local_pi_alpha_delta_R * rho_in[2];
    tmp += -local_pi_alpha_delta_R * rho_in[3];
    //tmp += pi_alpha * (B * eps_t_substep - one_div_m) * rho_in[14];
    tmp += local_H33 * rho_in[14];
    tmp += local_pi_alpha_delta_L * rho_in[10];
    drho_out[15] = tmp;


}
*/



/*
static __device__ __forceinline__
void commutator_v2(
    const float rho_in[16],
    float drho_out[16],

    const float eps_t_substep
)
{

    for (int i = 0; i < 16; i++) drho_out[i] = 0.f;

    // --- load drho_out into registers (accumulators) ---
    register float tmp;
    //const float H03 = 0.0f;

    // --- Commutator contribution (register-accumulated, one-term-per-line) ---

    // Diagonals
    tmp = 0.0f;
    tmp += -2.f * pi_alpha_delta_R * rho_in[5];
    tmp += -2.f * pi_alpha_delta_L * rho_in[7];
    //tmp += -2.f * H03 * rho_in[9];
    drho_out[0] = tmp;


    tmp = 0.0f;
    tmp += -2.f * -pi_alpha_delta_R * rho_in[5];
    tmp += -2.f * pi_alpha_delta_C * rho_in[11];
    tmp += -2.f * pi_alpha_delta_L * rho_in[13];
    drho_out[1] = tmp;

    tmp = 0.0f;
    tmp += -2.f * -pi_alpha_delta_L * rho_in[7];
    tmp += -2.f * -pi_alpha_delta_C * rho_in[11];
    tmp += -2.f * pi_alpha_delta_R * rho_in[15];
    drho_out[2] = tmp;

    tmp = 0.0f;
    //tmp += -2.f * -H03 * rho_in[9];
    tmp += -2.f * -pi_alpha_delta_L * rho_in[13];
    tmp += -2.f * -pi_alpha_delta_R * rho_in[15];
    drho_out[3] = tmp;

    // (0,1)
    tmp = 0.0f;
    tmp += pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in[5];
    tmp += -pi_alpha * eps_t_substep * rho_in[5];
    tmp += -pi_alpha_delta_L * rho_in[11];
    tmp += -pi_alpha_delta_C * rho_in[7];
    //tmp += -H03 * rho_in[13];
    tmp += -pi_alpha_delta_L * rho_in[9];
    drho_out[4] = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in[4];
    tmp += pi_alpha_delta_R * rho_in[0];
    tmp += -pi_alpha_delta_R * rho_in[1];
    tmp += -pi_alpha_delta_L * rho_in[10];
    //tmp += -H03 * rho_in[12];
    tmp += pi_alpha * eps_t_substep * rho_in[4];
    tmp += pi_alpha_delta_C * rho_in[6];
    tmp += pi_alpha_delta_L * rho_in[8];
    drho_out[5] = tmp;

    // (0,2)
    tmp = 0.0f;
    tmp += pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in[7];
    tmp += -(-pi_alpha * eps_t_substep) * rho_in[7];
    tmp += pi_alpha_delta_R * rho_in[11];
    tmp += -pi_alpha_delta_C * rho_in[5];
    //tmp += -H03 * rho_in[15];
    tmp += -pi_alpha_delta_R * rho_in[9];
    drho_out[6] = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in[6];
    tmp += -pi_alpha_delta_R * rho_in[10];
    tmp += pi_alpha_delta_L * rho_in[0];
    tmp += -pi_alpha_delta_L * rho_in[2];
    //tmp += -H03 * rho_in[14];
    tmp += -pi_alpha * eps_t_substep * rho_in[6];
    tmp += pi_alpha_delta_C * rho_in[4];
    tmp += pi_alpha_delta_R * rho_in[8];
    drho_out[7] = tmp;

    // (0,3)
    tmp = 0.0f;
    tmp += pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in[9];
    tmp += -pi_alpha * (B * eps_t_substep - one_div_m) * rho_in[9];
    tmp += pi_alpha_delta_R * rho_in[13];
    tmp += pi_alpha_delta_L * rho_in[15];
    tmp += -pi_alpha_delta_L * rho_in[5];
    tmp += -pi_alpha_delta_R * rho_in[7];
    drho_out[8] = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * (-B * eps_t_substep - one_div_m) * rho_in[8];
    tmp += -pi_alpha_delta_R * rho_in[12];
    tmp += -pi_alpha_delta_L * rho_in[14];
    //tmp += H03 * rho_in[0];
    //tmp += -H03 * rho_in[3];
    tmp += pi_alpha * (B * eps_t_substep - one_div_m) * rho_in[8];
    tmp += pi_alpha_delta_L * rho_in[4];
    tmp += pi_alpha_delta_R * rho_in[6];
    drho_out[9] = tmp;

    // (1,2)
    tmp = 0.0f;
    tmp += pi_alpha * eps_t_substep * rho_in[11];
    tmp += -(-pi_alpha * eps_t_substep) * rho_in[11];
    tmp += pi_alpha_delta_R * rho_in[7];
    tmp += pi_alpha_delta_L * rho_in[5];
    tmp += -pi_alpha_delta_L * rho_in[15];
    tmp += -pi_alpha_delta_R * rho_in[13];
    drho_out[10] = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * eps_t_substep * rho_in[10];
    tmp += -pi_alpha_delta_R * rho_in[6];
    tmp += pi_alpha_delta_C * rho_in[1];
    tmp += -pi_alpha_delta_C * rho_in[2];
    tmp += -pi_alpha_delta_L * rho_in[14];
    tmp += -pi_alpha * eps_t_substep * rho_in[10];
    tmp += pi_alpha_delta_L * rho_in[4];
    tmp += pi_alpha_delta_R * rho_in[12];
    drho_out[11] = tmp;

    // (1,3)
    tmp = 0.0f;
    tmp += pi_alpha * eps_t_substep * rho_in[13];
    tmp += -pi_alpha * (B * eps_t_substep - one_div_m) * rho_in[13];
    tmp += pi_alpha_delta_R * rho_in[9];
    //tmp += H03 * rho_in[5];
    tmp += pi_alpha_delta_C * rho_in[15];
    tmp += -pi_alpha_delta_R * rho_in[11];
    drho_out[12] = tmp;

    tmp = 0.0f;
    tmp += -pi_alpha * eps_t_substep * rho_in[12];
    tmp += -pi_alpha_delta_C * rho_in[14];
    tmp += -pi_alpha_delta_R * rho_in[8];
    //tmp += H03 * rho_in[4];
    tmp += pi_alpha_delta_L * rho_in[1];
    tmp += -pi_alpha_delta_L * rho_in[3];
    tmp += pi_alpha * (B * eps_t_substep - one_div_m) * rho_in[12];
    tmp += pi_alpha_delta_R * rho_in[10];
    drho_out[13] = tmp;

    // (2,3)
    tmp = 0.0f;
    tmp += -pi_alpha * eps_t_substep * rho_in[15];
    tmp += -pi_alpha * (B * eps_t_substep - one_div_m) * rho_in[15];
    tmp += pi_alpha_delta_L * rho_in[9];
    //tmp += H03 * rho_in[7];
    tmp += pi_alpha_delta_C * rho_in[13];
    tmp += pi_alpha_delta_L * rho_in[11];
    drho_out[14] = tmp;

    tmp = 0.0f;
    tmp += -(-pi_alpha * eps_t_substep) * rho_in[14];
    tmp += -pi_alpha_delta_C * rho_in[12];
    tmp += -pi_alpha_delta_L * rho_in[8];
    //tmp += H03 * rho_in[6];
    tmp += pi_alpha_delta_R * rho_in[2];
    tmp += -pi_alpha_delta_R * rho_in[3];
    tmp += pi_alpha * (B * eps_t_substep - one_div_m) * rho_in[14];
    tmp += pi_alpha_delta_L * rho_in[10];
    drho_out[15] = tmp;


}
*/


/*
__device__ __forceinline__
void commutator_v1(
    const float* __restrict__ rho_in,
          float* __restrict__ drho_out,
    const float H00,
    const float H01,
    const float H02,
    const float H03,
    const float H11,
    const float H12,
    const float H13,
    const float H22,
    const float H23,
    const float H33
) {

    for (int i = 0; i < 16; i++) drho_out[i] = 0.f;

    // --- Commutator contribution ---
    // Diagonals: dr_aa = -2 * sum_b H_ab * i_ab  (b != a)
    drho_out[0] += -2.f * (H01 * rho_in[5] + H02 * rho_in[7] + H03 * rho_in[9]);
    drho_out[1] += -2.f * (-H01 * rho_in[5] + H12 * rho_in[11] + H13 * rho_in[13]);
    drho_out[2] += -2.f * (-H02 * rho_in[7] - H12 * rho_in[11] + H23 * rho_in[15]);
    drho_out[3] += -2.f * (-H03 * rho_in[9] - H13 * rho_in[13] - H23 * rho_in[15]);

    // Off-diagonals (a<b). For each pair (a,b):
    // d(Re rho_ab) = sum_k [ H_a,k * Im rho_kb - H_k,b * Im rho_ak ]
    // d(Im rho_ab) = sum_k [ H_k,rho * Re rho_ak - H_a,k * Re rho_kb ]

    // (0,1)
    drho_out[4] += (H00 - H11) * rho_in[5] - H02 * rho_in[11] - H12 * rho_in[7] - H03 * rho_in[13] - H13 * rho_in[9];           // d r01
    drho_out[5] += -H00 * rho_in[4] + H01 * rho_in[0] - H01 * rho_in[1] - H02 * rho_in[10] - H03 * rho_in[12] + H11 * rho_in[4] + H12 * rho_in[6] + H13 * rho_in[8]; // d i01

    // (0,2)
    drho_out[6] += (H00 - H22) * rho_in[7] + H01 * rho_in[11] - H12 * rho_in[5] - H03 * rho_in[15] - H23 * rho_in[9];           // d r02
    drho_out[7] += -H00 * rho_in[6] - H01 * rho_in[10] + H02 * rho_in[0] - H02 * rho_in[2] - H03 * rho_in[14] + H22 * rho_in[6] + H12 * rho_in[4] + H23 * rho_in[8]; // d i02

    // (0,3)
    drho_out[8] += (H00 - H33) * rho_in[9] + H01 * rho_in[13] + H02 * rho_in[15] - H13 * rho_in[5] - H23 * rho_in[7];           // d r03
    drho_out[9] += -H00 * rho_in[8] - H01 * rho_in[12] - H02 * rho_in[14] + H03 * rho_in[0] - H03 * rho_in[3] + H33 * rho_in[8] + H13 * rho_in[4] + H23 * rho_in[6]; // d i03

    // (1,2)
    drho_out[10] += (H11 - H22) * rho_in[11] + H01 * rho_in[7] + H02 * rho_in[5] - H13 * rho_in[15] - H23 * rho_in[13];           // d r12
    drho_out[11] += -H11 * rho_in[10] - H01 * rho_in[6] + H12 * rho_in[1] - H12 * rho_in[2] - H13 * rho_in[14] + H22 * rho_in[10] + H02 * rho_in[4] + H23 * rho_in[12]; // d i12

    // (1,3)
    drho_out[12] += (H11 - H33) * rho_in[13] + H01 * rho_in[9] + H03 * rho_in[5] + H12 * rho_in[15] - H23 * rho_in[11];           // d r13
    drho_out[13] += -H11 * rho_in[12] - H12 * rho_in[14] - H01 * rho_in[8] + H03 * rho_in[4] + H13 * rho_in[1] - H13 * rho_in[3] + H33 * rho_in[12] + H23 * rho_in[10]; // d i13

    // (2,3)
    drho_out[14] += (H22 - H33) * rho_in[15] + H02 * rho_in[9] + H03 * rho_in[7] + H12 * rho_in[13] + H13 * rho_in[11];           // d r23
    drho_out[15] += -H22 * rho_in[14] - H12 * rho_in[12] - H02 * rho_in[8] + H03 * rho_in[6] + H23 * rho_in[2] - H23 * rho_in[3] + H33 * rho_in[14] + H13 * rho_in[10]; // d i23

}
*/



/*
__device__ __forceinline__
void commutator_v0(
    const float rho_in[16],
    float drho_out[16],

    const float eps_t_substep
) {

    for (int i = 0; i < 16; i++) drho_out[i] = 0.f;


    // --- extract density matrix ---
    float r0 = rho_in[0], r1 = rho_in[1], r2 = rho_in[2], r3 = rho_in[3];
    float r01 = rho_in[4], i01 = rho_in[5];
    float r02 = rho_in[6], i02 = rho_in[7];
    float r03 = rho_in[8], i03 = rho_in[9];
    float r12 = rho_in[10], i12 = rho_in[11];
    float r13 = rho_in[12], i13 = rho_in[13];
    float r23 = rho_in[14], i23 = rho_in[15];

    // imaginary parts of diagonal are zero



    const float local_delta_C = 0.0003482627266695272f;
    const float local_delta_L = 0.0002f;
    const float local_delta_R = 0.0001f;

    const float local_B = B;
    const float local_one_div_m = one_div_m;



    //printf("local_B = %f, local_one_div_m = %f\n", local_B, local_one_div_m);
    


    // --- Hamiltonian ---
    const float H00 = pi_alpha * (-local_B * eps_t_substep - local_one_div_m);
    const float H01 = pi_alpha * local_delta_R;
    const float H02 = pi_alpha * local_delta_L;
    const float H03 = 0.f;
    const float H11 = pi_alpha * eps_t_substep;
    const float H12 = pi_alpha * local_delta_C;
    const float H13 = pi_alpha * local_delta_L;
    const float H22 = -pi_alpha * eps_t_substep;
    const float H23 = pi_alpha * local_delta_R;
    const float H33 = pi_alpha * (local_B * eps_t_substep - local_one_div_m);


    // float H10 = H01, H20 = H02, H30 = H03; // symmetrical
    // float H21 = H12, H31 = H13, H32 = H23; 


    // --- Commutator contribution ---
    // Diagonals: dr_aa = -2 * sum_b H_ab * i_ab  (b != a)
    drho_out[0] += -2.f * (H01 * i01 + H02 * i02 + H03 * i03);
    drho_out[1] += -2.f * (-H01 * i01 + H12 * i12 + H13 * i13);
    drho_out[2] += -2.f * (-H02 * i02 - H12 * i12 + H23 * i23);
    drho_out[3] += -2.f * (-H03 * i03 - H13 * i13 - H23 * i23);


    // (0,1)
    drho_out[4] += (H00 - H11) * i01 - H02 * i12 - H12 * i02 - H03 * i13 - H13 * i03;           // d r01
    drho_out[5] += -H00 * r01 + H01 * r0 - H01 * r1 - H02 * r12 - H03 * r13 + H11 * r01 + H12 * r02 + H13 * r03; // d i01

    // (0,2)
    drho_out[6] += (H00 - H22) * i02 + H01 * i12 - H12 * i01 - H03 * i23 - H23 * i03;           // d r02
    drho_out[7] += -H00 * r02 - H01 * r12 + H02 * r0 - H02 * r2 - H03 * r23 + H22 * r02 + H12 * r01 + H23 * r03; // d i02

    // (0,3)
    drho_out[8] += (H00 - H33) * i03 + H01 * i13 + H02 * i23 - H13 * i01 - H23 * i02;           // d r03
    drho_out[9] += -H00 * r03 - H01 * r13 - H02 * r23 + H03 * r0 - H03 * r3 + H33 * r03 + H13 * r01 + H23 * r02; // d i03

    // (1,2)
    drho_out[10] += (H11 - H22) * i12 + H01 * i02 + H02 * i01 - H13 * i23 - H23 * i13;           // d r12
    drho_out[11] += -H11 * r12 - H01 * r02 + H12 * r1 - H12 * r2 - H13 * r23 + H22 * r12 + H02 * r01 + H23 * r13; // d i12

    // (1,3)
    drho_out[12] += (H11 - H33) * i13 + H01 * i03 + H03 * i01 + H12 * i23 - H23 * i12;           // d r13
    drho_out[13] += -H11 * r13 - H12 * r23 - H01 * r03 + H03 * r01 + H13 * r1 - H13 * r3 + H33 * r13 + H23 * r12; // d i13

    // (2,3)
    drho_out[14] += (H22 - H33) * i23 + H02 * i03 + H03 * i02 + H12 * i13 + H13 * i12;           // d r23
    drho_out[15] += -H22 * r23 - H12 * r13 - H02 * r03 + H03 * r02 + H23 * r2 - H23 * r3 + H33 * r23 + H13 * r12; // d i23

}
*/
