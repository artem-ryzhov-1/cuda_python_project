////////////////////////////////////////
// app/cuda/src/dissipators/dissipator_qubit_relax.cuh
////////////////////////////////////////

#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "constants.cuh"




__device__ __forceinline__
void dissipator_qubit_relax_unrolled(

    const float eps_t_substep,

    //const float r00,                  // rho_in_0
    const float r11, const float r22,   // rho_in_1, rho_in_2
    //const float r33,                  // rho_in_3
    const float r01, const float i01,   // rho_in_4, rho_in_5
    const float r02, const float i02,   // rho_in_6, rho_in_7
    //const float r03, const float i03, // rho_in_8, rho_in_9
    const float r12, const float i12,   // rho_in_10, rho_in_11
    const float r13, const float i13,   // rho_in_12, rho_in_13
    const float r23, const float i23,   // rho_in_14, rho_in_15

    //float& drho_out_D_eg_r00,                           // drho_out_0
    float& drho_out_D_eg_r11, float& drho_out_D_eg_r22,   // drho_out_1, drho_out_2
    //float& drho_out_D_eg_r33,                           // drho_out_3
    float& drho_out_D_eg_r01, float& drho_out_D_eg_i01,   // drho_out_4, drho_out_5
    float& drho_out_D_eg_r02, float& drho_out_D_eg_i02,   // drho_out_6, drho_out_7
    //float& drho_out_D_eg_r03, float& drho_out_D_eg_i03, // drho_out_8, drho_out_9
    float& drho_out_D_eg_r12, float& drho_out_D_eg_i12,   // drho_out_10, drho_out_11
    float& drho_out_D_eg_r13, float& drho_out_D_eg_i13,   // drho_out_12, drho_out_13
    float& drho_out_D_eg_r23, float& drho_out_D_eg_i23,   // drho_out_14, drho_out_15

    const float half_inv_radical,
    const float gp_sqr,
    const float gm_sqr,
    const float gp_gm
) {

    // option 1: without exp
    // const float Gamma_eg_loc = Gamma_eg0 * radical / delta_C; // add normalization factor

    // option 2: with exp
    const float radical_div_delta_C = 0.5f / (half_inv_radical * delta_C);
    const float Gamma_eg_half_loc = 0.5f * Gamma_eg0_norm * radical_div_delta_C * expf(-beta * radical_div_delta_C * radical_div_delta_C);

    //const float Gamma_eg_loc = Gamma_eg0_norm;

    register float tmp;

    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.5f;
    //drho_out_D_eg_r00 += tmp;


    tmp = 0.0f;
    tmp = -gm_sqr * gm_sqr * r11;
    tmp += gp_sqr * gp_sqr * r22;
    tmp += gp_gm * r12 * gm_sqr;
    tmp += -gp_gm * r12 * gp_sqr;
    tmp *= 2.0f * Gamma_eg_half_loc;
    drho_out_D_eg_r11 += tmp;


    tmp = 0.0f;
    tmp = -gp_sqr * gp_sqr * r22;
    tmp += gm_sqr * gm_sqr * r11;
    tmp += -gp_gm * r12 * gm_sqr;
    tmp += gp_gm * r12 * gp_sqr;
    tmp *= 2.0f * Gamma_eg_half_loc;
    drho_out_D_eg_r22 += tmp;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.5f;
    //drho_out_D_eg_r33 += tmp;


    tmp = 0.0f;
    tmp = gp_gm * r02;
    tmp += -gm_sqr * r01;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_r01 += tmp;


    tmp = 0.0f;
    tmp = gp_gm * i02;
    tmp += -gm_sqr * i01;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_i01 += tmp;


    tmp = 0.0f;
    tmp = gp_gm * r01;
    tmp += -gp_sqr * r02;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_r02 += tmp;


    tmp = 0.0f;
    tmp = gp_gm * i01;
    tmp += -gp_sqr * i02;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_i02 += tmp;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.5f;
    //drho_out_D_eg_r03 += tmp;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.5f;
    //drho_out_D_eg_i03 += tmp;


    tmp = 0.0f;
    tmp = -4.0f * r12 * gp_gm * gp_gm;
    tmp += -r12;
    tmp += 2.0f * gp_gm * r11 * gm_sqr;
    tmp += gp_gm * r11;
    tmp += 2.0f * gp_gm * r22 * gp_sqr;
    tmp += gp_gm * r22;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_r12 += tmp;


    tmp = 0.0f;
    tmp = -i12;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_i12 += tmp;


    tmp = 0.0f;
    tmp = gp_gm * r23;
    tmp += -gm_sqr * r13;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_r13 += tmp;


    tmp = 0.0f;
    tmp = gp_gm * i23;
    tmp += -gm_sqr * i13;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_i13 += tmp;


    tmp = 0.0f;
    tmp = gp_gm * r13;
    tmp += -gp_sqr * r23;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_r23 += tmp;


    tmp = 0.0f;
    tmp = gp_gm * i13;
    tmp += -gp_sqr * i23;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_i23 += tmp;


}


__device__ __forceinline__
void dissipator_qubit_relax_unrolled_log(

    const float eps_t_substep,

    //const float r00,                  // rho_in_0
    const float r11, const float r22,   // rho_in_1, rho_in_2
    //const float r33,                  // rho_in_3
    const float r01, const float i01,   // rho_in_4, rho_in_5
    const float r02, const float i02,   // rho_in_6, rho_in_7
    //const float r03, const float i03, // rho_in_8, rho_in_9
    const float r12, const float i12,   // rho_in_10, rho_in_11
    const float r13, const float i13,   // rho_in_12, rho_in_13
    const float r23, const float i23,   // rho_in_14, rho_in_15

    //float& drho_out_D_eg_r00,                           // drho_out_0
    float& drho_out_D_eg_r11, float& drho_out_D_eg_r22,   // drho_out_1, drho_out_2
    //float& drho_out_D_eg_r33,                           // drho_out_3
    float& drho_out_D_eg_r01, float& drho_out_D_eg_i01,   // drho_out_4, drho_out_5
    float& drho_out_D_eg_r02, float& drho_out_D_eg_i02,   // drho_out_6, drho_out_7
    //float& drho_out_D_eg_r03, float& drho_out_D_eg_i03, // drho_out_8, drho_out_9
    float& drho_out_D_eg_r12, float& drho_out_D_eg_i12,   // drho_out_10, drho_out_11
    float& drho_out_D_eg_r13, float& drho_out_D_eg_i13,   // drho_out_12, drho_out_13
    float& drho_out_D_eg_r23, float& drho_out_D_eg_i23,   // drho_out_14, drho_out_15

    const float half_inv_radical,
    const float gp_sqr,
    const float gm_sqr,
    const float gp_gm,

    // log
    LogEntry* __restrict__ d_log_buffer,
    const int t_idx_substep
) {

    // option 1: without exp
    // const float Gamma_eg_loc = Gamma_eg0 * radical / delta_C; // add normalization factor

    // option 2: with exp
    const float radical_div_delta_C = 0.5f /(half_inv_radical * delta_C);
    const float Gamma_eg_half_loc = 0.5f * Gamma_eg0_norm * radical_div_delta_C * expf(-beta * radical_div_delta_C * radical_div_delta_C);


    d_log_buffer[t_idx_substep].Gamma_eg = 2.0f * Gamma_eg_half_loc;
    

    d_log_buffer[t_idx_substep].debug_eps_t_substep = eps_t_substep;
    d_log_buffer[t_idx_substep].debug_delta_C = delta_C;
    d_log_buffer[t_idx_substep].debug_radical = 0.5f / half_inv_radical;
    d_log_buffer[t_idx_substep].debig_radical_div_delta_C = radical_div_delta_C;
    d_log_buffer[t_idx_substep].debug_Gamma_eg0_norm = Gamma_eg0_norm;
    d_log_buffer[t_idx_substep].debug_beta = beta;
    d_log_buffer[t_idx_substep].debug_Gamma_eg_loc = 2.0f * Gamma_eg_half_loc;


    register float tmp;

    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.5f;
    //drho_out_D_eg_r00 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_r00 = 0.0f;

    
    tmp = 0.0f;
    tmp = -gm_sqr * gm_sqr * r11;
    tmp += gp_sqr * gp_sqr * r22;
    tmp += gp_gm * r12 * gm_sqr;
    tmp += -gp_gm * r12 * gp_sqr;
    tmp *= 2.0f * Gamma_eg_half_loc;
    drho_out_D_eg_r11 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_r11 = tmp;


    tmp = 0.0f;
    tmp = -gp_sqr * gp_sqr * r22;
    tmp += gm_sqr * gm_sqr * r11;
    tmp += -gp_gm * r12 * gm_sqr;
    tmp += gp_gm * r12 * gp_sqr;
    tmp *= 2.0f * Gamma_eg_half_loc;
    drho_out_D_eg_r22 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_r22 = tmp;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.5f;
    //drho_out_D_eg_r33 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_r33 = 0.0f;


    tmp = 0.0f;
    tmp = gp_gm * r02;
    tmp += -gm_sqr * r01;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_r01 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_r01 = tmp;


    tmp = 0.0f;
    tmp = gp_gm * i02;
    tmp += -gm_sqr * i01;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_i01 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_i01 = tmp;


    tmp = 0.0f;
    tmp = gp_gm * r01;
    tmp += -gp_sqr * r02;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_r02 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_r02 = tmp;


    tmp = 0.0f;
    tmp = gp_gm * i01;
    tmp += -gp_sqr * i02;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_i02 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_i02 = tmp;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.5f;
    //drho_out_D_eg_r03 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_r03 = 0.0f;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.5f;
    //drho_out_D_eg_i03 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_i03 = 0.0f;


    tmp = 0.0f;
    tmp = -4.0f * r12 * gp_gm * gp_gm;
    tmp += -r12;
    tmp += 2.0f * gp_gm * r11 * gm_sqr;
    tmp += gp_gm * r11;
    tmp += 2.0f * gp_gm * r22 * gp_sqr;
    tmp += gp_gm * r22;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_r12 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_r12 = tmp;


    tmp = 0.0f;
    tmp = -i12;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_i12 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_i12 = tmp;


    tmp = 0.0f;
    tmp = gp_gm * r23;
    tmp += -gm_sqr * r13;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_r13 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_r13 = tmp;


    tmp = 0.0f;
    tmp = gp_gm * i23;
    tmp += -gm_sqr * i13;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_i13 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_i13 = tmp;


    tmp = 0.0f;
    tmp = gp_gm * r13;
    tmp += -gp_sqr * r23;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_r23 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_r23 = tmp;


    tmp = 0.0f;
    tmp = gp_gm * i13;
    tmp += -gp_sqr * i23;
    tmp *= Gamma_eg_half_loc;
    drho_out_D_eg_i23 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_eg_i23 = tmp;

}