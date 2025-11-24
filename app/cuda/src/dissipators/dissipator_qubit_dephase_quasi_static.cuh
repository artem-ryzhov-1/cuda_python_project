// src/dissipators/dissipator_qubit_dephase_quasi_static.cuh

#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "constants.cuh"




__device__ __forceinline__
void dissipator_qubit_dephase_quasi_static_unrolled(

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

    //float& drho_out_D_phi_r00,                           // drho_out_0
    float& drho_out_D_phi_r11, float& drho_out_D_phi_r22,   // drho_out_1, drho_out_2
    //float& drho_out_D_phi_r33,                           // drho_out_3
    float& drho_out_D_phi_r01, float& drho_out_D_phi_i01,   // drho_out_4, drho_out_5
    float& drho_out_D_phi_r02, float& drho_out_D_phi_i02,   // drho_out_6, drho_out_7
    //float& drho_out_D_phi_r03, float& drho_out_D_phi_i03, // drho_out_8, drho_out_9
    float& drho_out_D_phi_r12, float& drho_out_D_phi_i12,   // drho_out_10, drho_out_11
    float& drho_out_D_phi_r13, float& drho_out_D_phi_i13,   // drho_out_12, drho_out_13
    float& drho_out_D_phi_r23, float& drho_out_D_phi_i23,   // drho_out_14, drho_out_15

    const float half_inv_radical,
    const float gp_sqr,
    const float gm_sqr,
    const float gp_gm
) {

    //const float gp_sqr = 0.5f + eps_t_substep * half_inv_radical;
    //const float gm_sqr = 0.5f - eps_t_substep * half_inv_radical;
    //const float gp_gm = delta_C * half_inv_radical;


    const float Gamma_phi_loc = Gamma_phi0 * fabsf(eps_t_substep) * 2.0f * half_inv_radical;




    register float tmp;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.25f;
    //drho_out_D_phi_r00 += tmp;


    tmp = 0.0f;
    tmp = -gp_gm * gp_gm * r11;
    tmp += -r12 * gm_sqr * gp_gm;
    tmp += r12 * gp_gm * gp_sqr;
    tmp += gp_gm * gp_gm * r22;
    tmp *= 2.0f * Gamma_phi_loc;
    drho_out_D_phi_r11 += tmp;


    tmp = 0.0f;
    tmp = -gp_gm * gp_gm * r22;
    tmp += r12 * gm_sqr * gp_gm;
    tmp += -r12 * gp_gm * gp_sqr;
    tmp += gp_gm * gp_gm * r11;
    tmp *= 2.0f * Gamma_phi_loc;
    drho_out_D_phi_r22 += tmp;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.25f;
    //drho_out_D_phi_r33 += tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * r01;
    tmp *= 0.25f;
    drho_out_D_phi_r01 += tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * i01;
    tmp *= 0.25f;
    drho_out_D_phi_i01 += tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * r02;
    tmp *= 0.25f;
    drho_out_D_phi_r02 += tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * i02;
    tmp *= 0.25f;
    drho_out_D_phi_i02 += tmp;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.25f;
    //drho_out_D_phi_r03 += tmp;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.25f;
    //drho_out_D_phi_i03 += tmp;


    tmp = 0.0f;
    //tmp = r12 * (-gm_sqr * gm_sqr + 2 * gp_gm * gp_gm - gp_sqr * gp_sqr); can be simplified
    tmp = -r12 * gm_sqr * gm_sqr;
    tmp += 2.0f * r12 * gp_gm * gp_gm;
    tmp += -r12 * gp_sqr * gp_sqr;
    tmp += -gp_gm * r11 * gm_sqr;
    tmp += gp_gm * r11 * gp_sqr;
    tmp += gp_gm * r22 * gm_sqr;
    tmp += -gp_gm * r22 * gp_sqr;
    tmp *= Gamma_phi_loc;
    drho_out_D_phi_r12 += tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * i12;
    tmp *= 1.0f;
    drho_out_D_phi_i12 += tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * r13;
    tmp *= 0.25f;
    drho_out_D_phi_r13 += tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * i13;
    tmp *= 0.25f;
    drho_out_D_phi_i13 += tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * r23;
    tmp *= 0.25f;
    drho_out_D_phi_r23 += tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * i23;
    tmp *= 0.25f;
    drho_out_D_phi_i23 += tmp;



}






__device__ __forceinline__
void dissipator_qubit_dephase_quasi_static_unrolled_log(

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

    //float& drho_out_D_phi_r00,                           // drho_out_0
    float& drho_out_D_phi_r11, float& drho_out_D_phi_r22,   // drho_out_1, drho_out_2
    //float& drho_out_D_phi_r33,                           // drho_out_3
    float& drho_out_D_phi_r01, float& drho_out_D_phi_i01,   // drho_out_4, drho_out_5
    float& drho_out_D_phi_r02, float& drho_out_D_phi_i02,   // drho_out_6, drho_out_7
    //float& drho_out_D_phi_r03, float& drho_out_D_phi_i03, // drho_out_8, drho_out_9
    float& drho_out_D_phi_r12, float& drho_out_D_phi_i12,   // drho_out_10, drho_out_11
    float& drho_out_D_phi_r13, float& drho_out_D_phi_i13,   // drho_out_12, drho_out_13
    float& drho_out_D_phi_r23, float& drho_out_D_phi_i23,   // drho_out_14, drho_out_15

    const float half_inv_radical,
    const float gp_sqr,
    const float gm_sqr,
    const float gp_gm,

    // log
    LogEntry* __restrict__ d_log_buffer,
    const int t_idx_substep
) {

    //const float gp_sqr = 0.5f + eps_t_substep * half_inv_radical;
    //const float gm_sqr = 0.5f - eps_t_substep * half_inv_radical;
    //const float gp_gm = delta_C * half_inv_radical;

    const float Gamma_phi_loc = Gamma_phi0 * fabsf(eps_t_substep) * 2.0f * half_inv_radical;

    d_log_buffer[t_idx_substep].Gamma_phi = Gamma_phi_loc;


    //d_log_buffer[t_idx_substep].debug_eps_t_substep = eps_t_substep;
    //d_log_buffer[t_idx_substep].debug_delta_C = delta_C;
    //d_log_buffer[t_idx_substep].debug_radical = radical;


    register float tmp;







    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.25f;
    //drho_out_D_phi_r00 += tmp;
    //d_log_buffer[t_idx_substep].drho_out_D_phi_r00 = tmp;


    tmp = 0.0f;
    tmp = -gp_gm * gp_gm * r11;
    tmp += -r12 * gm_sqr * gp_gm;
    tmp += r12 * gp_gm * gp_sqr;
    tmp += gp_gm * gp_gm * r22;
    tmp *= 2.0f * Gamma_phi_loc;
    drho_out_D_phi_r11 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_phi_r11 = tmp;


    tmp = 0.0f;
    tmp = -gp_gm * gp_gm * r22;
    tmp += r12 * gm_sqr * gp_gm;
    tmp += -r12 * gp_gm * gp_sqr;
    tmp += gp_gm * gp_gm * r11;
    tmp *= 2.0f * Gamma_phi_loc;
    drho_out_D_phi_r22 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_phi_r22 = tmp;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.25f;
    //drho_out_D_phi_r33 += tmp;
    //d_log_buffer[t_idx_substep].drho_out_D_phi_r33 = tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * r01;
    tmp *= 0.25f;
    drho_out_D_phi_r01 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_phi_r01 = tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * i01;
    tmp *= 0.25f;
    drho_out_D_phi_i01 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_phi_i01 = tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * r02;
    tmp *= 0.25f;
    drho_out_D_phi_r02 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_phi_r02 = tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * i02;
    tmp *= 0.25f;
    drho_out_D_phi_i02 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_phi_i02 = tmp;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.25f;
    //drho_out_D_phi_r03 += tmp;
    //d_log_buffer[t_idx_substep].drho_out_D_phi_r03 = tmp;


    //tmp = 0.0f;
    //tmp = 0;
    //tmp *= 0.25f;
    //drho_out_D_phi_i03 += tmp;
    //d_log_buffer[t_idx_substep].drho_out_D_phi_i03 = tmp;


    tmp = 0.0f;
    //tmp = r12 * (-gm_sqr * gm_sqr + 2 * gp_gm * gp_gm - gp_sqr * gp_sqr); can be simplified
    tmp = -r12 * gm_sqr * gm_sqr;
    tmp += 2.0f * r12 * gp_gm * gp_gm;
    tmp += -r12 * gp_sqr * gp_sqr;
    tmp += -gp_gm * r11 * gm_sqr;
    tmp += gp_gm * r11 * gp_sqr;
    tmp += gp_gm * r22 * gm_sqr;
    tmp += -gp_gm * r22 * gp_sqr;
    tmp *= Gamma_phi_loc;
    drho_out_D_phi_r12 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_phi_r12 = tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * i12;
    tmp *= 1.0f;
    drho_out_D_phi_i12 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_phi_i12 = tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * r13;
    tmp *= 0.25f;
    drho_out_D_phi_r13 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_phi_r13 = tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * i13;
    tmp *= 0.25f;
    drho_out_D_phi_i13 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_phi_i13 = tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * r23;
    tmp *= 0.25f;
    drho_out_D_phi_r23 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_phi_r23 = tmp;


    tmp = 0.0f;
    tmp = -Gamma_phi_loc * i23;
    tmp *= 0.25f;
    drho_out_D_phi_i23 += tmp;
    d_log_buffer[t_idx_substep].drho_out_D_phi_i23 = tmp;



}