////////////////////////////////////////
// app/cuda/src/dissipators/dissipator_dot_lead_sparse.cuh
////////////////////////////////////////

#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "constants.cuh"







static __device__ __forceinline__
void dissipator_dot_lead_unrolled_interval_1(

    const float eps_t_substep,

    const float r00, const float r11, const float r22, const float r33,
    const float r01, const float i01, const float r02, const float i02,
    const float r03, const float i03, const float r12, const float i12,
    const float r13, const float i13, const float r23, const float i23,

    float& drho_out_D_dl_r00, float& drho_out_D_dl_r11, float& drho_out_D_dl_r22,
    float& drho_out_D_dl_r33, float& drho_out_D_dl_r01, float& drho_out_D_dl_i01,
    float& drho_out_D_dl_r02, float& drho_out_D_dl_i02, float& drho_out_D_dl_r03,
    float& drho_out_D_dl_i03, float& drho_out_D_dl_r12, float& drho_out_D_dl_i12,
    float& drho_out_D_dl_r13, float& drho_out_D_dl_i13,
    float& drho_out_D_dl_r23, float& drho_out_D_dl_i23,

    const float gp_sqr,
    const float gm_sqr,
    const float gp_gm,
    const float Gamma_LR0_loc,
    const float Gamma_lprm,
    const float Gamma_lmrp

) {


    /*
    float Gamma_10 = 0.0f;
    float Gamma_20 = 0.0f;
    //float Gamma_30 = 0.0f;
    //float Gamma_21 = 0.0f;
    float Gamma_31 = 0.0f;
    float Gamma_32 = 0.0f;

    // Interval 1
    //if (eps_t_substep < -epsilon_L)
    {
        Gamma_10 = compute_W(U, 1, 0);  // Delta N = -1
        Gamma_20 = compute_W(U, 2, 0);  // Delta N = -1
        //Gamma_30 = 0.0f;                // Delta N = -2 -> forbidden
        //Gamma_21 = 0.0f;                // Delta N =  0 -> forbidden
        Gamma_31 = compute_W(U, 3, 1);  // Delta N = -1
        Gamma_32 = compute_W(U, 3, 2);  // Delta N = -1
    }
    */


    //const float inv_denom = 0.5f / sqrtf(delta_C * delta_C + eps_t_substep * eps_t_substep);

    //const float gp_sqr = 0.5f + eps_t_substep * inv_denom;
    //const float gm_sqr = 0.5f - eps_t_substep * inv_denom;
    //const float gp_gm = delta_C * inv_denom;

    //const float Gamma_LR0_loc = Gamma_LR0;

    //const float Gamma_lprm = Gamma_L0 * gp_sqr + Gamma_R0 * gm_sqr;
    //const float Gamma_lmrp = Gamma_L0 * gm_sqr + Gamma_R0 * gp_sqr;



    register float tmp;



    tmp = 0.0f;
    tmp = 2 * r11 * (Gamma_lmrp * gp_sqr + Gamma_lprm * gm_sqr);
    tmp += 2 * r12 * (2 * Gamma_lmrp * gp_gm - 2 * Gamma_lprm * gp_gm);
    tmp += 2 * r22 * (Gamma_lmrp * gm_sqr + Gamma_lprm * gp_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r00 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r11 * (-Gamma_lmrp * gp_sqr - Gamma_lprm * gm_sqr);
    tmp += 2 * r12 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp += 2 * r33 * (Gamma_lmrp * gm_sqr + Gamma_lprm * gp_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r11 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r12 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp += 2 * r22 * (-Gamma_lmrp * gm_sqr - Gamma_lprm * gp_sqr);
    tmp += 2 * r33 * (Gamma_lmrp * gp_sqr + Gamma_lprm * gm_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r22 += tmp;
    


    tmp = 0.0f;
    tmp = -2 * Gamma_LR0_loc * r33;
    tmp *= 0.5f;
    drho_out_D_dl_r33 += tmp;
    


    tmp = 0.0f;
    tmp = r01 * (-Gamma_lmrp * gp_sqr - Gamma_lprm * gm_sqr);
    tmp += r02 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_r01 += tmp;
    


    tmp = 0.0f;
    tmp = i01 * (-Gamma_lmrp * gp_sqr - Gamma_lprm * gm_sqr);
    tmp += i02 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_i01 += tmp;
    


    tmp = 0.0f;
    tmp = r01 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp += r02 * (-Gamma_lmrp * gm_sqr - Gamma_lprm * gp_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r02 += tmp;
    


    tmp = 0.0f;
    tmp = i01 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp += i02 * (-Gamma_lmrp * gm_sqr - Gamma_lprm * gp_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_i02 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * r03;
    tmp *= 0.5f;
    drho_out_D_dl_r03 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * i03;
    tmp *= 0.5f;
    drho_out_D_dl_i03 += tmp;
    


    tmp = 0.0f;
    tmp = r11 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp += r22 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp += -Gamma_LR0_loc * r12;
    tmp += 2 * r33 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_r12 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * i12;
    tmp *= 0.5f;
    drho_out_D_dl_i12 += tmp;
    


    tmp = 0.0f;
    tmp = r13 * (-Gamma_LR0_loc - Gamma_lmrp * gp_sqr - Gamma_lprm * gm_sqr);
    tmp += r23 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_r13 += tmp;
    


    tmp = 0.0f;
    tmp = i13 * (-Gamma_LR0_loc - Gamma_lmrp * gp_sqr - Gamma_lprm * gm_sqr);
    tmp += i23 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_i13 += tmp;
    


    tmp = 0.0f;
    tmp = r13 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp += r23 * (-Gamma_LR0_loc - Gamma_lmrp * gm_sqr - Gamma_lprm * gp_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r23 += tmp;
    


    tmp = 0.0f;
    tmp = i13 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp += i23 * (-Gamma_LR0_loc - Gamma_lmrp * gm_sqr - Gamma_lprm * gp_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_i23 += tmp;
    


}









static __device__ __forceinline__
void dissipator_dot_lead_unrolled_interval_2(

    const float eps_t_substep,

    const float r00, const float r11, const float r22, const float r33,
    const float r01, const float i01, const float r02, const float i02,
    const float r03, const float i03, const float r12, const float i12,
    const float r13, const float i13, const float r23, const float i23,

    float& drho_out_D_dl_r00, float& drho_out_D_dl_r11, float& drho_out_D_dl_r22,
    float& drho_out_D_dl_r33, float& drho_out_D_dl_r01, float& drho_out_D_dl_i01,
    float& drho_out_D_dl_r02, float& drho_out_D_dl_i02, float& drho_out_D_dl_r03,
    float& drho_out_D_dl_i03, float& drho_out_D_dl_r12, float& drho_out_D_dl_i12,
    float& drho_out_D_dl_r13, float& drho_out_D_dl_i13,
    float& drho_out_D_dl_r23, float& drho_out_D_dl_i23,

    const float gp_sqr,
    const float gm_sqr,
    const float gp_gm,
    const float Gamma_LR0_loc,
    const float Gamma_lprm,
    const float Gamma_lmrp

) {


    /*
    float Gamma_10 = 0.0f;
    //float Gamma_20 = 0.0f;
    float Gamma_30 = 0.0f;
    float Gamma_21 = 0.0f;
    //float Gamma_31 = 0.0f;
    float Gamma_32 = 0.0f;

    // Interval 2
    //if (eps_t_substep < -epsilon_R)
    {
        Gamma_10 = compute_W(U, 0, 1);  // Delta N = +1 -> reverse
        //Gamma_20 = 0.0f;                // Delta N = +2 -> forbidden
        Gamma_30 = compute_W(U, 3, 0);  // Delta N = -1
        Gamma_21 = compute_W(U, 2, 1);  // Delta N = -1
        //Gamma_31 = 0.0f;                // Delta N = +2 -> forbidden
        Gamma_32 = compute_W(U, 3, 2);  // Delta N = -1
    }
    */


    //const float inv_denom = 0.5f / sqrtf(delta_C * delta_C + eps_t_substep * eps_t_substep);

    //const float gp_sqr = 0.5f + eps_t_substep * inv_denom;
    //const float gm_sqr = 0.5f - eps_t_substep * inv_denom;
    //const float gp_gm = delta_C * inv_denom;

    //const float Gamma_LR0_loc = Gamma_LR0;

    //const float Gamma_lprm = Gamma_L0 * gp_sqr + Gamma_R0 * gm_sqr;
    //const float Gamma_lmrp = Gamma_L0 * gm_sqr + Gamma_R0 * gp_sqr;


    register float tmp;



    tmp = 0.0f;
    tmp = -2 * Gamma_lmrp * r00;
    tmp += -4 * Gamma_lprm * gp_gm * r12;
    tmp += 2 * Gamma_lprm * gm_sqr * r11;
    tmp += 2 * Gamma_lprm * gp_sqr * r22;
    tmp *= 0.5f;
    drho_out_D_dl_r00 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r33 * (Gamma_lmrp * gm_sqr + Gamma_lprm * gp_sqr);
    tmp += -2 * Gamma_lprm * gm_sqr * r11;
    tmp += 2 * Gamma_lmrp * gp_sqr * r00;
    tmp += 2 * Gamma_lprm * gp_gm * r12;
    tmp *= 0.5f;
    drho_out_D_dl_r11 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r33 * (Gamma_lmrp * gp_sqr + Gamma_lprm * gm_sqr);
    tmp += -2 * Gamma_lprm * gp_sqr * r22;
    tmp += 2 * Gamma_lmrp * gm_sqr * r00;
    tmp += 2 * Gamma_lprm * gp_gm * r12;
    tmp *= 0.5f;
    drho_out_D_dl_r22 += tmp;
    


    tmp = 0.0f;
    tmp = -2 * Gamma_LR0_loc * r33;
    tmp *= 0.5f;
    drho_out_D_dl_r33 += tmp;
    


    tmp = 0.0f;
    tmp = r01 * (-Gamma_lmrp - Gamma_lprm * gm_sqr);
    tmp += Gamma_lprm * gp_gm * r02;
    tmp *= 0.5f;
    drho_out_D_dl_r01 += tmp;
    


    tmp = 0.0f;
    tmp = i01 * (-Gamma_lmrp - Gamma_lprm * gm_sqr);
    tmp += Gamma_lprm * gp_gm * i02;
    tmp *= 0.5f;
    drho_out_D_dl_i01 += tmp;
    


    tmp = 0.0f;
    tmp = r02 * (-Gamma_lmrp - Gamma_lprm * gp_sqr);
    tmp += Gamma_lprm * gp_gm * r01;
    tmp *= 0.5f;
    drho_out_D_dl_r02 += tmp;
    


    tmp = 0.0f;
    tmp = i02 * (-Gamma_lmrp - Gamma_lprm * gp_sqr);
    tmp += Gamma_lprm * gp_gm * i01;
    tmp *= 0.5f;
    drho_out_D_dl_i02 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_lprm * r03;
    tmp += -2 * Gamma_lmrp * r03;
    tmp *= 0.5f;
    drho_out_D_dl_r03 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_lprm * i03;
    tmp += -2 * Gamma_lmrp * i03;
    tmp *= 0.5f;
    drho_out_D_dl_i03 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_lprm * r12;
    tmp += 2 * r33 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp += Gamma_lprm * gp_gm * r11;
    tmp += Gamma_lprm * gp_gm * r22;
    tmp += 2 * Gamma_lmrp * gp_gm * r00;
    tmp *= 0.5f;
    drho_out_D_dl_r12 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_lprm * i12;
    tmp *= 0.5f;
    drho_out_D_dl_i12 += tmp;
    


    tmp = 0.0f;
    tmp = r13 * (-Gamma_LR0_loc - Gamma_lprm * gm_sqr);
    tmp += Gamma_lprm * gp_gm * r23;
    tmp *= 0.5f;
    drho_out_D_dl_r13 += tmp;
    


    tmp = 0.0f;
    tmp = i13 * (-Gamma_LR0_loc - Gamma_lprm * gm_sqr);
    tmp += Gamma_lprm * gp_gm * i23;
    tmp *= 0.5f;
    drho_out_D_dl_i13 += tmp;
    


    tmp = 0.0f;
    tmp = r23 * (-Gamma_LR0_loc - Gamma_lprm * gp_sqr);
    tmp += Gamma_lprm * gp_gm * r13;
    tmp *= 0.5f;
    drho_out_D_dl_r23 += tmp;
    


    tmp = 0.0f;
    tmp = i23 * (-Gamma_LR0_loc - Gamma_lprm * gp_sqr);
    tmp += Gamma_lprm * gp_gm * i13;
    tmp *= 0.5f;
    drho_out_D_dl_i23 += tmp;
    


}






static __device__ __forceinline__
void dissipator_dot_lead_unrolled_interval_3(

    const float eps_t_substep,

    const float r00, const float r11, const float r22, const float r33,
    const float r01, const float i01, const float r02, const float i02,
    const float r03, const float i03, const float r12, const float i12,
    const float r13, const float i13, const float r23, const float i23,

    float& drho_out_D_dl_r00, float& drho_out_D_dl_r11, float& drho_out_D_dl_r22,
    float& drho_out_D_dl_r33, float& drho_out_D_dl_r01, float& drho_out_D_dl_i01,
    float& drho_out_D_dl_r02, float& drho_out_D_dl_i02, float& drho_out_D_dl_r03,
    float& drho_out_D_dl_i03, float& drho_out_D_dl_r12, float& drho_out_D_dl_i12,
    float& drho_out_D_dl_r13, float& drho_out_D_dl_i13,
    float& drho_out_D_dl_r23, float& drho_out_D_dl_i23,

    const float gp_sqr,
    const float gm_sqr,
    const float gp_gm,
    const float Gamma_LR0_loc,
    const float Gamma_lprm,
    const float Gamma_lmrp

) {


    /*
    //float Gamma_10 = 0.0f;
    float Gamma_20 = 0.0f;
    float Gamma_30 = 0.0f;
    float Gamma_21 = 0.0f;
    float Gamma_31 = 0.0f;
    //float Gamma_32 = 0.0f;

    // Interval 3
    //if (eps_t_substep < 0.0f)
    {
        //Gamma_10 = 0.0f;                // Delta N = 0
        Gamma_20 = compute_W(U, 0, 2);  // Delta N = +1 -> reverse
        Gamma_30 = compute_W(U, 3, 0);  // Delta N = -1
        Gamma_21 = compute_W(U, 1, 2);  // Delta N = +1 -> reverse
        Gamma_31 = compute_W(U, 3, 1);  // Delta N = -1
        //Gamma_32 = 0.0f;                // Delta N = 0
    }
    */


    //const float inv_denom = 0.5f / sqrtf(delta_C * delta_C + eps_t_substep * eps_t_substep);

    //const float gp_sqr = 0.5f + eps_t_substep * inv_denom;
    //const float gm_sqr = 0.5f - eps_t_substep * inv_denom;
    //const float gp_gm = delta_C * inv_denom;

    //const float Gamma_LR0_loc = Gamma_LR0;

    //const float Gamma_lprm = Gamma_L0 * gp_sqr + Gamma_R0 * gm_sqr;
    //const float Gamma_lmrp = Gamma_L0 * gm_sqr + Gamma_R0 * gp_sqr;



    register float tmp;



    tmp = 0.0f;
    tmp = -2 * Gamma_LR0_loc * r00;
    tmp *= 0.5f;
    drho_out_D_dl_r00 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r00 * (Gamma_lmrp * gp_sqr + Gamma_lprm * gm_sqr);
    tmp += 2 * r33 * (Gamma_lmrp * gm_sqr + Gamma_lprm * gp_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r11 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r00 * (Gamma_lmrp * gm_sqr + Gamma_lprm * gp_sqr);
    tmp += 2 * r33 * (Gamma_lmrp * gp_sqr + Gamma_lprm * gm_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r22 += tmp;
    


    tmp = 0.0f;
    tmp = -2 * Gamma_LR0_loc * r33;
    tmp *= 0.5f;
    drho_out_D_dl_r33 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * r01;
    tmp *= 0.5f;
    drho_out_D_dl_r01 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * i01;
    tmp *= 0.5f;
    drho_out_D_dl_i01 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * r02;
    tmp *= 0.5f;
    drho_out_D_dl_r02 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * i02;
    tmp *= 0.5f;
    drho_out_D_dl_i02 += tmp;
    


    tmp = 0.0f;
    tmp = -2 * Gamma_LR0_loc * r03;
    tmp *= 0.5f;
    drho_out_D_dl_r03 += tmp;
    


    tmp = 0.0f;
    tmp = -2 * Gamma_LR0_loc * i03;
    tmp *= 0.5f;
    drho_out_D_dl_i03 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r00 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp += 2 * r33 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_r12 += tmp;
    


    tmp = 0.0f;
    tmp = 0;
    tmp *= 0.5f;
    drho_out_D_dl_i12 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * r13;
    tmp *= 0.5f;
    drho_out_D_dl_r13 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * i13;
    tmp *= 0.5f;
    drho_out_D_dl_i13 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * r23;
    tmp *= 0.5f;
    drho_out_D_dl_r23 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * i23;
    tmp *= 0.5f;
    drho_out_D_dl_i23 += tmp;
    



}






static __device__ __forceinline__
void dissipator_dot_lead_unrolled_interval_4(

    const float eps_t_substep,

    const float r00, const float r11, const float r22, const float r33,
    const float r01, const float i01, const float r02, const float i02,
    const float r03, const float i03, const float r12, const float i12,
    const float r13, const float i13, const float r23, const float i23,

    float& drho_out_D_dl_r00, float& drho_out_D_dl_r11, float& drho_out_D_dl_r22,
    float& drho_out_D_dl_r33, float& drho_out_D_dl_r01, float& drho_out_D_dl_i01,
    float& drho_out_D_dl_r02, float& drho_out_D_dl_i02, float& drho_out_D_dl_r03,
    float& drho_out_D_dl_i03, float& drho_out_D_dl_r12, float& drho_out_D_dl_i12,
    float& drho_out_D_dl_r13, float& drho_out_D_dl_i13,
    float& drho_out_D_dl_r23, float& drho_out_D_dl_i23,

    const float gp_sqr,
    const float gm_sqr,
    const float gp_gm,
    const float Gamma_LR0_loc,
    const float Gamma_lprm,
    const float Gamma_lmrp

) {


    /*
    //float Gamma_10 = 0.0f;
    float Gamma_20 = 0.0f;
    float Gamma_30 = 0.0f;
    float Gamma_21 = 0.0f;
    float Gamma_31 = 0.0f;
    //float Gamma_32 = 0.0f;

    // Interval 4
    //if (eps_t_substep < epsilon_R)
    {
        //Gamma_10 = 0.0f;                // Delta N = 0
        Gamma_20 = compute_W(U, 2, 0);  // Delta N = -1
        Gamma_30 = compute_W(U, 0, 3);  // Delta N = +1 -> reverse
        Gamma_21 = compute_W(U, 2, 1);  // Delta N = -1
        Gamma_31 = compute_W(U, 1, 3);  // Delta N = +1 -> reverse
        //Gamma_32 = 0.0f;                // Delta N = 0
    }
    */


    //const float inv_denom = 0.5f / sqrtf(delta_C * delta_C + eps_t_substep * eps_t_substep);

    //const float gp_sqr = 0.5f + eps_t_substep * inv_denom;
    //const float gm_sqr = 0.5f - eps_t_substep * inv_denom;
    //const float gp_gm = delta_C * inv_denom;

    //const float Gamma_LR0_loc = Gamma_LR0;

    //const float Gamma_lprm = Gamma_L0 * gp_sqr + Gamma_R0 * gm_sqr;
    //const float Gamma_lmrp = Gamma_L0 * gm_sqr + Gamma_R0 * gp_sqr;


    register float tmp;



    tmp = 0.0f;
    tmp = -2 * Gamma_LR0_loc * r00;
    tmp *= 0.5f;
    drho_out_D_dl_r00 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r00 * (Gamma_lmrp * gp_sqr + Gamma_lprm * gm_sqr);
    tmp += 2 * r33 * (Gamma_lmrp * gm_sqr + Gamma_lprm * gp_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r11 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r00 * (Gamma_lmrp * gm_sqr + Gamma_lprm * gp_sqr);
    tmp += 2 * r33 * (Gamma_lmrp * gp_sqr + Gamma_lprm * gm_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r22 += tmp;
    


    tmp = 0.0f;
    tmp = -2 * Gamma_LR0_loc * r33;
    tmp *= 0.5f;
    drho_out_D_dl_r33 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * r01;
    tmp *= 0.5f;
    drho_out_D_dl_r01 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * i01;
    tmp *= 0.5f;
    drho_out_D_dl_i01 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * r02;
    tmp *= 0.5f;
    drho_out_D_dl_r02 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * i02;
    tmp *= 0.5f;
    drho_out_D_dl_i02 += tmp;
    


    tmp = 0.0f;
    tmp = -2 * Gamma_LR0_loc * r03;
    tmp *= 0.5f;
    drho_out_D_dl_r03 += tmp;
    


    tmp = 0.0f;
    tmp = -2 * Gamma_LR0_loc * i03;
    tmp *= 0.5f;
    drho_out_D_dl_i03 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r00 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp += 2 * r33 * (-Gamma_lmrp * gp_gm + Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_r12 += tmp;
    


    tmp = 0.0f;
    tmp = 0;
    tmp *= 0.5f;
    drho_out_D_dl_i12 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * r13;
    tmp *= 0.5f;
    drho_out_D_dl_r13 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * i13;
    tmp *= 0.5f;
    drho_out_D_dl_i13 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * r23;
    tmp *= 0.5f;
    drho_out_D_dl_r23 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * i23;
    tmp *= 0.5f;
    drho_out_D_dl_i23 += tmp;
    





}










static __device__ __forceinline__
void dissipator_dot_lead_unrolled_interval_5(

    const float eps_t_substep,

    const float r00, const float r11, const float r22, const float r33,
    const float r01, const float i01, const float r02, const float i02,
    const float r03, const float i03, const float r12, const float i12,
    const float r13, const float i13, const float r23, const float i23,

    float& drho_out_D_dl_r00, float& drho_out_D_dl_r11, float& drho_out_D_dl_r22,
    float& drho_out_D_dl_r33, float& drho_out_D_dl_r01, float& drho_out_D_dl_i01,
    float& drho_out_D_dl_r02, float& drho_out_D_dl_i02, float& drho_out_D_dl_r03,
    float& drho_out_D_dl_i03, float& drho_out_D_dl_r12, float& drho_out_D_dl_i12,
    float& drho_out_D_dl_r13, float& drho_out_D_dl_i13,
    float& drho_out_D_dl_r23, float& drho_out_D_dl_i23,

    const float gp_sqr,
    const float gm_sqr,
    const float gp_gm,
    const float Gamma_LR0_loc,
    const float Gamma_lprm,
    const float Gamma_lmrp

) {


    /*
    float Gamma_10 = 0.0f;
    //float Gamma_20 = 0.0f;
    float Gamma_30 = 0.0f;
    float Gamma_21 = 0.0f;
    //float Gamma_31 = 0.0f;
    float Gamma_32 = 0.0f;

    // Interval 5
    //if (eps_t_substep < epsilon_L)
    {
        Gamma_10 = compute_W(U, 1, 0);  // Delta N = -1
        //Gamma_20 = 0.0f;                // Delta N = +2
        Gamma_30 = compute_W(U, 0, 3);  // Delta N = +1 -> reverse
        Gamma_21 = compute_W(U, 1, 2);  // Delta N = +1 -> reverse
        //Gamma_31 = 0.0f;                // Delta N = +2
        Gamma_32 = compute_W(U, 2, 3);  // Delta N = +1 -> reverse
    }
    */



    //const float inv_denom = 0.5f / sqrtf(delta_C * delta_C + eps_t_substep * eps_t_substep);

    //const float gp_sqr = 0.5f + eps_t_substep * inv_denom;
    //const float gm_sqr = 0.5f - eps_t_substep * inv_denom;
    //const float gp_gm = delta_C * inv_denom;

    //const float Gamma_LR0_loc = Gamma_LR0;

    //const float Gamma_lprm = Gamma_L0 * gp_sqr + Gamma_R0 * gm_sqr;
    //const float Gamma_lmrp = Gamma_L0 * gm_sqr + Gamma_R0 * gp_sqr;


    register float tmp;



    tmp = 0.0f;
    tmp = -2 * Gamma_LR0_loc * r00;
    tmp *= 0.5f;
    drho_out_D_dl_r00 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r00 * (Gamma_lmrp * gp_sqr + Gamma_lprm * gm_sqr);
    tmp += -2 * Gamma_lmrp * gm_sqr * r11;
    tmp += 2 * Gamma_lmrp * gp_gm * r12;
    tmp += 2 * Gamma_lprm * gp_sqr * r33;
    tmp *= 0.5f;
    drho_out_D_dl_r11 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r00 * (Gamma_lmrp * gm_sqr + Gamma_lprm * gp_sqr);
    tmp += -2 * Gamma_lmrp * gp_sqr * r22;
    tmp += 2 * Gamma_lmrp * gp_gm * r12;
    tmp += 2 * Gamma_lprm * gm_sqr * r33;
    tmp *= 0.5f;
    drho_out_D_dl_r22 += tmp;
    


    tmp = 0.0f;
    tmp = -2 * Gamma_lprm * r33;
    tmp += -4 * Gamma_lmrp * gp_gm * r12;
    tmp += 2 * Gamma_lmrp * gm_sqr * r11;
    tmp += 2 * Gamma_lmrp * gp_sqr * r22;
    tmp *= 0.5f;
    drho_out_D_dl_r33 += tmp;
    


    tmp = 0.0f;
    tmp = r01 * (-Gamma_LR0_loc - Gamma_lmrp * gm_sqr);
    tmp += Gamma_lmrp * gp_gm * r02;
    tmp *= 0.5f;
    drho_out_D_dl_r01 += tmp;
    


    tmp = 0.0f;
    tmp = i01 * (-Gamma_LR0_loc - Gamma_lmrp * gm_sqr);
    tmp += Gamma_lmrp * gp_gm * i02;
    tmp *= 0.5f;
    drho_out_D_dl_i01 += tmp;
    


    tmp = 0.0f;
    tmp = r02 * (-Gamma_LR0_loc - Gamma_lmrp * gp_sqr);
    tmp += Gamma_lmrp * gp_gm * r01;
    tmp *= 0.5f;
    drho_out_D_dl_r02 += tmp;
    


    tmp = 0.0f;
    tmp = i02 * (-Gamma_LR0_loc - Gamma_lmrp * gp_sqr);
    tmp += Gamma_lmrp * gp_gm * i01;
    tmp *= 0.5f;
    drho_out_D_dl_i02 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_lmrp * r03;
    tmp += -2 * Gamma_lprm * r03;
    tmp *= 0.5f;
    drho_out_D_dl_r03 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_lmrp * i03;
    tmp += -2 * Gamma_lprm * i03;
    tmp *= 0.5f;
    drho_out_D_dl_i03 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_lmrp * r12;
    tmp += 2 * r00 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp += Gamma_lmrp * gp_gm * r11;
    tmp += Gamma_lmrp * gp_gm * r22;
    tmp += 2 * Gamma_lprm * gp_gm * r33;
    tmp *= 0.5f;
    drho_out_D_dl_r12 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_lmrp * i12;
    tmp *= 0.5f;
    drho_out_D_dl_i12 += tmp;
    


    tmp = 0.0f;
    tmp = r13 * (-Gamma_lmrp * gm_sqr - Gamma_lprm);
    tmp += Gamma_lmrp * gp_gm * r23;
    tmp *= 0.5f;
    drho_out_D_dl_r13 += tmp;
    


    tmp = 0.0f;
    tmp = i13 * (-Gamma_lmrp * gm_sqr - Gamma_lprm);
    tmp += Gamma_lmrp * gp_gm * i23;
    tmp *= 0.5f;
    drho_out_D_dl_i13 += tmp;
    


    tmp = 0.0f;
    tmp = r23 * (-Gamma_lmrp * gp_sqr - Gamma_lprm);
    tmp += Gamma_lmrp * gp_gm * r13;
    tmp *= 0.5f;
    drho_out_D_dl_r23 += tmp;
    


    tmp = 0.0f;
    tmp = i23 * (-Gamma_lmrp * gp_sqr - Gamma_lprm);
    tmp += Gamma_lmrp * gp_gm * i13;
    tmp *= 0.5f;
    drho_out_D_dl_i23 += tmp;
    


}









static __device__ __forceinline__
void dissipator_dot_lead_unrolled_interval_6(

    const float eps_t_substep,

    const float r00, const float r11, const float r22, const float r33,
    const float r01, const float i01, const float r02, const float i02,
    const float r03, const float i03, const float r12, const float i12,
    const float r13, const float i13, const float r23, const float i23,

    float& drho_out_D_dl_r00, float& drho_out_D_dl_r11, float& drho_out_D_dl_r22,
    float& drho_out_D_dl_r33, float& drho_out_D_dl_r01, float& drho_out_D_dl_i01,
    float& drho_out_D_dl_r02, float& drho_out_D_dl_i02, float& drho_out_D_dl_r03,
    float& drho_out_D_dl_i03, float& drho_out_D_dl_r12, float& drho_out_D_dl_i12,
    float& drho_out_D_dl_r13, float& drho_out_D_dl_i13,
    float& drho_out_D_dl_r23, float& drho_out_D_dl_i23,

    const float gp_sqr,
    const float gm_sqr,
    const float gp_gm,
    const float Gamma_LR0_loc,
    const float Gamma_lprm,
    const float Gamma_lmrp

) {


    /*
    float Gamma_10 = 0.0f;
    float Gamma_20 = 0.0f;
    //float Gamma_30 = 0.0f;
    //float Gamma_21 = 0.0f;
    float Gamma_31 = 0.0f;
    float Gamma_32 = 0.0f;

    // Interval 6
    {
        Gamma_10 = compute_W(U, 0, 1);  // Delta N = +1 -> reverse
        Gamma_20 = compute_W(U, 0, 2);  // Delta N = +1 -> reverse
        //Gamma_30 = 0.0f;                // Delta N = -2
        //Gamma_21 = 0.0f;                // Delta N = 0
        Gamma_31 = compute_W(U, 1, 3);  // Delta N = +1 -> reverse
        Gamma_32 = compute_W(U, 2, 3);  // Delta N = +1 -> reverse
    }
    */


    //const float inv_denom = 0.5f / sqrtf(delta_C * delta_C + eps_t_substep * eps_t_substep);

    //const float gp_sqr = 0.5f + eps_t_substep * inv_denom;
    //const float gm_sqr = 0.5f - eps_t_substep * inv_denom;
    //const float gp_gm = delta_C * inv_denom;

    //const float Gamma_LR0_loc = Gamma_LR0;

    //const float Gamma_lprm = Gamma_L0 * gp_sqr + Gamma_R0 * gm_sqr;
    //const float Gamma_lmrp = Gamma_L0 * gm_sqr + Gamma_R0 * gp_sqr;

    register float tmp;



    tmp = 0.0f;
    tmp = -2 * Gamma_LR0_loc * r00;
    tmp *= 0.5f;
    drho_out_D_dl_r00 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r00 * (Gamma_lmrp * gp_sqr + Gamma_lprm * gm_sqr);
    tmp += 2 * r11 * (-Gamma_lmrp * gm_sqr - Gamma_lprm * gp_sqr);
    tmp += 2 * r12 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_r11 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r00 * (Gamma_lmrp * gm_sqr + Gamma_lprm * gp_sqr);
    tmp += 2 * r12 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp += 2 * r22 * (-Gamma_lmrp * gp_sqr - Gamma_lprm * gm_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r22 += tmp;
    


    tmp = 0.0f;
    tmp = 2 * r11 * (Gamma_lmrp * gm_sqr + Gamma_lprm * gp_sqr);
    tmp += 2 * r12 * (-2 * Gamma_lmrp * gp_gm + 2 * Gamma_lprm * gp_gm);
    tmp += 2 * r22 * (Gamma_lmrp * gp_sqr + Gamma_lprm * gm_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r33 += tmp;
    


    tmp = 0.0f;
    tmp = r01 * (-Gamma_LR0_loc - Gamma_lmrp * gm_sqr - Gamma_lprm * gp_sqr);
    tmp += r02 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_r01 += tmp;
    


    tmp = 0.0f;
    tmp = i01 * (-Gamma_LR0_loc - Gamma_lmrp * gm_sqr - Gamma_lprm * gp_sqr);
    tmp += i02 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_i01 += tmp;
    


    tmp = 0.0f;
    tmp = r01 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp += r02 * (-Gamma_LR0_loc - Gamma_lmrp * gp_sqr - Gamma_lprm * gm_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r02 += tmp;
    


    tmp = 0.0f;
    tmp = i01 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp += i02 * (-Gamma_LR0_loc - Gamma_lmrp * gp_sqr - Gamma_lprm * gm_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_i02 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * r03;
    tmp *= 0.5f;
    drho_out_D_dl_r03 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * i03;
    tmp *= 0.5f;
    drho_out_D_dl_i03 += tmp;
    


    tmp = 0.0f;
    tmp = r11 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp += r22 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp += -Gamma_LR0_loc * r12;
    tmp += 2 * r00 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_r12 += tmp;
    


    tmp = 0.0f;
    tmp = -Gamma_LR0_loc * i12;
    tmp *= 0.5f;
    drho_out_D_dl_i12 += tmp;
    


    tmp = 0.0f;
    tmp = r13 * (-Gamma_lmrp * gm_sqr - Gamma_lprm * gp_sqr);
    tmp += r23 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_r13 += tmp;
    


    tmp = 0.0f;
    tmp = i13 * (-Gamma_lmrp * gm_sqr - Gamma_lprm * gp_sqr);
    tmp += i23 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp *= 0.5f;
    drho_out_D_dl_i13 += tmp;
    


    tmp = 0.0f;
    tmp = r13 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp += r23 * (-Gamma_lmrp * gp_sqr - Gamma_lprm * gm_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_r23 += tmp;
    


    tmp = 0.0f;
    tmp = i13 * (Gamma_lmrp * gp_gm - Gamma_lprm * gp_gm);
    tmp += i23 * (-Gamma_lmrp * gp_sqr - Gamma_lprm * gm_sqr);
    tmp *= 0.5f;
    drho_out_D_dl_i23 += tmp;
    


}





__device__ __forceinline__
void dissipator_dot_lead_unrolled_sparse(

    const float eps_t_substep,

    const float rho_in_r00, const float rho_in_r11, const float rho_in_r22,
    const float rho_in_r33, const float rho_in_r01, const float rho_in_i01,
    const float rho_in_r02, const float rho_in_i02, const float rho_in_r03,
    const float rho_in_i03, const float rho_in_r12, const float rho_in_i12,
    const float rho_in_r13, const float rho_in_i13,
    const float rho_in_r23, const float rho_in_i23,

    float& drho_out_D_dl_r00, float& drho_out_D_dl_r11, float& drho_out_D_dl_r22,
    float& drho_out_D_dl_r33, float& drho_out_D_dl_r01, float& drho_out_D_dl_i01,
    float& drho_out_D_dl_r02, float& drho_out_D_dl_i02, float& drho_out_D_dl_r03,
    float& drho_out_D_dl_i03, float& drho_out_D_dl_r12, float& drho_out_D_dl_i12,
    float& drho_out_D_dl_r13, float& drho_out_D_dl_i13,
    float& drho_out_D_dl_r23, float& drho_out_D_dl_i23,

    const float gp_sqr,
    const float gm_sqr,
    const float gp_gm

) {

    const float epsilon_L_loc = epsilon_L;
    const float epsilon_R_loc = epsilon_R;


    const float Gamma_LR0_loc = Gamma_LR0;
    const float Gamma_lprm = Gamma_L0 * gp_sqr + Gamma_R0 * gm_sqr;
    const float Gamma_lmrp = Gamma_L0 * gm_sqr + Gamma_R0 * gp_sqr;



    // Interval 1
    if (eps_t_substep < -epsilon_L_loc) {
        dissipator_dot_lead_unrolled_interval_1(

            eps_t_substep,

            rho_in_r00, rho_in_r11, rho_in_r22, rho_in_r33,
            rho_in_r01, rho_in_i01, rho_in_r02, rho_in_i02, rho_in_r03, rho_in_i03,
            rho_in_r12, rho_in_i12, rho_in_r13, rho_in_i13, rho_in_r23, rho_in_i23,

            drho_out_D_dl_r00, drho_out_D_dl_r11, drho_out_D_dl_r22, drho_out_D_dl_r33,
            drho_out_D_dl_r01, drho_out_D_dl_i01, drho_out_D_dl_r02, drho_out_D_dl_i02,
            drho_out_D_dl_r03, drho_out_D_dl_i03, drho_out_D_dl_r12, drho_out_D_dl_i12,
            drho_out_D_dl_r13, drho_out_D_dl_i13, drho_out_D_dl_r23, drho_out_D_dl_i23,

            gp_sqr,
            gm_sqr,
            gp_gm,
            Gamma_LR0_loc,
            Gamma_lprm,
            Gamma_lmrp

        );

    }

    // Interval 2
    else if (eps_t_substep < -epsilon_R_loc) {
        dissipator_dot_lead_unrolled_interval_2(

            eps_t_substep,

            rho_in_r00, rho_in_r11, rho_in_r22, rho_in_r33,
            rho_in_r01, rho_in_i01, rho_in_r02, rho_in_i02, rho_in_r03, rho_in_i03,
            rho_in_r12, rho_in_i12, rho_in_r13, rho_in_i13, rho_in_r23, rho_in_i23,

            drho_out_D_dl_r00, drho_out_D_dl_r11, drho_out_D_dl_r22, drho_out_D_dl_r33,
            drho_out_D_dl_r01, drho_out_D_dl_i01, drho_out_D_dl_r02, drho_out_D_dl_i02,
            drho_out_D_dl_r03, drho_out_D_dl_i03, drho_out_D_dl_r12, drho_out_D_dl_i12,
            drho_out_D_dl_r13, drho_out_D_dl_i13, drho_out_D_dl_r23, drho_out_D_dl_i23,

            gp_sqr,
            gm_sqr,
            gp_gm,
            Gamma_LR0_loc,
            Gamma_lprm,
            Gamma_lmrp

        );

    }

    // Interval 3
    else if (eps_t_substep < 0.0f) {
        dissipator_dot_lead_unrolled_interval_3(

            eps_t_substep,

            rho_in_r00, rho_in_r11, rho_in_r22, rho_in_r33,
            rho_in_r01, rho_in_i01, rho_in_r02, rho_in_i02, rho_in_r03, rho_in_i03,
            rho_in_r12, rho_in_i12, rho_in_r13, rho_in_i13, rho_in_r23, rho_in_i23,

            drho_out_D_dl_r00, drho_out_D_dl_r11, drho_out_D_dl_r22, drho_out_D_dl_r33,
            drho_out_D_dl_r01, drho_out_D_dl_i01, drho_out_D_dl_r02, drho_out_D_dl_i02,
            drho_out_D_dl_r03, drho_out_D_dl_i03, drho_out_D_dl_r12, drho_out_D_dl_i12,
            drho_out_D_dl_r13, drho_out_D_dl_i13, drho_out_D_dl_r23, drho_out_D_dl_i23,

            gp_sqr,
            gm_sqr,
            gp_gm,
            Gamma_LR0_loc,
            Gamma_lprm,
            Gamma_lmrp

        );

    }

    // Interval 4
    else if (eps_t_substep < epsilon_R_loc) {
        dissipator_dot_lead_unrolled_interval_4(

            eps_t_substep,

            rho_in_r00, rho_in_r11, rho_in_r22, rho_in_r33,
            rho_in_r01, rho_in_i01, rho_in_r02, rho_in_i02, rho_in_r03, rho_in_i03,
            rho_in_r12, rho_in_i12, rho_in_r13, rho_in_i13, rho_in_r23, rho_in_i23,

            drho_out_D_dl_r00, drho_out_D_dl_r11, drho_out_D_dl_r22, drho_out_D_dl_r33,
            drho_out_D_dl_r01, drho_out_D_dl_i01, drho_out_D_dl_r02, drho_out_D_dl_i02,
            drho_out_D_dl_r03, drho_out_D_dl_i03, drho_out_D_dl_r12, drho_out_D_dl_i12,
            drho_out_D_dl_r13, drho_out_D_dl_i13, drho_out_D_dl_r23, drho_out_D_dl_i23,

            gp_sqr,
            gm_sqr,
            gp_gm,
            Gamma_LR0_loc,
            Gamma_lprm,
            Gamma_lmrp

        );

    }

    // Interval 5
    else if (eps_t_substep < epsilon_L_loc) {
        dissipator_dot_lead_unrolled_interval_5(

            eps_t_substep,

            rho_in_r00, rho_in_r11, rho_in_r22, rho_in_r33,
            rho_in_r01, rho_in_i01, rho_in_r02, rho_in_i02, rho_in_r03, rho_in_i03,
            rho_in_r12, rho_in_i12, rho_in_r13, rho_in_i13, rho_in_r23, rho_in_i23,

            drho_out_D_dl_r00, drho_out_D_dl_r11, drho_out_D_dl_r22, drho_out_D_dl_r33,
            drho_out_D_dl_r01, drho_out_D_dl_i01, drho_out_D_dl_r02, drho_out_D_dl_i02,
            drho_out_D_dl_r03, drho_out_D_dl_i03, drho_out_D_dl_r12, drho_out_D_dl_i12,
            drho_out_D_dl_r13, drho_out_D_dl_i13, drho_out_D_dl_r23, drho_out_D_dl_i23,

            gp_sqr,
            gm_sqr,
            gp_gm,
            Gamma_LR0_loc,
            Gamma_lprm,
            Gamma_lmrp

        );

    }

    // Interval 6
    else if (epsilon_L_loc <= eps_t_substep) {
        dissipator_dot_lead_unrolled_interval_6(

            eps_t_substep,

            rho_in_r00, rho_in_r11, rho_in_r22, rho_in_r33,
            rho_in_r01, rho_in_i01, rho_in_r02, rho_in_i02, rho_in_r03, rho_in_i03,
            rho_in_r12, rho_in_i12, rho_in_r13, rho_in_i13, rho_in_r23, rho_in_i23,

            drho_out_D_dl_r00, drho_out_D_dl_r11, drho_out_D_dl_r22, drho_out_D_dl_r33,
            drho_out_D_dl_r01, drho_out_D_dl_i01, drho_out_D_dl_r02, drho_out_D_dl_i02,
            drho_out_D_dl_r03, drho_out_D_dl_i03, drho_out_D_dl_r12, drho_out_D_dl_i12,
            drho_out_D_dl_r13, drho_out_D_dl_i13, drho_out_D_dl_r23, drho_out_D_dl_i23,

            gp_sqr,
            gm_sqr,
            gp_gm,
            Gamma_LR0_loc,
            Gamma_lprm,
            Gamma_lmrp

        );

    }




}


