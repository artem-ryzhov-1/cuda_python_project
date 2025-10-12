// src/rk4/rk4_substep.cuh

#pragma once
#include <cuda_runtime.h>
// #include "lindblad_helpers.cuh"
#include "commutator/commutator.cuh"
//#include "lambda_jacobi_diagonalizer.cuh"
//#include "dissipator_dot_lead.cuh"
#include "dissipators/dissipator_dot_lead_sparse.cuh"
#include "dissipators/dissipator_dot_lead_sparse_log.cuh"

//#include "dissipator_qubit.cuh"
#include "dissipators/dissipator_qubit_relax.cuh"
#include "dissipators/dissipator_qubit_dephase_quasi_static.cuh"

//#include <math.h>
//#include <math_functions.h>
//#include "cuda_intellisense_fixes.cuh"
//#include "constants.cuh"





__device__ __forceinline__
void compute_drho(
    const float rho_in[16],
    float drho_out[16],

    const float eps_t_substep
) {

    //    const unsigned FULL_MASK = 0xFFFFFFFFu;
    //    int lane = threadIdx.x & 31;
    //    bool is_main = ((lane & 1) == 0);       // even = main
    //    int main_lane = lane & ~1;              // index of paired main (even)
    //
    //
    //    if (!is_main) {
    //        // Allocate temporary storage for data from main thread
    //        float rho_in_main[16];
    //        float eps_t_substep_main;
    //
    //        // Fetch rho_in from paired main thread using shuffle
    //#pragma unroll
    //        for (int i = 0; i < 16; ++i) {
    //            rho_in_main[i] = __shfl_sync(FULL_MASK, rho_in[i], main_lane);
    //        }
    //
    //        // Fetch eps_t_substep from main thread
    //        eps_t_substep_main = __shfl_sync(FULL_MASK, eps_t_substep, main_lane);
    //
    //        // Perform the computation
    //        commutator_v2(rho_in_main, drho_out, eps_t_substep_main);
    //
    //        register float eigvals[4];
    //        register float U[16];
    //
    //        diagonalizer_sorted(
    //            eigvals,
    //            U,
    //            eps_t_substep_main
    //        );
    //    }




    for (int i = 0; i < 16; ++i) drho_out[i] = 0.0f;

    commutator_v2(rho_in, drho_out, eps_t_substep);





    /*
    register float eigvals[4];
    register float U[16];

    diagonalizer_sorted(
        eigvals,
        U,
        eps_t_substep
    );


    dissipator_dot_res(
        eps_t_substep,
        rho_in,
        drho_out,
        U
    );
    */




}



__device__ __forceinline__
void compute_drho_log(
    const float rho_in[16],
    float drho_out[16],

    const float eps_t_substep,

    // log
    LogEntry* __restrict__ d_log_buffer,
    const int t_idx_substep
) {

    //    const unsigned FULL_MASK = 0xFFFFFFFFu;
    //    int lane = threadIdx.x & 31;
    //    bool is_main = ((lane & 1) == 0);       // even = main
    //    int main_lane = lane & ~1;              // index of paired main (even)
    //
    //
    //    if (!is_main) {
    //        // Allocate temporary storage for data from main thread
    //        float rho_in_main[16];
    //        float eps_t_substep_main;
    //
    //        // Fetch rho_in from paired main thread using shuffle
    //#pragma unroll
    //        for (int i = 0; i < 16; ++i) {
    //            rho_in_main[i] = __shfl_sync(FULL_MASK, rho_in[i], main_lane);
    //        }
    //
    //        // Fetch eps_t_substep from main thread
    //        eps_t_substep_main = __shfl_sync(FULL_MASK, eps_t_substep, main_lane);
    //
    //        // Perform the computation
    //        commutator_v2(rho_in_main, drho_out, eps_t_substep_main);
    //
    //        register float eigvals[4];
    //        register float U[16];
    //
    //        diagonalizer_sorted(
    //            eigvals,
    //            U,
    //            eps_t_substep_main
    //        );
    //    }




    for (int i = 0; i < 16; ++i) drho_out[i] = 0.0f;

    commutator_v2(rho_in, drho_out, eps_t_substep);





    /*
    register float eigvals[4];
    register float U[16];

    diagonalizer_sorted(
        eigvals,
        U,
        eps_t_substep
    );


    dissipator_dot_res_log(
        eps_t_substep,
        rho_in,
        drho_out,
        U,

        d_log_buffer, t_idx_substep
    );
    */




}



__device__ __forceinline__
void compute_drho_unrolled(
    const float rho_in_0, const float rho_in_1, const float rho_in_2,
    const float rho_in_3, const float rho_in_4, const float rho_in_5,
    const float rho_in_6, const float rho_in_7, const float rho_in_8,
    const float rho_in_9, const float rho_in_10, const float rho_in_11,
    const float rho_in_12, const float rho_in_13,
    const float rho_in_14, const float rho_in_15,

    float& drho_out_0, float& drho_out_1, float& drho_out_2,
    float& drho_out_3, float& drho_out_4, float& drho_out_5,
    float& drho_out_6, float& drho_out_7, float& drho_out_8,
    float& drho_out_9, float& drho_out_10, float& drho_out_11,
    float& drho_out_12, float& drho_out_13,
    float& drho_out_14, float& drho_out_15,

    const float eps_t_substep
) {



    drho_out_0 = 0.0f;
    drho_out_1 = 0.0f;
    drho_out_2 = 0.0f;
    drho_out_3 = 0.0f;
    drho_out_4 = 0.0f;
    drho_out_5 = 0.0f;
    drho_out_6 = 0.0f;
    drho_out_7 = 0.0f;
    drho_out_8 = 0.0f;
    drho_out_9 = 0.0f;
    drho_out_10 = 0.0f;
    drho_out_11 = 0.0f;
    drho_out_12 = 0.0f;
    drho_out_13 = 0.0f;
    drho_out_14 = 0.0f;
    drho_out_15 = 0.0f;



    





    
    commutator_v3_unrolled(
        rho_in_0, rho_in_1, rho_in_2, rho_in_3,
        rho_in_4, rho_in_5, rho_in_6, rho_in_7,
        rho_in_8, rho_in_9, rho_in_10, rho_in_11,
        rho_in_12, rho_in_13, rho_in_14, rho_in_15,

        drho_out_0, drho_out_1, drho_out_2, drho_out_3,
        drho_out_4, drho_out_5, drho_out_6, drho_out_7,
        drho_out_8, drho_out_9, drho_out_10, drho_out_11,
        drho_out_12, drho_out_13, drho_out_14, drho_out_15,

        eps_t_substep
    );

    //const float radical = sqrtf(delta_C * delta_C + eps_t_substep * eps_t_substep);
    const float half_inv_radical = 0.5 * rsqrtf(delta_C * delta_C + eps_t_substep * eps_t_substep);
    const float gp_sqr = 0.5f + eps_t_substep * half_inv_radical;
    const float gm_sqr = 0.5f - eps_t_substep * half_inv_radical;
    const float gp_gm = delta_C * half_inv_radical;

    dissipator_dot_lead_unrolled_sparse(
        eps_t_substep,

        rho_in_0, rho_in_1, rho_in_2, rho_in_3,
        rho_in_4, rho_in_5, rho_in_6, rho_in_7,
        rho_in_8, rho_in_9, rho_in_10, rho_in_11,
        rho_in_12, rho_in_13, rho_in_14, rho_in_15,

        drho_out_0, drho_out_1, drho_out_2, drho_out_3,
        drho_out_4, drho_out_5, drho_out_6, drho_out_7,
        drho_out_8, drho_out_9, drho_out_10, drho_out_11,
        drho_out_12, drho_out_13, drho_out_14, drho_out_15,

        gp_sqr,
        gm_sqr,
        gp_gm
    );


    /*
    dissipator_qubit_unrolled(
        eps_t_substep,

        rho_in_1, rho_in_2, rho_in_4, rho_in_5,
        rho_in_6, rho_in_7, rho_in_10, rho_in_11,
        rho_in_12, rho_in_13, rho_in_14, rho_in_15,

        drho_out_1, drho_out_2, drho_out_4, drho_out_5,
        drho_out_6, drho_out_7, drho_out_10, drho_out_11,
        drho_out_12, drho_out_13, drho_out_14, drho_out_15
    );
    */
    
    dissipator_qubit_relax_unrolled(
        eps_t_substep,

        rho_in_1, rho_in_2, rho_in_4, rho_in_5,
        rho_in_6, rho_in_7, rho_in_10, rho_in_11,
        rho_in_12, rho_in_13, rho_in_14, rho_in_15,

        drho_out_1, drho_out_2, drho_out_4, drho_out_5,
        drho_out_6, drho_out_7, drho_out_10, drho_out_11,
        drho_out_12, drho_out_13, drho_out_14, drho_out_15,

        half_inv_radical,
        gp_sqr,
        gm_sqr,
        gp_gm
    );
    
    dissipator_qubit_dephase_quasi_static_unrolled(
        eps_t_substep,

        rho_in_1, rho_in_2, rho_in_4, rho_in_5,
        rho_in_6, rho_in_7, rho_in_10, rho_in_11,
        rho_in_12, rho_in_13, rho_in_14, rho_in_15,

        drho_out_1, drho_out_2, drho_out_4, drho_out_5,
        drho_out_6, drho_out_7, drho_out_10, drho_out_11,
        drho_out_12, drho_out_13, drho_out_14, drho_out_15,

        half_inv_radical,
        gp_sqr,
        gm_sqr,
        gp_gm
    );
    


    
    ////instead of commutator
    //float result = simulate_flops(3000);
    //if (result > 5.0f) {
    //    result = result + 2.0f;
    //}

    //float x = simulate_flops(200);



    /*
    float eigvals[4];
    float U[16];

    
    diagonalizer_sorted(
        eigvals,
        U,
        eps_t_substep
    );

    

    //float U[16] = {
    //    0.1f, 0.2f, 0.3f, 0.4f,
    //    0.5f, 0.6f, 0.7f, 0.8f,
    //    0.9f, 1.0f, 1.1f, 1.2f,
    //    1.3f, 1.4f, 1.5f, 1.6f
    //};


    
    //dissipator_dot_res_unrolled
    //dissipator_dot_lead_adb_unrolled
    dissipator_dot_lead_adb_unrolled(
        eps_t_substep,

        rho_in_0, rho_in_1, rho_in_2, rho_in_3,
        rho_in_4, rho_in_5, rho_in_6, rho_in_7,
        rho_in_8, rho_in_9, rho_in_10, rho_in_11,
        rho_in_12, rho_in_13, rho_in_14, rho_in_15,

        drho_out_0, drho_out_1, drho_out_2, drho_out_3,
        drho_out_4, drho_out_5, drho_out_6, drho_out_7,
        drho_out_8, drho_out_9, drho_out_10, drho_out_11,
        drho_out_12, drho_out_13, drho_out_14, drho_out_15,

        U
    );
    */
    
    


}




__device__ __forceinline__
void compute_drho_unrolled_log(
    const float rho_in_0, const float rho_in_1, const float rho_in_2,
    const float rho_in_3, const float rho_in_4, const float rho_in_5,
    const float rho_in_6, const float rho_in_7, const float rho_in_8,
    const float rho_in_9, const float rho_in_10, const float rho_in_11,
    const float rho_in_12, const float rho_in_13,
    const float rho_in_14, const float rho_in_15,

    float& drho_out_0, float& drho_out_1, float& drho_out_2,
    float& drho_out_3, float& drho_out_4, float& drho_out_5,
    float& drho_out_6, float& drho_out_7, float& drho_out_8,
    float& drho_out_9, float& drho_out_10, float& drho_out_11,
    float& drho_out_12, float& drho_out_13,
    float& drho_out_14, float& drho_out_15,

    const float eps_t_substep,

    // log
    LogEntry* __restrict__ d_log_buffer,
    const int t_idx_substep,

    const int t_idx_step,
    const int substep_num,
    const float t_step,
    const float t_substep
) {
    d_log_buffer[t_idx_substep].eps_t_substep = eps_t_substep;

    d_log_buffer[t_idx_substep].t_idx_step    = t_idx_step;
    d_log_buffer[t_idx_substep].t_idx_substep = t_idx_substep;
    d_log_buffer[t_idx_substep].t_step        = t_step;
    d_log_buffer[t_idx_substep].t_substep     = t_substep;
    d_log_buffer[t_idx_substep].substep_num   = substep_num;


    d_log_buffer[t_idx_substep].rho_in_0 = rho_in_0;
    d_log_buffer[t_idx_substep].rho_in_1 = rho_in_1;
    d_log_buffer[t_idx_substep].rho_in_2 = rho_in_2;
    d_log_buffer[t_idx_substep].rho_in_3 = rho_in_3;
    d_log_buffer[t_idx_substep].rho_in_4 = rho_in_4;
    d_log_buffer[t_idx_substep].rho_in_5 = rho_in_5;
    d_log_buffer[t_idx_substep].rho_in_6 = rho_in_6;
    d_log_buffer[t_idx_substep].rho_in_7 = rho_in_7;
    d_log_buffer[t_idx_substep].rho_in_8 = rho_in_8;
    d_log_buffer[t_idx_substep].rho_in_9 = rho_in_9;
    d_log_buffer[t_idx_substep].rho_in_10 = rho_in_10;
    d_log_buffer[t_idx_substep].rho_in_11 = rho_in_11;
    d_log_buffer[t_idx_substep].rho_in_12 = rho_in_12;
    d_log_buffer[t_idx_substep].rho_in_13 = rho_in_13;
    d_log_buffer[t_idx_substep].rho_in_14 = rho_in_14;
    d_log_buffer[t_idx_substep].rho_in_15 = rho_in_15;


    drho_out_0 = 0.0f;
    drho_out_1 = 0.0f;
    drho_out_2 = 0.0f;
    drho_out_3 = 0.0f;
    drho_out_4 = 0.0f;
    drho_out_5 = 0.0f;
    drho_out_6 = 0.0f;
    drho_out_7 = 0.0f;
    drho_out_8 = 0.0f;
    drho_out_9 = 0.0f;
    drho_out_10 = 0.0f;
    drho_out_11 = 0.0f;
    drho_out_12 = 0.0f;
    drho_out_13 = 0.0f;
    drho_out_14 = 0.0f;
    drho_out_15 = 0.0f;

    
    commutator_v3_unrolled_log(
        rho_in_0, rho_in_1, rho_in_2, rho_in_3,
        rho_in_4, rho_in_5, rho_in_6, rho_in_7,
        rho_in_8, rho_in_9, rho_in_10, rho_in_11,
        rho_in_12, rho_in_13, rho_in_14, rho_in_15,

        drho_out_0, drho_out_1, drho_out_2, drho_out_3,
        drho_out_4, drho_out_5, drho_out_6, drho_out_7,
        drho_out_8, drho_out_9, drho_out_10, drho_out_11,
        drho_out_12, drho_out_13, drho_out_14, drho_out_15,

        eps_t_substep,

        d_log_buffer, t_idx_substep
    );

    //const float radical = sqrtf(delta_C * delta_C + eps_t_substep * eps_t_substep);
    const float half_inv_radical = 0.5 * rsqrtf(delta_C * delta_C + eps_t_substep * eps_t_substep);
    const float gp_sqr = 0.5f + eps_t_substep * half_inv_radical;
    const float gm_sqr = 0.5f - eps_t_substep * half_inv_radical;
    const float gp_gm = delta_C * half_inv_radical;

    dissipator_dot_lead_unrolled_sparse_log(
        eps_t_substep,

        rho_in_0, rho_in_1, rho_in_2, rho_in_3,
        rho_in_4, rho_in_5, rho_in_6, rho_in_7,
        rho_in_8, rho_in_9, rho_in_10, rho_in_11,
        rho_in_12, rho_in_13, rho_in_14, rho_in_15,

        drho_out_0, drho_out_1, drho_out_2, drho_out_3,
        drho_out_4, drho_out_5, drho_out_6, drho_out_7,
        drho_out_8, drho_out_9, drho_out_10, drho_out_11,
        drho_out_12, drho_out_13, drho_out_14, drho_out_15,

        gp_sqr,
        gm_sqr,
        gp_gm,

        d_log_buffer, t_idx_substep
    );

    /*
    dissipator_qubit_unrolled_log(
        eps_t_substep,

        rho_in_1, rho_in_2, rho_in_4, rho_in_5,
        rho_in_6, rho_in_7, rho_in_10, rho_in_11,
        rho_in_12, rho_in_13, rho_in_14, rho_in_15,

        drho_out_1, drho_out_2, drho_out_4, drho_out_5,
        drho_out_6, drho_out_7, drho_out_10, drho_out_11,
        drho_out_12, drho_out_13, drho_out_14, drho_out_15,

        d_log_buffer, t_idx_substep
    );
    */
    
    dissipator_qubit_relax_unrolled_log(
        eps_t_substep,

        rho_in_1, rho_in_2, rho_in_4, rho_in_5,
        rho_in_6, rho_in_7, rho_in_10, rho_in_11,
        rho_in_12, rho_in_13, rho_in_14, rho_in_15,

        drho_out_1, drho_out_2, drho_out_4, drho_out_5,
        drho_out_6, drho_out_7, drho_out_10, drho_out_11,
        drho_out_12, drho_out_13, drho_out_14, drho_out_15,

        half_inv_radical,
        gp_sqr,
        gm_sqr,
        gp_gm,

        d_log_buffer, t_idx_substep
    );

    dissipator_qubit_dephase_quasi_static_unrolled_log(
        eps_t_substep,

        rho_in_1, rho_in_2, rho_in_4, rho_in_5,
        rho_in_6, rho_in_7, rho_in_10, rho_in_11,
        rho_in_12, rho_in_13, rho_in_14, rho_in_15,

        drho_out_1, drho_out_2, drho_out_4, drho_out_5,
        drho_out_6, drho_out_7, drho_out_10, drho_out_11,
        drho_out_12, drho_out_13, drho_out_14, drho_out_15,

        half_inv_radical,
        gp_sqr,
        gm_sqr,
        gp_gm,

        d_log_buffer, t_idx_substep
    );
    
    














    
    /*
    register float eigvals[4];
    register float U[16];

    diagonalizer_sorted_log(
        eigvals,
        U,
        eps_t_substep,

        d_log_buffer, log_idx
    );


    dissipator_dot_res_unrolled_log(
        eps_t_substep,

        rho_in_0, rho_in_1, rho_in_2, rho_in_3,
        rho_in_4, rho_in_5, rho_in_6, rho_in_7,
        rho_in_8, rho_in_9, rho_in_10, rho_in_11,
        rho_in_12, rho_in_13, rho_in_14, rho_in_15,

        drho_out_0, drho_out_1, drho_out_2, drho_out_3,
        drho_out_4, drho_out_5, drho_out_6, drho_out_7,
        drho_out_8, drho_out_9, drho_out_10, drho_out_11,
        drho_out_12, drho_out_13, drho_out_14, drho_out_15,

        U,
        d_log_buffer, log_idx
    );

    */

    

    d_log_buffer[t_idx_substep].drho_out_total_0 = drho_out_0;
    d_log_buffer[t_idx_substep].drho_out_total_1 = drho_out_1;
    d_log_buffer[t_idx_substep].drho_out_total_2 = drho_out_2;
    d_log_buffer[t_idx_substep].drho_out_total_3 = drho_out_3;
    d_log_buffer[t_idx_substep].drho_out_total_4 = drho_out_4;
    d_log_buffer[t_idx_substep].drho_out_total_5 = drho_out_5;
    d_log_buffer[t_idx_substep].drho_out_total_6 = drho_out_6;
    d_log_buffer[t_idx_substep].drho_out_total_7 = drho_out_7;
    d_log_buffer[t_idx_substep].drho_out_total_8 = drho_out_8;
    d_log_buffer[t_idx_substep].drho_out_total_9 = drho_out_9;
    d_log_buffer[t_idx_substep].drho_out_total_10 = drho_out_10;
    d_log_buffer[t_idx_substep].drho_out_total_11 = drho_out_11;
    d_log_buffer[t_idx_substep].drho_out_total_12 = drho_out_12;
    d_log_buffer[t_idx_substep].drho_out_total_13 = drho_out_13;
    d_log_buffer[t_idx_substep].drho_out_total_14 = drho_out_14;
    d_log_buffer[t_idx_substep].drho_out_total_15 = drho_out_15;

}

