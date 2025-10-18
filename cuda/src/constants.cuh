// src/constants.cuh

#pragma once
#include <cuda_runtime.h>

////////////////////////////

// Math constants


#ifndef M_PIf
constexpr float M_PIf = 3.1415926535897932384626433832f;
#endif



////////////////////////////

// Trajectory-per-warp settings


constexpr int K_TRAJ_PER_WARP = 8;

constexpr int WARP_SIZE = 32;           // should only be 32. macro var intriduces only for convinience
constexpr int LANE_GROUP_SIZE = WARP_SIZE / K_TRAJ_PER_WARP;




////////////////////////////


// device constants
constexpr float a = 0.1f;
constexpr float m = 10.0f;
constexpr float B = (a + 2.0f) * (m + 1.0f) / (a * (m - 1.0f));

constexpr float one_div_m = 1.f / m;

constexpr float muL = 0.0f;
constexpr float muR = 0.0f;
constexpr float kT  = 0.0f;


// max number of noise samples allowed (limited by constant memory size)
constexpr int MAX_NOISE_SAMPLES = 8192;

// device constants

// store Gaussian noise offsets in constant memory
__device__ __constant__ float c_eps_offsets[MAX_NOISE_SAMPLES];

__device__ __constant__ float pi_alpha;
//__device__ __constant__ float B;
//__constant__ float m;
//__device__ __constant__ float one_div_m;
__device__ __constant__ float omega;
__device__ __constant__ float epsilon_L;
__device__ __constant__ float epsilon_R;

//__constant__ float delta_C;
//__constant__ float delta_L;
//__constant__ float delta_R;

__device__ __constant__ float delta_C;
__device__ __constant__ float pi_alpha_delta_C;
__device__ __constant__ float pi_alpha_delta_L;
__device__ __constant__ float pi_alpha_delta_R;

__device__ __constant__ float rho00_init;
__device__ __constant__ float rho11_init;
__device__ __constant__ float rho22_init;
__device__ __constant__ float rho33_init;

__device__ __constant__ int   Npoints;   // just a declaration
__device__ __constant__ int   N_steps_per_period;
__device__ __constant__ int   N_periods;
__device__ __constant__ float dt;

__device__ __constant__ float Gamma_L0;
__device__ __constant__ float Gamma_R0;
__device__ __constant__ float Gamma_LR0;
//__device__ __constant__ float muL;
//__device__ __constant__ float muR;
//__device__ __constant__ float kT;

__device__ __constant__ float Gamma_eg0;
__device__ __constant__ float beta;
__device__ __constant__ float Gamma_eg0_norm;

__device__ __constant__ float Gamma_phi0;



//struct WarpLaneInfo {
//    int lane_id;
//    int group_id;
//    int lane_in_group;
//    int group_base_lane;
//    unsigned int group_mask;
//};





// This macro helps build the HDF5 compound type
#define ADD_FIELD(hdf_type, struct_type, field) \
    hdf_type.insertMember(#field, HOFFSET(struct_type, field), H5::PredType::NATIVE_FLOAT);


struct LogEntry {

    int t_idx_step;
    int t_idx_substep;

    float t_step;
    float t_substep;

    int substep_num;

    float eps_t_substep;

    float gp_sqr;
    float gm_sqr;
    float gp_gm;

    float Gamma_lprm;
    float Gamma_lmrp;

    float Gamma_10;
    float Gamma_20;
    float Gamma_30;
    float Gamma_21;
    float Gamma_31;
    float Gamma_32;

    float Gamma_L0_log;
    float Gamma_R0_log;

    float Gamma_eg;

            float debug_eps_t_substep;
            float debug_delta_C;
            float debug_radical;
            float debig_radical_div_delta_C;
            float debug_Gamma_eg0_norm;
            float debug_beta;
            float debug_Gamma_eg_loc;
    
    float Gamma_phi;

    int interval_diagonalizer;
    int interval_dissipator;


    float W_L_1_0, W_R_1_0;
    //float W_L_0_1, W_R_0_1;

    float W_L_2_0, W_R_2_0;
    //float W_L_0_2, W_R_0_2;

    float W_L_3_0, W_R_3_0;
    //float W_L_0_3, W_R_0_3;

    float W_L_2_1, W_R_2_1;
    //float W_L_1_2, W_R_1_2;

    float W_L_3_1, W_R_3_1;
    //float W_L_1_3, W_R_1_3;

    float W_L_3_2, W_R_3_2;
    //float W_L_2_3, W_R_2_3;

    float U00, U01, U02, U03;
    float U10, U11, U12, U13;
    float U20, U21, U22, U23;
    float U30, U31, U32, U33;

    float E0, E1, E2, E3;

    int rule_col_0, rule_col_1, rule_col_2, rule_col_3;

    float rho_in_0;
    float rho_in_1;
    float rho_in_2;
    float rho_in_3;
    float rho_in_4;
    float rho_in_5;
    float rho_in_6;
    float rho_in_7;
    float rho_in_8;
    float rho_in_9;
    float rho_in_10;
    float rho_in_11;
    float rho_in_12;
    float rho_in_13;
    float rho_in_14;
    float rho_in_15;


    float drho_out_comm_0;
    float drho_out_comm_1;
    float drho_out_comm_2;
    float drho_out_comm_3;
    float drho_out_comm_4;
    float drho_out_comm_5;
    float drho_out_comm_6;
    float drho_out_comm_7;
    float drho_out_comm_8;
    float drho_out_comm_9;
    float drho_out_comm_10;
    float drho_out_comm_11;
    float drho_out_comm_12;
    float drho_out_comm_13;
    float drho_out_comm_14;
    float drho_out_comm_15;

    float drho_out_D_dl_r00;
    float drho_out_D_dl_r11;
    float drho_out_D_dl_r22;
    float drho_out_D_dl_r33;
    float drho_out_D_dl_r01;
    float drho_out_D_dl_i01;
    float drho_out_D_dl_r02;
    float drho_out_D_dl_i02;
    float drho_out_D_dl_r03;
    float drho_out_D_dl_i03;
    float drho_out_D_dl_r12;
    float drho_out_D_dl_i12;
    float drho_out_D_dl_r13;
    float drho_out_D_dl_i13;
    float drho_out_D_dl_r23;
    float drho_out_D_dl_i23;

    //float drho_out_D_eg_r00;
    float drho_out_D_eg_r11;
    float drho_out_D_eg_r22;
    //float drho_out_D_eg_r33;
    float drho_out_D_eg_r01;
    float drho_out_D_eg_i01;
    float drho_out_D_eg_r02;
    float drho_out_D_eg_i02;
    //float drho_out_D_eg_r03;
    //float drho_out_D_eg_i03;
    float drho_out_D_eg_r12;
    float drho_out_D_eg_i12;
    float drho_out_D_eg_r13;
    float drho_out_D_eg_i13;
    float drho_out_D_eg_r23;
    float drho_out_D_eg_i23;

    //float drho_out_D_phi_r00;
    float drho_out_D_phi_r11;
    float drho_out_D_phi_r22;
    //float drho_out_D_phi_r33;
    float drho_out_D_phi_r01;
    float drho_out_D_phi_i01;
    float drho_out_D_phi_r02;
    float drho_out_D_phi_i02;
    //float drho_out_D_phi_r03;
    //float drho_out_D_phi_i03;
    float drho_out_D_phi_r12;
    float drho_out_D_phi_i12;
    float drho_out_D_phi_r13;
    float drho_out_D_phi_i13;
    float drho_out_D_phi_r23;
    float drho_out_D_phi_i23;

    //float drho_out_D_egphi_r00;
    float drho_out_D_egphi_r11;
    float drho_out_D_egphi_r22;
    //float drho_out_D_egphi_r33;
    float drho_out_D_egphi_r01;
    float drho_out_D_egphi_i01;
    float drho_out_D_egphi_r02;
    float drho_out_D_egphi_i02;
    //float drho_out_D_egphi_r03;
    //float drho_out_D_egphi_i03;
    float drho_out_D_egphi_r12;
    float drho_out_D_egphi_i12;
    float drho_out_D_egphi_r13;
    float drho_out_D_egphi_i13;
    float drho_out_D_egphi_r23;
    float drho_out_D_egphi_i23;

    float drho_out_total_0;
    float drho_out_total_1;
    float drho_out_total_2;
    float drho_out_total_3;
    float drho_out_total_4;
    float drho_out_total_5;
    float drho_out_total_6;
    float drho_out_total_7;
    float drho_out_total_8;
    float drho_out_total_9;
    float drho_out_total_10;
    float drho_out_total_11;
    float drho_out_total_12;
    float drho_out_total_13;
    float drho_out_total_14;
    float drho_out_total_15;

    // not needed. was used in dissipator_dot_lead.cuh
    //float drho_out_dissdl_0;
    //float drho_out_dissdl_1;
    //float drho_out_dissdl_2;
    //float drho_out_dissdl_3;
    //float drho_out_dissdl_4;
    //float drho_out_dissdl_5;
    //float drho_out_dissdl_6;
    //float drho_out_dissdl_7;
    //float drho_out_dissdl_8;
    //float drho_out_dissdl_9;
    //float drho_out_dissdl_10;
    //float drho_out_dissdl_11;
    //float drho_out_dissdl_12;
    //float drho_out_dissdl_13;
    //float drho_out_dissdl_14;
    //float drho_out_dissdl_15;

    // dissipator dot-lead in adiabatic basis in dissipator_dot_lead.cuh
    //float D00;
    //float D11;
    //float D22;
    //float D33;
    //float D01re;
    //float D01im;
    //float D02re;
    //float D02im;
    //float D03re;
    //float D03im;
    //float D12re;
    //float D12im;
    //float D13re;
    //float D13im;
    //float D23re;
    //float D23im;

    //bool Uisnan[4][4];
    //bool Uisinf[4][4];

};

// a list of fields for logging
#define LOG_ENTRY_FIELDS \
    X(t_idx_step)       \
    X(substep_num)      \
    X(t_idx_substep)    \
    X(t_step)           \
    X(t_substep)        \
    X(eps_t_substep)    \
    X(interval_dissipator)       \
    X(gp_sqr)           \
    X(gm_sqr)           \
    X(gp_gm)            \
    X(Gamma_lprm)        \
    X(Gamma_lmrp)        \
    X(Gamma_eg)          \
            X(debug_eps_t_substep)        \
            X(debug_delta_C)        \
            X(debug_radical)        \
            X(debig_radical_div_delta_C)        \
            X(debug_Gamma_eg0_norm)        \
            X(debug_beta)        \
            X(debug_Gamma_eg_loc)        \
    X(Gamma_phi)         \
    X(rho_in_0)          \
    X(rho_in_1)          \
    X(rho_in_2)          \
    X(rho_in_3)          \
    X(rho_in_4)          \
    X(rho_in_5)          \
    X(rho_in_6)          \
    X(rho_in_7)          \
    X(rho_in_8)          \
    X(rho_in_9)          \
    X(rho_in_10)         \
    X(rho_in_11)         \
    X(rho_in_12)         \
    X(rho_in_13)         \
    X(rho_in_14)         \
    X(rho_in_15)         \
    X(drho_out_comm_0)          \
    X(drho_out_comm_1)          \
    X(drho_out_comm_2)          \
    X(drho_out_comm_3)          \
    X(drho_out_comm_4)          \
    X(drho_out_comm_5)          \
    X(drho_out_comm_6)          \
    X(drho_out_comm_7)          \
    X(drho_out_comm_8)          \
    X(drho_out_comm_9)          \
    X(drho_out_comm_10)         \
    X(drho_out_comm_11)         \
    X(drho_out_comm_12)         \
    X(drho_out_comm_13)         \
    X(drho_out_comm_14)         \
    X(drho_out_comm_15)         \
    X(drho_out_D_dl_r00)           \
    X(drho_out_D_dl_r11)           \
    X(drho_out_D_dl_r22)           \
    X(drho_out_D_dl_r33)           \
    X(drho_out_D_dl_r01)           \
    X(drho_out_D_dl_i01)           \
    X(drho_out_D_dl_r02)           \
    X(drho_out_D_dl_i02)           \
    X(drho_out_D_dl_r03)           \
    X(drho_out_D_dl_i03)           \
    X(drho_out_D_dl_r12)           \
    X(drho_out_D_dl_i12)           \
    X(drho_out_D_dl_r13)           \
    X(drho_out_D_dl_i13)           \
    X(drho_out_D_dl_r23)           \
    X(drho_out_D_dl_i23)           \
    /*X(drho_out_D_eg_r00)  */         \
    X(drho_out_D_eg_r11)           \
    X(drho_out_D_eg_r22)           \
    /*X(drho_out_D_eg_r33)  */         \
    X(drho_out_D_eg_r01)           \
    X(drho_out_D_eg_i01)           \
    X(drho_out_D_eg_r02)           \
    X(drho_out_D_eg_i02)           \
    /*X(drho_out_D_eg_r03)  */         \
    /*X(drho_out_D_eg_i03)  */         \
    X(drho_out_D_eg_r12)           \
    X(drho_out_D_eg_i12)           \
    X(drho_out_D_eg_r13)           \
    X(drho_out_D_eg_i13)           \
    X(drho_out_D_eg_r23)           \
    X(drho_out_D_eg_i23)           \
    /*X(drho_out_D_phi_r00)*/           \
    X(drho_out_D_phi_r11)           \
    X(drho_out_D_phi_r22)           \
    /*X(drho_out_D_phi_r33) */          \
    X(drho_out_D_phi_r01)           \
    X(drho_out_D_phi_i01)           \
    X(drho_out_D_phi_r02)           \
    X(drho_out_D_phi_i02)           \
    /*X(drho_out_D_phi_r03)*/           \
    /*X(drho_out_D_phi_i03)*/           \
    X(drho_out_D_phi_r12)           \
    X(drho_out_D_phi_i12)           \
    X(drho_out_D_phi_r13)           \
    X(drho_out_D_phi_i13)           \
    X(drho_out_D_phi_r23)           \
    X(drho_out_D_phi_i23)           \
    /*X(drho_out_D_egphi_r00)*/           \
    X(drho_out_D_egphi_r11)           \
    X(drho_out_D_egphi_r22)           \
    /*X(drho_out_D_egphi_r33) */          \
    X(drho_out_D_egphi_r01)           \
    X(drho_out_D_egphi_i01)           \
    X(drho_out_D_egphi_r02)           \
    X(drho_out_D_egphi_i02)           \
    /*X(drho_out_D_egphi_r03)*/           \
    /*X(drho_out_D_egphi_i03)*/           \
    X(drho_out_D_egphi_r12)           \
    X(drho_out_D_egphi_i12)           \
    X(drho_out_D_egphi_r13)           \
    X(drho_out_D_egphi_i13)           \
    X(drho_out_D_egphi_r23)           \
    X(drho_out_D_egphi_i23)           \
    X(drho_out_total_0)         \
    X(drho_out_total_1)         \
    X(drho_out_total_2)         \
    X(drho_out_total_3)         \
    X(drho_out_total_4)         \
    X(drho_out_total_5)         \
    X(drho_out_total_6)         \
    X(drho_out_total_7)         \
    X(drho_out_total_8)         \
    X(drho_out_total_9)         \
    X(drho_out_total_10)        \
    X(drho_out_total_11)        \
    X(drho_out_total_12)        \
    X(drho_out_total_13)        \
    X(drho_out_total_14)        \
    X(drho_out_total_15)





/*



        X(Gamma_10)          \
        X(Gamma_20)          \
        X(Gamma_30)          \
        X(Gamma_21)          \
        X(Gamma_31)          \
        X(Gamma_32)          \
        X(Gamma_L0_log)       \
        X(Gamma_R0_log)       \
 
        X(interval_diagonalizer)     \

        X(W_L_1_0)           \
        X(W_R_1_0)           \
        X(W_L_2_0)           \
        X(W_R_2_0)           \
        X(W_L_3_0)           \
        X(W_R_3_0)           \
        X(W_L_2_1)           \
        X(W_R_2_1)           \
        X(W_L_3_1)           \
        X(W_R_3_1)           \
        X(W_L_3_2)           \
        X(W_R_3_2)           \

        X(U00)               \
        X(U01)               \
        X(U02)               \
        X(U03)               \
        X(U10)               \
        X(U11)               \
        X(U12)               \
        X(U13)               \
        X(U20)               \
        X(U21)               \
        X(U22)               \
        X(U23)               \
        X(U30)               \
        X(U31)               \
        X(U32)               \
        X(U33)               \

        X(E0)                \
        X(E1)                \
        X(E2)                \
        X(E3)                \

        X(rule_col_0)        \
        X(rule_col_1)        \
        X(rule_col_2)        \
        X(rule_col_3)        \
*/





