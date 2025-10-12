// src/host_branch_single.cuh

#pragma once

#include <cuda_runtime.h> // Required for CUDA calls
#include <iostream>       // Used: std::cout, std::cerr
#include <vector>         // Used: std::vector
#include <cmath>          // Used: M_PIf, std::max, etc.
#include <string>
#include "cuda_intellisense_fixes.cuh"
#include "lindblad_kernel_single.cuh"
#include "constants.cuh"
#include "writer.cuh"
#include "host_helpers.cuh"
#include "hdf5_writer.cuh"





// #include <cuda_profiler_api.h> // for #NCU
//#include <nvtx3/nvToolsExt.h> // for #NCU








// -------------------------
// single mode execution without precomputed L
// -------------------------



__host__ inline void run_single_mode(
    const float eps0_target,
    const float A_target,

    const int host_N_steps_per_period,
    const int host_N_periods,

    const float host_dt,
    const float nu,
    const float alpha,

    const float host_rho00_init, const float host_rho11_init,
    const float host_rho22_init, const float host_rho33_init,

    const float host_delta_C, const float host_delta_L, const float host_delta_R,

    const float g_en, const float g_phi, const float gL_en,
    const float gL_phi, const float gR_en, const float gR_phi,

    const std::string& path_dynamics_single_mode_output_csv,
    const std::string& ouput_option,
    const std::string& unrolled_option,
    const bool single_mode_log_option,
    const std::string& path_dynamics_single_mode_output_log_csv,
    const std::string& path_dynamics_single_mode_output_log_hdf5,
    const std::string& threads_per_traj_opt,

    const float host_Gamma_L0,
    const float host_Gamma_R0,
    const float host_Gamma_eg0,
    const float omega_c_norm,
    const float host_Gamma_phi0
)
{

    std::cout << "Launching run_single_mode branch" << std::endl;


    const float host_GammaLR0 = host_Gamma_L0 + host_Gamma_R0;
    const float host_omega = 2.0f * M_PIf * nu;
    const float host_pi_alpha = M_PIf * alpha;
    const int N_steps_total = host_N_steps_per_period * host_N_periods;

    std::cout << " eps0_target: " << eps0_target << std::endl;
    std::cout << " A_target   : " << A_target << std::endl;

    std::cout << " unrolled_option: " << unrolled_option << std::endl;

    std::cout << " Gamma_L0: " << host_Gamma_L0 << std::endl;
    std::cout << " Gamma_R0: " << host_Gamma_R0 << std::endl;
    //std::cout << " muL: " << host_muL << std::endl;
    //std::cout << " muR: " << host_muR << std::endl;
    //std::cout << " T_K: " << T_K << std::endl;

    std::cout << " Gamma_eg0: "  << host_Gamma_eg0  << std::endl;
    std::cout << " Gamma_phi0: " << host_Gamma_phi0 << std::endl;



    std::vector<float> rho_avg(16);

    // allocate outputs
    float* d_rho_avg = nullptr;
    float* d_rho_dynamics = nullptr;
    float* d_time_dynamics = nullptr;
    float* d_eps_dynamics = nullptr;


    gpuCheck(cudaMalloc(&d_rho_avg, 16 * sizeof(float)), "cudaMalloc d_rho_avg");
    gpuCheck(cudaMalloc(&d_rho_dynamics, N_steps_total * 4 * sizeof(float)), "cudaMalloc d_rho_dynamics");
    gpuCheck(cudaMalloc(&d_time_dynamics, N_steps_total * sizeof(float)), "cudaMalloc d_time_dynamics");
    gpuCheck(cudaMalloc(&d_eps_dynamics, N_steps_total * sizeof(float)), "cudaMalloc d_eps_dynamics");



    // allocate host vectors
    std::vector<float> h_rho_dynamics(N_steps_total * 4);
    std::vector<float> h_time_dynamics(N_steps_total);
    std::vector<float> h_eps_dynamics(N_steps_total);


    // when delta_L, delta_R are non-zero. probably approximate solution
    //const float host_epsilon_R = 1 / (m * (B + 1)) + (m * (B + 1) * host_delta_L * host_delta_L) / (2 * B) - (m * host_delta_C * host_delta_C) / 2;
    //const float host_epsilon_L = 1 / (m * (B - 1)) - (m * (B - 1) * host_delta_R * host_delta_R) / (2 * B) + (m * host_delta_C * host_delta_C) / 2;

    // when delta_L, delta_R are zero. exact solution
    const float radical = std::sqrt(1 + m * m * (B * B - 1) * host_delta_C * host_delta_C);
    const float host_epsilon_L = (B + radical) / (m * (B * B - 1));
    const float host_epsilon_R = (B - radical) / (m * (B * B - 1));

    //const float host_one_div_m = 1.f / host_m;

    const float host_pi_alpha_delta_C = host_pi_alpha * host_delta_C;
    const float host_pi_alpha_delta_L = host_pi_alpha * host_delta_L;
    const float host_pi_alpha_delta_R = host_pi_alpha * host_delta_R;

    const float host_beta = host_delta_C * host_delta_C / (omega_c_norm * omega_c_norm);
    const float host_Gamma_eg0_norm = host_Gamma_eg0 * expf(host_beta);


    std::cout << "debugging: host_delta_C   : " << host_delta_C << std::endl;
    std::cout << "debugging: omega_c_norm   : " << omega_c_norm << std::endl;
    std::cout << "debugging: host_beta   : " << host_beta << std::endl;
    std::cout << "debugging: host_Gamma_eg0   : " << host_Gamma_eg0 << std::endl;
    std::cout << "debugging: host_Gamma_eg0_norm   : " << host_Gamma_eg0_norm << std::endl;






    gpuCheck(cudaMemcpyToSymbol(pi_alpha, &host_pi_alpha, sizeof(float)), "cudaMemcpyToSymbol pi_alpha");
    //gpuCheck(cudaMemcpyToSymbol(B, &host_B, sizeof(float)), "cudaMemcpyToSymbol B");
    //cudaMemcpyToSymbol(m, &host_m, sizeof(float));
    //gpuCheck(cudaMemcpyToSymbol(one_div_m, &host_one_div_m, sizeof(float)), "cudaMemcpyToSymbol one_div_m");
    gpuCheck(cudaMemcpyToSymbol(omega, &host_omega, sizeof(float)), "cudaMemcpyToSymbol omega");
    gpuCheck(cudaMemcpyToSymbol(epsilon_R, &host_epsilon_R, sizeof(float)), "cudaMemcpyToSymbol epsilon_R");
    gpuCheck(cudaMemcpyToSymbol(epsilon_L, &host_epsilon_L, sizeof(float)), "cudaMemcpyToSymbol epsilon_L");

    gpuCheck(cudaMemcpyToSymbol(delta_C, &host_delta_C, sizeof(float)), "cudaMemcpyToSymbol delta_C");
    gpuCheck(cudaMemcpyToSymbol(pi_alpha_delta_C, &host_pi_alpha_delta_C, sizeof(float)), "cudaMemcpyToSymbol pi_alpha_delta_C");
    gpuCheck(cudaMemcpyToSymbol(pi_alpha_delta_L, &host_pi_alpha_delta_L, sizeof(float)), "cudaMemcpyToSymbol pi_alpha_delta_L");
    gpuCheck(cudaMemcpyToSymbol(pi_alpha_delta_R, &host_pi_alpha_delta_R, sizeof(float)), "cudaMemcpyToSymbol pi_alpha_delta_R");

    //cudaMemcpyToSymbol(delta_C, &host_delta_C, sizeof(float));
    //cudaMemcpyToSymbol(delta_L, &host_delta_L, sizeof(float));
    //cudaMemcpyToSymbol(delta_R, &host_delta_R, sizeof(float));

    gpuCheck(cudaMemcpyToSymbol(rho00_init, &host_rho00_init, sizeof(float)), "cudaMemcpyToSymbol rho00_init");
    gpuCheck(cudaMemcpyToSymbol(rho11_init, &host_rho11_init, sizeof(float)), "cudaMemcpyToSymbol rho11_init");
    gpuCheck(cudaMemcpyToSymbol(rho22_init, &host_rho22_init, sizeof(float)), "cudaMemcpyToSymbol rho22_init");
    gpuCheck(cudaMemcpyToSymbol(rho33_init, &host_rho33_init, sizeof(float)), "cudaMemcpyToSymbol rho33_init");

    //gpuCheck(cudaMemcpyToSymbol(Npoints, &host_N_points, sizeof(int)), "cudaMemcpyToSymbol Npoints");
    gpuCheck(cudaMemcpyToSymbol(N_steps_per_period, &host_N_steps_per_period, sizeof(int)), "cudaMemcpyToSymbol N_steps_per_period");
    gpuCheck(cudaMemcpyToSymbol(N_periods, &host_N_periods, sizeof(int)), "cudaMemcpyToSymbol N_periods");
    gpuCheck(cudaMemcpyToSymbol(dt, &host_dt, sizeof(float)), "cudaMemcpyToSymbol dt");

    gpuCheck(cudaMemcpyToSymbol(Gamma_LR0, &host_GammaLR0, sizeof(float)), "cudaMemcpyToSymbol Gamma_LR0");
    gpuCheck(cudaMemcpyToSymbol(Gamma_L0,  &host_Gamma_L0, sizeof(float)), "cudaMemcpyToSymbol Gamma_L0");
    gpuCheck(cudaMemcpyToSymbol(Gamma_R0,  &host_Gamma_R0, sizeof(float)), "cudaMemcpyToSymbol Gamma_R0");
    //gpuCheck(cudaMemcpyToSymbol(muL, &host_muL, sizeof(float)), "cudaMemcpyToSymbol muL");
    //gpuCheck(cudaMemcpyToSymbol(muR, &host_muR, sizeof(float)), "cudaMemcpyToSymbol muR");
    //gpuCheck(cudaMemcpyToSymbol(kT, &host_kT, sizeof(float)), "cudaMemcpyToSymbol kT");
    gpuCheck(cudaMemcpyToSymbol(Gamma_eg0,  &host_Gamma_eg0,  sizeof(float)), "cudaMemcpyToSymbol Gamma_eg0");
    gpuCheck(cudaMemcpyToSymbol(Gamma_phi0, &host_Gamma_phi0, sizeof(float)), "cudaMemcpyToSymbol Gamma_phi0");

    gpuCheck(cudaMemcpyToSymbol(beta,           &host_beta,           sizeof(float)), "cudaMemcpyToSymbol beta");
    gpuCheck(cudaMemcpyToSymbol(Gamma_eg0_norm, &host_Gamma_eg0_norm, sizeof(float)), "cudaMemcpyToSymbol Gamma_eg0_norm");



    // for logging

    int log_size = 0;
    LogEntry* d_log_buffer = nullptr;

    if (single_mode_log_option == true){

        log_size = host_N_steps_per_period * host_N_periods * 4;
        const int log_size_bytes = log_size * sizeof(LogEntry);

        if (log_size_bytes > 1825361100) {
            std::cerr << "WARNING: total time steps is " << host_N_steps_per_period * host_N_periods
                << " and log buffer size " << log_size_bytes/ 1048576
                << " MB exceeds 1.7 GB. Program might lack GPU RAM and crush." << std::endl;
        }

        // Allocate device memory for log buffer
        gpuCheck(cudaMalloc(&d_log_buffer, log_size_bytes), "cudaMalloc d_log_buffer");

    }


    // ==================================================
    // launch kernel

    // starting #NCU marker
    // nvtxRangePushA("lindblad_rk4_kernel_shared");

    // threads per block
    int threads_per_block;
    if (threads_per_traj_opt == "one_thread_per_traj") {
        threads_per_block = 1;    // launch only 1 thread
    }
    else if (threads_per_traj_opt == "thread_group_in_warp_per_traj_shuffle" ||
             threads_per_traj_opt == "thread_group_in_warp_per_traj_shmem") {
        threads_per_block = LANE_GROUP_SIZE;   // launch threads only for calculating 1 trajectory
    }



    if (unrolled_option == "as_arrays"
        && single_mode_log_option == false
        && threads_per_traj_opt == "one_thread_per_traj") {

        /* good. only for faster compilation
        printf("Launching kernel singlemode_as_arrays without log: blocks=1 threads_per_block=%d\n", threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        lindblad_rk4_kernel_singlemode <<< 1, threads_per_block >>> (
            eps0_target, A_target,

            d_rho_avg,
            d_rho_dynamics,
            d_time_dynamics,
            d_eps_dynamics
        );
        */
    }
    else if (unrolled_option == "as_arrays"
        && single_mode_log_option == true
        && threads_per_traj_opt == "one_thread_per_traj") {

        /* good. only for faster compilation
        printf("Launching kernel singlemode_as_arrays with log: blocks=1 threads_per_block=%d\n", threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        lindblad_rk4_kernel_singlemode_log <<< 1, threads_per_block >>> (
            eps0_target, A_target,

            d_rho_avg,
            d_rho_dynamics,
            d_time_dynamics,
            d_eps_dynamics,

            d_log_buffer
        );
        */
    }
    else if (unrolled_option == "unrolled"
        && single_mode_log_option == false
        && threads_per_traj_opt == "one_thread_per_traj")
    {

        printf("Launching kernel singlemode_unrolled without log: blocks=1 threads_per_block=%d\n", threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        lindblad_rk4_kernel_singlemode_unrolled <<< 1, threads_per_block >>> (
            eps0_target, A_target,

            d_rho_avg,
            d_rho_dynamics,
            d_time_dynamics,
            d_eps_dynamics
        );

    }
    else if (unrolled_option == "unrolled"
        && single_mode_log_option == true
        && threads_per_traj_opt == "one_thread_per_traj")
    {

        printf("Launching kernel singlemode_unrolled with log: blocks=1 threads_per_block=%d\n", threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        lindblad_rk4_kernel_singlemode_unrolled_log <<< 1, threads_per_block >>> (
            eps0_target, A_target,

            d_rho_avg,
            d_rho_dynamics,
            d_time_dynamics,
            d_eps_dynamics,

            d_log_buffer
            );

    }
    else if (unrolled_option == "unrolled"
        && single_mode_log_option == false
        && threads_per_traj_opt == "thread_group_in_warp_per_traj_shuffle")
    {

        printf("Launching kernel singlemode_unrolled_warp without log: blocks=1 threads_per_block=%d\n", threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        lindblad_rk4_kernel_singlemode_unrolled_warp <<< 1, threads_per_block >>> (
            eps0_target, A_target,

            d_rho_avg,
            d_rho_dynamics,
            d_time_dynamics,
            d_eps_dynamics
            );

    }
    else if (unrolled_option == "unrolled"
        && single_mode_log_option == true
        && threads_per_traj_opt == "thread_group_in_warp_per_traj_shuffle")
    {
        /* not yet implemented. for faster compilation
        printf("Launching kernel singlemode_unrolled_warp with log: blocks=1 threads_per_block=%d\n", threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        lindblad_rk4_kernel_singlemode_unrolled_warp_log <<< 1, threads_per_block >>> (
            eps0_target, A_target,

            d_rho_avg,
            d_rho_dynamics,
            d_time_dynamics,
            d_eps_dynamics,

            d_log_buffer
        );
        */
    }


    // check for launch errors immediately
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        std::abort();   // stops program immediately
        exit(1);
    }

    gpuCheck(cudaGetLastError(), "Kernel launch succeded");
    gpuCheck(cudaDeviceSynchronize(), "Sync after single kernel");

    // ending #NCU marker
    // nvtxRangePop();

    // copy data back
    gpuCheck(cudaMemcpy(rho_avg.data(),         d_rho_avg,       16 * sizeof(float),                cudaMemcpyDeviceToHost), "cudaMemcpy single rho_avg");
    gpuCheck(cudaMemcpy(h_rho_dynamics.data(),  d_rho_dynamics,  N_steps_total * 4 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy rho_dynamics");
    gpuCheck(cudaMemcpy(h_time_dynamics.data(), d_time_dynamics, N_steps_total * sizeof(float),     cudaMemcpyDeviceToHost), "cudaMemcpy time_dynamics");
    gpuCheck(cudaMemcpy(h_eps_dynamics.data(),  d_eps_dynamics,  N_steps_total * sizeof(float),     cudaMemcpyDeviceToHost), "cudaMemcpy eps_dynamics");



    if (ouput_option != "ssd_csv") {
        std::cerr << "ERROR: ouput_option = 'ram' is not implemented for singlemode" << std::endl;
    }

    // -----------------------
    // write CSV with averages
    // -----------------------
    //MKDIR("output"); // ensure output dir exists
    //std::ofstream ofs("output/rho_single_avg_out.csv");
    std::ofstream ofs(path_dynamics_single_mode_output_csv);
    ofs << std::setprecision(8) << std::fixed;

    // --- First line: averaged intervals ---
    ofs << "eps0,A,";
    ofs << "avg00_whole,avg01_whole,avg10_whole,avg11_whole,";
    ofs << "avg00_last,avg01_last,avg10_last,avg11_last,";
    ofs << "avg00_2last,avg01_2last,avg10_2last,avg11_2last,";
    ofs << "avg00_3last,avg01_3last,avg10_3last,avg11_3last\n";

    // --- Second line: average values ---
    ofs << eps0_target << "," << A_target << ",";
    for (int k = 0; k < 16; k++) {
        ofs << rho_avg[k];
        if (k + 1 < 16) ofs << ",";
    }
    ofs << std::endl;

    // --- Third line: column headers for dynamics data ---
    ofs << "time,epst,p00,p01,p10,p11\n";

    for (int t = 0; t < N_steps_total; ++t) {
        float time = h_time_dynamics[t];
        float eps = h_eps_dynamics[t];

        // populations (diagonals)
        const int base = t * 4;
        float p00 = h_rho_dynamics[base + 0];
        float p01 = h_rho_dynamics[base + 1];
        float p10 = h_rho_dynamics[base + 2];
        float p11 = h_rho_dynamics[base + 3];

        ofs << time << "," << eps << ","
            << p00 << "," << p01 << ","
            << p10 << "," << p11 << std::endl;
    }
    ofs << std::endl;

    ofs.flush();    // flush buffers to OS
    //int fd = ::open("output/rho_single_avg_out.csv", O_RDWR);
    int fd = ::open(path_dynamics_single_mode_output_csv.c_str(), O_RDWR);

#ifdef _WIN32
    if (fd != -1) { _commit(fd); _close(fd); }
#else
    if (fd != -1) { fsync(fd); close(fd); }
#endif

    ofs.close();
    std::cout << "Averaged results and dynamics results saved to output/rho_single_dynamics_out.csv" << std::endl;


    if (single_mode_log_option == true) {


        // Copy back log data from d_log_buffer to host
        std::vector<LogEntry> h_log_buffer(log_size);
        gpuCheck(cudaMemcpy(h_log_buffer.data(), d_log_buffer, log_size * sizeof(LogEntry), cudaMemcpyDeviceToHost), "cudaMemcpy log_buffer");
        cudaFree(d_log_buffer);

        // -----------------------
        // write log in HDF5 file
        // -----------------------

        write_log_entries_to_hdf5(
            h_log_buffer,
            path_dynamics_single_mode_output_log_hdf5,
            
            host_pi_alpha,
            host_pi_alpha_delta_C,
            host_delta_C,
            host_Gamma_L0,
            host_Gamma_R0,
            host_Gamma_eg0,
            host_Gamma_eg0_norm,
            host_beta,
            host_Gamma_phi0,
            host_epsilon_L,
            host_epsilon_R,

            6
        );

        // -----------------------
        // write log in CSV file
        // -----------------------

        /*
        //MKDIR("output"); // ensure output dir exists
        std::ofstream ofs_log(path_dynamics_single_mode_output_log_csv);
        ofs_log << std::setprecision(8) << std::fixed;


        //// --- First line: column headers for dynamics data ---
        //ofs_log << "time,epst,Gamma_10,Gamma_20,Gamma_30,Gamma_21,Gamma_31,Gamma_32\n";
        //for (int idx = 0; idx < log_size; ++idx) {
        //    LogEntry log_entry = h_log_buffer[idx];
        //    float time = log_entry.time;
        //    float eps_t = log_entry.eps_t;
        //    float Gamma_10 = log_entry.Gamma_10;
        //    float Gamma_20 = log_entry.Gamma_20;
        //    float Gamma_30 = log_entry.Gamma_30;
        //    float Gamma_21 = log_entry.Gamma_21;
        //    float Gamma_31 = log_entry.Gamma_31;
        //    float Gamma_32 = log_entry.Gamma_32;
        //    ofs_log << time << "," << eps_t << ","
        //        << Gamma_10 << "," << Gamma_20 << "," << Gamma_30 << "," 
        //        << Gamma_21 << "," << Gamma_31 << "," << Gamma_32 << std::endl;
        //}
        

        // CSV header
        #define X(name) ofs_log << #name << ",";
        LOG_ENTRY_FIELDS
        #undef X

        //ofs_log << "U00isnan,U00isinf,U01isnan,U01isinf,U02isnan,U02isinf,U03isnan,U03isinf,";
        //ofs_log << "U10isnan,U10isinf,U11isnan,U11isinf,U12isnan,U12isinf,U13isnan,U13isinf,";
        //ofs_log << "U20isnan,U20isinf,U21isnan,U21isinf,U22isnan,U22isinf,U23isnan,U23isinf,";
        //ofs_log << "U30isnan,U30isinf,U31isnan,U31isinf,U32isnan,U32isinf,U33isnan,U33isinf";
        ofs_log << std::endl;
        
        // CSV data
        for (int idx = 0; idx < log_size; ++idx) {
            const LogEntry& log_entry = h_log_buffer[idx];

            #define X(name) ofs_log << log_entry.name << ",";
            LOG_ENTRY_FIELDS
            #undef X

            //for (int row = 0; row < 4; row++) {
            //    for (int col = 0; col < 4; col++) {
            //        if (log_entry.Uisnan[row][col] == false) {
            //            ofs << "0" << ",";
            //        }
            //        else {
            //            ofs << "1" << ",";
            //        }
            //        if (log_entry.Uisinf[row][col] == false) {
            //            ofs << "0" << ",";
            //        }
            //        else {
            //            ofs << "1" << ",";
            //        }
            //    }
            //}

            ofs_log << std::endl;
        }
        

        ofs_log << std::endl;

        ofs_log.flush();    // flush buffers to OS

        int fd_log = ::open(path_dynamics_single_mode_output_log_csv.c_str(), O_RDWR);

#ifdef _WIN32
        if (fd_log != -1) { _commit(fd_log); _close(fd_log); }
#else
        if (fd_log != -1) { fsync(fd_log); close(fd_log); }
#endif

        ofs_log.close();
        std::cout << "Log results saved to csv" << std::endl;
        */

    }



    // cleanup
    cudaFree(d_rho_avg);
    cudaFree(d_rho_dynamics);
    cudaFree(d_time_dynamics);
    cudaFree(d_eps_dynamics);


}





