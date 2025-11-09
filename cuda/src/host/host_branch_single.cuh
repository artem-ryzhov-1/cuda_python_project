// src/host_branch_single.cuh

#pragma once

#include <cuda_runtime.h> // Required for CUDA calls
#include <iostream>       // Used: std::cout, std::cerr
#include <vector>         // Used: std::vector
#include <cmath>          // Used: M_PIf, std::max, etc.
#include <string>
// #include "cuda_intellisense_fixes.cuh"
#include "kernels/lindblad_kernel_single.cuh"
#include "constants.cuh"
// #include "writer.cuh"
#include "host_helpers.cuh"
// #include "hdf5_writer.cuh"
#include <iomanip>
#include <fstream>


#ifdef _WIN32
    #include <io.h>     // for _commit(), _close() on Windows
#else
    #include <fcntl.h>  // for open() on Unix-like systems
    #include <unistd.h> // for fsync(), close() on Unix-like systems
#endif


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
    const int N_samples_noise,
    const bool quasi_static_ensemble_dephasing_flag,
    float* eps_offsets,

    const float host_dt,
    const float nu,
    const float alpha,

    const float host_rho00_init, const float host_rho11_init,
    const float host_rho22_init, const float host_rho33_init,

    const float host_delta_C, const float host_delta_L, const float host_delta_R,

    const float g_en, const float g_phi, const float gL_en,
    const float gL_phi, const float gR_en, const float gR_phi,

    const std::string& path_dynamics_single_mode_output_csv,
    const std::string& path_output_bin_file_singlemode,
    const std::string& output_option,
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
    float* d_rho_avg_singlemode = nullptr;
    float* d_rho_dynamics = nullptr;
    float* d_time_dynamics = nullptr;
    float* d_eps_dynamics = nullptr;


    gpuCheck(cudaMalloc(&d_rho_avg_singlemode, 16 * sizeof(float)), "cudaMalloc d_rho_avg_singlemode");
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
    int blocks;

    if (threads_per_traj_opt == "one_thread_per_traj") {
        
        if (!quasi_static_ensemble_dephasing_flag){
            threads_per_block = 1;    // launch only 1 thread
            blocks = 1;               
        }
        else {
            threads_per_block = std::min(128, N_samples_noise);  // launch min(128, N_samples_noise) threads for ensemble
            blocks = (N_samples_noise + threads_per_block - 1) / threads_per_block;   // standard
        }

    }



    if (unrolled_option == "as_arrays"
        && single_mode_log_option == false
        && threads_per_traj_opt == "one_thread_per_traj") {

        /* good. only for faster compilation
        printf("Launching kernel singlemode as_arrays without log: blocks=1 threads_per_block=%d\n", threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        lindblad_rk4_kernel_singlemode <<< blocks, threads_per_block >>> (
            eps0_target, A_target,

            d_rho_avg_singlemode,
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
        printf("Launching kernel singlemode as_arrays with log: blocks=1 threads_per_block=%d\n", threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        lindblad_rk4_kernel_singlemode_log <<< blocks, threads_per_block >>> (
            eps0_target, A_target,

            d_rho_avg_singlemode,
            d_rho_dynamics,
            d_time_dynamics,
            d_eps_dynamics,

            d_log_buffer
        );
        */
    }
    else if (unrolled_option == "unrolled"
        && single_mode_log_option == false
        && threads_per_traj_opt == "one_thread_per_traj"
        && !quasi_static_ensemble_dephasing_flag)
    {

        printf("Launching kernel singlemode unrolled no_ensemble without log: blocks=1 threads_per_block=%d\n", threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        lindblad_rk4_kernel_singlemode_unrolled_fsal <<< blocks, threads_per_block >>> (
            eps0_target, A_target,

            d_rho_avg_singlemode,
            d_rho_dynamics,
            d_time_dynamics,
            d_eps_dynamics
        );

    }
    else if (unrolled_option == "unrolled"
        && single_mode_log_option == true
        && threads_per_traj_opt == "one_thread_per_traj"
        && !quasi_static_ensemble_dephasing_flag)
    {

        printf("Launching kernel singlemode unrolled no_ensemble with log: blocks=1 threads_per_block=%d\n", threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        lindblad_rk4_kernel_singlemode_unrolled_log <<< blocks, threads_per_block >>> (
            eps0_target, A_target,

            d_rho_avg_singlemode,
            d_rho_dynamics,
            d_time_dynamics,
            d_eps_dynamics,

            d_log_buffer
            );

    }
    else if (unrolled_option == "unrolled"
        && single_mode_log_option == false
        && threads_per_traj_opt == "one_thread_per_traj"
        && quasi_static_ensemble_dephasing_flag)
    {

        printf("Launching kernel singlemode unrolled ensemble without log: blocks=1 threads_per_block=%d\n", threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        //lindblad_rk4_kernel_singlemode_unrolled_ensemble <<< blocks, threads_per_block >>> (
        //    eps0_target, A_target,

        //    d_rho_avg_singlemode,
        //    d_rho_dynamics,
        //    d_time_dynamics,
        //    d_eps_dynamics,
        //    N_samples_noise
        //    );

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
    gpuCheck(cudaMemcpy(rho_avg.data(),         d_rho_avg_singlemode,       16 * sizeof(float),                cudaMemcpyDeviceToHost), "cudaMemcpy single rho_avg");
    gpuCheck(cudaMemcpy(h_rho_dynamics.data(),  d_rho_dynamics,  N_steps_total * 4 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy rho_dynamics");
    gpuCheck(cudaMemcpy(h_time_dynamics.data(), d_time_dynamics, N_steps_total * sizeof(float),     cudaMemcpyDeviceToHost), "cudaMemcpy time_dynamics");
    gpuCheck(cudaMemcpy(h_eps_dynamics.data(),  d_eps_dynamics,  N_steps_total * sizeof(float),     cudaMemcpyDeviceToHost), "cudaMemcpy eps_dynamics");

    if (output_option == "bin_file") {

        //// Variables for timing
        //cudaEvent_t start, stop;
        //// Create events to track the start and stop times
        //cudaEventCreate(&start);
        //cudaEventCreate(&stop);
        //// Start the timer
        //cudaEventRecord(start, 0);

        // Open binary file for writing
        std::ofstream ofs(path_output_bin_file_singlemode, std::ios::binary);
        if (!ofs) {
            std::cerr << "Failed to open file!" << std::endl;
            return;
        }


        // Write header: N_steps_total (int) and fixed sizes (for clarity)
        // Store N_steps_total so python knows how many steps to read
        int header[1] = { N_steps_total };
        ofs.write(reinterpret_cast<char*>(header), sizeof(int));

        // Write rho_avg (16 floats)
        ofs.write(reinterpret_cast<const char*>(rho_avg.data()), 16 * sizeof(float));

        // Write rho_dynamics (N_steps_total * 4 floats)
        ofs.write(reinterpret_cast<const char*>(h_rho_dynamics.data()), N_steps_total * 4 * sizeof(float));

        // Write time_dynamics (N_steps_total floats)
        ofs.write(reinterpret_cast<const char*>(h_time_dynamics.data()), N_steps_total * sizeof(float));

        // Write eps_dynamics (N_steps_total floats)
        ofs.write(reinterpret_cast<const char*>(h_eps_dynamics.data()), N_steps_total * sizeof(float));

        ofs.close();

        //// Stop the timer
        //cudaEventRecord(stop, 0);
        //cudaEventSynchronize(stop);  // Ensure the stop event is complete
        //// Calculate the elapsed time
        //float milliseconds = 0.0f;
        //cudaEventElapsedTime(&milliseconds, start, stop);  // Time in milliseconds
        //// Print the elapsed time
        //std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
        //// Cleanup the events
        //cudaEventDestroy(start);
        //cudaEventDestroy(stop);

    }

/*
    else if (output_option == "ssd_csv") {
        
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

    }

    else if (output_option != "ssd_csv") {
        std::cerr << "ERROR: output_option = 'ram' is not implemented for singlemode" << std::endl;
    }
*/

    // cleanup
    cudaFree(d_rho_avg_singlemode);
    cudaFree(d_rho_dynamics);
    cudaFree(d_time_dynamics);
    cudaFree(d_eps_dynamics);


}





