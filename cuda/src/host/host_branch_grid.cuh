// src/host_branch_grid.cuh

#pragma once

#include <cuda_runtime.h> // Required for CUDA calls
#include <iostream>       // Used: std::cout, std::cerr
#include <vector>         // Used: std::vector
#include <cmath>          // Used: M_PIf, std::max, etc, sqrt
#include <string>
// #include "cuda_intellisense_fixes.cuh"
#include "kernels/lindblad_kernel_grid.cuh"
#include "constants.cuh"
// #include "writer.cuh"
#include "host_helpers.cuh"
#include <cassert>


#include <fstream>
//#include <cstdint>
//#include <algorithm>
#include <iomanip>
//#include <fcntl.h>    // open
//#include <windows.h>



// #include <cuda_profiler_api.h> // for #NCU
//#include <nvtx3/nvToolsExt.h> // for #NCU



#define K_TRAJ_PER_WARP 2
#define LANE_GROUP_SIZE (32 / K_TRAJ_PER_WARP) // 16


// -------------------------
// grid mode execution without precomputed L
// -------------------------
__host__ inline void run_grid_mode(
    const float eps0_min,
    const float eps0_max,
    const float A_min,
    const float A_max,
    const int N_points_eps0_range,
    const int N_points_A_range,
    const int host_N_steps_per_period,
    const int host_N_periods,
    const int N_periods_avg,
    const int N_samples_noise,
    const bool quasi_static_ensemble_dephasing_flag,
    float* eps_offsets,
    const float host_dt,
    const float nu,
    const float alpha,
    const std::string& path_output_csv,
    const std::string& path_output_bin_file,
    const std::string& avg_periods_ouput_option,
    const std::string& ouput_option,
    const std::string& unrolled_option,
    const std::string& ram_shared_mmap_name,
    const std::string& threads_per_traj_opt,

    const float host_rho00_init, const float host_rho11_init,
    const float host_rho22_init, const float host_rho33_init,
    const float host_delta_C, const float host_delta_L, const float host_delta_R,

    const float g_en, const float g_phi,
    const float gL_en, const float gL_phi,
    const float gR_en, const float gR_phi,

    const float host_Gamma_L0,
    const float host_Gamma_R0,
    const float host_Gamma_eg0,
    const float omega_c_norm,
    const float host_Gamma_phi0
)
{

    std::cout << "Launching run_grid_mode branch" << std::endl;

    int rho_avg_dim, rho_avg_dim_u;

    if (avg_periods_ouput_option == "last") {
        rho_avg_dim = 4;
        rho_avg_dim_u = 4u;
    }
    else if (avg_periods_ouput_option == "last_2last") {
        rho_avg_dim = 8;
        rho_avg_dim_u = 8u;
    }
    else if (avg_periods_ouput_option == "whole_last_2last_3last") {
        rho_avg_dim = 16;
        rho_avg_dim_u = 16u;
    }


    const float host_GammaLR0 = host_Gamma_L0 + host_Gamma_R0;
    const float host_omega = 2.0f * M_PIf * nu;
    const float host_pi_alpha = M_PIf * alpha;

    const int host_N_points = N_points_eps0_range * N_points_A_range;

    std::cout << " unrolled_option: " << unrolled_option << std::endl;
    std::cout << " eps0 range: " << eps0_min << " .. " << eps0_max << std::endl;
    std::cout << " A range   : " << A_min << " .. " << A_max << std::endl;
    std::cout << " N_points  : " << host_N_points << std::endl;
    std::cout << " N_points_eps0_range: " << N_points_eps0_range << std::endl;
    std::cout << " N_points_A_range   : " << N_points_A_range << std::endl;

    std::cout << " Gamma_L0: " << host_Gamma_L0 << std::endl;
    std::cout << " Gamma_R0: " << host_Gamma_R0 << std::endl;
    //std::cout << " muL: " << host_muL << std::endl;
    //std::cout << " muR: " << host_muR << std::endl;
    //std::cout << " T_K: " << T_K << std::endl;

    std::cout << " Gamma_eg0: "  << host_Gamma_eg0  << std::endl;
    std::cout << " Gamma_phi0: " << host_Gamma_phi0 << std::endl;



    std::vector<float> eps0_list(host_N_points), A_list(host_N_points);
    int idx = 0;
    for (int i = 0; i < N_points_eps0_range; i++) {
        float e = eps0_min + (eps0_max - eps0_min) * (float(i) / float(std::max(1, N_points_eps0_range - 1)));
        for (int j = 0; j < N_points_A_range; j++) {
            float a = A_min + (A_max - A_min) * (float(j) / float(std::max(1, N_points_A_range - 1)));
            if (idx < host_N_points) { eps0_list[idx] = e; A_list[idx] = a; idx++; }
        }
    }

    if (idx != host_N_points) {
        std::cerr << "ERROR: idx != host_N_points. idx = " << idx << ", host_N_points = " << host_N_points << std::endl;
    }

    std::cout << "Grid built: " << host_N_points << " trajectories.\n";

    std::vector<float> rho_avg(host_N_points * rho_avg_dim);

    // allocate device arrays for traj params
    float* d_eps0 = nullptr, * d_A = nullptr, * d_rho_avg = nullptr;
    gpuCheck(cudaMalloc(&d_eps0, host_N_points * sizeof(float)), "cudaMalloc d_eps0");
    gpuCheck(cudaMalloc(&d_A, host_N_points * sizeof(float)), "cudaMalloc d_A");
    gpuCheck(cudaMalloc(&d_rho_avg, host_N_points * rho_avg_dim * sizeof(float)), "cudaMalloc d_rho_avg");

    gpuCheck(cudaMemcpy(d_eps0, eps0_list.data(), host_N_points * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy eps0");
    gpuCheck(cudaMemcpy(d_A, A_list.data(), host_N_points * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy A");

    
    
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

    gpuCheck(cudaMemcpyToSymbol(Npoints, &host_N_points, sizeof(int)), "cudaMemcpyToSymbol Npoints");
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



    // ==================================================
    // launch kernel


    // threads per block
    int threads_per_block;

    // number of blocks to cover all Npoints
    int blocks;
    

    // Calculate shared memory size: pairs per block = threads_per_block / 2, 52 floats per pair
    //size_t pairs_per_block = threads_per_block / 2;
    //size_t shared_bytes = pairs_per_block * 52 * sizeof(float);  // e.g., 64 * 52 * 4 = 13312 bytes
    //printf("Launching kernel: blocks=%d threads_per_block=%d shared_bytes=%zu\n", blocks, threads_per_block, shared_bytes);




    // start timer
    cudaEvent_t start, stop;
    float timer_milliseconds = 0;

    if (cudaEventCreate(&start) != cudaSuccess ||
        cudaEventCreate(&stop) != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA events\n");
        std::abort();
    }

    // starting #NCU marker
    // nvtxRangePushA("lindblad_rk4_kernel_shared");

    cudaEventRecord(start);

    // launch kernel

    if (threads_per_traj_opt == "one_thread_per_traj"
    && unrolled_option == "unrolled") {

    threads_per_block = 128; // (common choice)
    blocks = (host_N_points + threads_per_block - 1) / threads_per_block;   // standard

    if (!quasi_static_ensemble_dephasing_flag) {

        printf("Launching kernel gridmode unrolled one_thread_one_traj no_ensemble fsal: blocks=%d threads_per_block=%d\n", blocks, threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        lindblad_rk4_kernel_unrolled_fsal <<<blocks, threads_per_block >>> (
            d_eps0, d_A,
            d_rho_avg,

            N_periods_avg
            );

    }
    else if (quasi_static_ensemble_dephasing_flag) {

        printf("Launching kernel gridmode unrolled one_thread_one_traj ensemble fsal: blocks=%d threads_per_block=%d\n", blocks, threads_per_block);
        fflush(stdout);  // forces the buffer to flush immediately

        //lindblad_rk4_kernel_unrolled_ensemble_opt1 <<<blocks, threads_per_block >>> (
        //    d_eps0, d_A,
        //    d_rho_avg,
        //    N_periods_avg,
        //    N_samples_noise
        //    );

        //const float scale_total = 1.0f / (host_N_steps_per_period * N_periods_avg * N_samples_noise);
        //lindblad_rk4_kernel_unrolled_ensemble_opt2 <<<blocks, threads_per_block >>> (
        //    d_eps0, d_A,
        //    d_rho_avg,
        //    N_periods_avg,
        //    N_samples_noise,
        //    scale_total
        //    );

        //const float scale_total = 1.0f / (host_N_steps_per_period * N_periods_avg * N_samples_noise);
        //lindblad_rk4_kernel_unrolled_ensemble_opt3 << <blocks, threads_per_block >> > (
        //    d_eps0, d_A,
        //    d_rho_avg,
        //    N_periods_avg,
        //    N_samples_noise,
        //    scale_total
        //    );

        // const float scale_total = 1.0f / (host_N_steps_per_period * N_periods_avg * N_samples_noise);
        // lindblad_rk4_kernel_unrolled_ensemble_opt10 <<<blocks, threads_per_block >>> (
        //     d_eps0, d_A,
        //     d_rho_avg,
        //     N_periods_avg,
        //     N_samples_noise,
        //     scale_total,
        //     eps_offsets
        //     );

        const float scale_total = 1.0f / (host_N_steps_per_period * N_periods_avg * N_samples_noise);
        lindblad_rk4_kernel_unrolled_ensemble_opt10_fsal <<<blocks, threads_per_block >>> (
            d_eps0, d_A,
            d_rho_avg,
            N_periods_avg,
            N_samples_noise,
            scale_total,
            eps_offsets
            );

    }


}





    // check for launch errors immediately
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        std::abort();   // stops program immediately
        exit(1);
    }

    // ==================================================

    gpuCheck(cudaGetLastError(), "Kernel launch succeded");
    gpuCheck(cudaDeviceSynchronize(), "Sync after grid kernel");


    // Record stop time and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&timer_milliseconds, start, stop);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // ending #NCU marker
    // nvtxRangePop();

    // copy data back
    gpuCheck(cudaMemcpy(rho_avg.data(), d_rho_avg, host_N_points * rho_avg_dim * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy rho_avg");







    std::cout << "Kernel execution time: " << timer_milliseconds << " ms" << std::endl;
    std::cout << "Kernel execution time: "
        << std::fixed << std::setprecision(1)
        << (timer_milliseconds / 1000.0f) << " s" << std::endl;

    double speed;
    if (!quasi_static_ensemble_dephasing_flag) {
        speed = static_cast<double>(static_cast<long long>(host_N_points) * host_N_steps_per_period * host_N_periods) * 1e3 / (static_cast<double>(timer_milliseconds));
        std::cout << "Npoints*N_steps_period*N_periods/run_time = " << std::endl;
    }
    else if (quasi_static_ensemble_dephasing_flag) {
        speed = static_cast<double>(static_cast<long long>(host_N_points) * host_N_steps_per_period * host_N_periods * N_samples_noise) * 1e3 / (static_cast<double>(timer_milliseconds));
        std::cout << "Npoints*N_steps_period*N_periods*N_samples_noise/run_time = " << std::endl;
    }

    std::cout << " = "
        << std::fixed << std::setprecision(1)
        << speed / 1e9
        << " billion points(eps0,A)*timesteps/second for 16 ODEs = \n = "
        << speed / 1e6
        << " million points(eps0,A)*timesteps/second for 16 ODEs"
        << std::endl;




    // -----------------------
    // output results
    // -----------------------

    if (ouput_option == "bin_file") {

        /*
        // Open binary file for writing
        std::ofstream ofs(path_output_bin_file, std::ios::binary);
        if (!ofs) {
            std::cerr << "Failed to open file!" << std::endl;
            return;
        }

        // Write data to the binary file
        for (int t_idx = 0; t_idx < host_N_points; t_idx++) {
            // Write eps0 and A
            ofs.write(reinterpret_cast<char*>(&eps0_list[t_idx]), sizeof(float));
            ofs.write(reinterpret_cast<char*>(&A_list[t_idx]), sizeof(float));

            // Write 4 columns from rho_avg
            size_t base = t_idx * rho_avg_dim;
            for (int k = 0; k < rho_avg_dim; k++) {
                ofs.write(reinterpret_cast<char*>(&rho_avg[base + k]), sizeof(float));
            }
        }
        ofs.close();
        */

        //// Variables for timing
        //cudaEvent_t start, stop;
        //// Create events to track the start and stop times
        //cudaEventCreate(&start);
        //cudaEventCreate(&stop);
        //// Start the timer
        //cudaEventRecord(start, 0);

        // Open binary file for writing
        std::ofstream ofs(path_output_bin_file, std::ios::binary);
        if (!ofs) {
            std::cerr << "Failed to open file!" << std::endl;
            return;
        }

        // Write header (dimensions)
        int header[2] = { N_points_A_range, N_points_eps0_range };
        ofs.write(reinterpret_cast<char*>(header), 2 * sizeof(int));

        // Reorganize data: write all avg00, then all avg01, then avg10, then avg11
        // For each level k, write in order: A varies fastest, eps0 varies slowest
        for (int k = 0; k < rho_avg_dim; k++) {
            for (int i = 0; i < N_points_eps0_range; i++) {
                for (int j = 0; j < N_points_A_range; j++) {
                    int idx = i * N_points_A_range + j;
                    size_t base = idx * rho_avg_dim;
                    ofs.write(reinterpret_cast<char*>(&rho_avg[base + k]), sizeof(float));
                }
            }
        }

        // Write grids separately at the end for easy access
        for (int j = 0; j < N_points_A_range; j++) {
            ofs.write(reinterpret_cast<char*>(&A_list[j]), sizeof(float));
        }
        for (int i = 0; i < N_points_eps0_range; i++) {
            int idx = i * N_points_A_range;
            ofs.write(reinterpret_cast<char*>(&eps0_list[idx]), sizeof(float));
        }

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
    else if (ouput_option == "ssd_csv") {
        write_to_csv(
            rho_avg, eps0_list, A_list,
            eps0_min, eps0_max, A_min, A_max,
            host_N_points, N_points_eps0_range, N_points_A_range,
            host_N_steps_per_period, host_N_periods, N_periods_avg,
            host_dt, nu, alpha, B, m,
            path_output_csv, avg_periods_ouput_option, ouput_option, unrolled_option,
            host_rho00_init, host_rho11_init, host_rho22_init, host_rho33_init,
            host_delta_C, host_delta_L, host_delta_R,
            g_en, g_phi, gL_en, gL_phi, gR_en, gR_phi,

            rho_avg_dim, rho_avg_dim_u
        );
    }
    else if (ouput_option == "ram")
    {
        write_to_shared_memory_ram(
            ram_shared_mmap_name,

            rho_avg, eps0_list, A_list,

            eps0_min, eps0_max, A_min, A_max,
            host_N_points, N_points_eps0_range, N_points_A_range,
            host_N_steps_per_period, host_N_periods, N_periods_avg,
            host_dt, nu, alpha, B, m,
            path_output_csv, avg_periods_ouput_option, ouput_option, unrolled_option,
            host_rho00_init, host_rho11_init, host_rho22_init, host_rho33_init,
            host_delta_C, host_delta_L, host_delta_R,
            g_en, g_phi, gL_en, gL_phi, gR_en, gR_phi

        );
    }
*/



    // cleanup
    cudaFree(d_eps0);
    cudaFree(d_A);
    cudaFree(d_rho_avg);

    if (quasi_static_ensemble_dephasing_flag) {
        cudaFree(eps_offsets);
    }
}











