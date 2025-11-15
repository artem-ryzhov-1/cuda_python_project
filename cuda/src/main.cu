// src/main.cu

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <fstream>
#include <nlohmann/json.hpp>
#include "constants.cuh"

// #include "cuda_intellisense_fixes.cuh"
#include "host/host_branch_grid.cuh"
#include "host/host_branch_single.cuh"
#include "host/host_helpers.cuh"

#ifdef _WIN32
#include <windows.h>
#endif





// #include <cuda_profiler_api.h> // for #NCU
//#include <nvtx3/nvToolsExt.h> // for #NCU


using json = nlohmann::json;

// Helper function to safely get values with NaN support for floats
template<typename T>
T safe_get(const json& j, const std::string& key, T default_val) {
    if (!j.contains(key)) return default_val;
    if (j[key].is_null()) return default_val;
    return j[key].get<T>();
}

// Specialized for float to handle "nan" strings
template<>
float safe_get<float>(const json& j, const std::string& key, float default_val) {
    if (!j.contains(key)) return default_val;
    if (j[key].is_null()) return default_val;
    if (j[key].is_string()) {
        std::string val = j[key].get<std::string>();
        if (val == "nan" || val == "NaN" || val == "null") {
            return std::nanf("");
        }
    }
    return j[key].get<float>();
}

// Specialized for int to handle "null" strings
template<>
int safe_get<int>(const json& j, const std::string& key, int default_val) {
    if (!j.contains(key)) return default_val;
    if (j[key].is_null()) return default_val;
    if (j[key].is_string()) {
        std::string val = j[key].get<std::string>();
        if (val == "null") return INT_MIN;
    }
    return j[key].get<int>();
}

int main(int argc, char** argv)
{
    std::cout << "\n==============================================\n";
    std::cout << "GPU Lindblad Simulator\n";
    std::cout << "==============================================\n\n";

    if (argc != 2) {
        std::cerr << "ERROR: Expected 1 argument (config file path).\n";
        std::cerr << "Usage: " << argv[0] << " <config.json>\n";
        std::cerr << "Received " << argc - 1 << " arguments. Exiting program.\n";
        std::exit(EXIT_FAILURE);
    }

#ifdef _WIN32
    SetConsoleCtrlHandler(NULL, FALSE);
#endif

    // Read JSON configuration file
    std::string config_path = argv[1];
    std::cout << "Reading configuration from: " << config_path << "\n\n";
    
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "ERROR: Could not open config file: " << config_path << "\n";
        std::exit(EXIT_FAILURE);
    }

    json config;
    try {
        config_file >> config;
    } catch (const json::exception& e) {
        std::cerr << "ERROR: Failed to parse JSON config: " << e.what() << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::cout << "Configuration loaded successfully.\n";
    if (config.contains("version")) {
        std::cout << "Config version: " << config["version"] << "\n";
    }
    std::cout << "\n--- Extracting parameters ---\n";

    // Extract parameters
    const std::string grid_single_mode = config["grid_single_mode"];
    const std::string avg_periods_ouput_option = config["avg_periods_ouput_option"];
    const std::string ouput_option = config["ouput_option"];
    const std::string unrolled_option = config["unrolled_option"];
    const std::string ram_shared_mmap_name = config["ram_shared_mmap_name"];

    const float eps0_min = safe_get<float>(config, "eps0_min", std::nanf(""));
    const float eps0_max = safe_get<float>(config, "eps0_max", std::nanf(""));
    const float A_min = safe_get<float>(config, "A_min", std::nanf(""));
    const float A_max = safe_get<float>(config, "A_max", std::nanf(""));

    const int N_points_eps0_range = safe_get<int>(config, "N_points_eps0_range", INT_MIN);
    const int N_points_A_range = safe_get<int>(config, "N_points_A_range", INT_MIN);

    const int N_steps_period = config["N_steps_period"];
    const int N_periods = config["N_periods"];
    const int N_periods_avg = config["N_periods_avg"];
    const int N_samples_noise = safe_get<int>(config, "N_samples_noise", INT_MIN);

    const float alpha = config["alpha"];
    const float nu = config["nu"];

    const float eps0_target = safe_get<float>(config, "eps0_target_singlepoint", std::nanf(""));
    const float A_target = safe_get<float>(config, "A_target_singlepoint", std::nanf(""));

    float rho00_init = config["rho00_init"];
    float rho11_init = config["rho11_init"];
    float rho22_init = config["rho22_init"];
    float rho33_init = config["rho33_init"];

    const std::string path_output_csv = config["path_output_csv"];
    const std::string path_output_bin_file_gridmode = config["path_output_bin_file_gridmode"];
    const std::string path_output_bin_file_singlemode = config["path_output_bin_file_singlemode"];
    const std::string path_dynamics_single_mode_output_csv = config["path_dynamics_single_mode_output_csv"];

    const float delta_C = config["delta_C"];
    const float delta_L = config["delta_L"];
    const float delta_R = config["delta_R"];

    const float g_en = config["g_en"];  // these Gammas are only for printing into CSV. not used elsewhere in program
    const float g_phi = config["g_phi"];
    const float gL_en = config["gL_en"];
    const float gL_phi = config["gL_phi"];
    const float gR_en = config["gR_en"];
    const float gR_phi = config["gR_phi"];

    const float host_Gamma_L0 = config["GammaL0"];     // prefactor (GHz etc.)
    const float host_Gamma_R0 = config["GammaR0"];     // prefactor (GHz etc.)
    const float host_muL = config["muL"];    // µeV
    const float host_muR = config["muR"];    // µeV
    const float T_K = config["T_K"];    // Kelvin

    const float host_Gamma_eg0 = config["Gamma_eg0"];    // prefactor (GHz etc.)
    const float omega_c_norm = config["omega_c"];    // high-frequency cutoff

    const float host_Gamma_phi0 = safe_get<float>(config, "Gamma_phi0", std::nanf(""));    // prefactor (GHz etc.)
    const float sigma_eps = safe_get<float>(config, "sigma_eps", std::nanf(""));

    const bool single_mode_log_option = config["single_mode_log_option"];
    const std::string path_dynamics_single_mode_output_log_csv = config["path_dynamics_single_mode_output_log_csv"];
    const std::string path_dynamics_single_mode_output_log_hdf5 = config["path_dynamics_single_mode_output_log_hdf5"];
    
    const std::string threads_per_traj_opt = config["threads_per_traj_opt"];
    const bool quasi_static_ensemble_dephasing_flag = config["quasi_static_ensemble_dephasing_flag"];

    std::cout << "Parameters extracted successfully.\n\n";
    std::cout << "==============================================\n";

    std::cout << "\n--- Listing received arguments ---\n";

    std::cout << "1.  grid_single_mode: " << grid_single_mode << "\n";
    std::cout << "2.  avg_periods_ouput_option: " << avg_periods_ouput_option << "\n";
    std::cout << "3.  ouput_option: " << ouput_option << "\n";
    std::cout << "4.  unrolled_option: " << unrolled_option << "\n";
    std::cout << "5.  ram_shared_mmap_name: " << ram_shared_mmap_name << "\n";

    std::cout << "6.  eps0_min: " << eps0_min << "\n";
    std::cout << "7.  eps0_max: " << eps0_max << "\n";
    std::cout << "8.  A_min: " << A_min << "\n";
    std::cout << "9.  A_max: " << A_max << "\n";

    std::cout << "10. N_points_eps0_range: " << N_points_eps0_range << "\n";
    std::cout << "11. N_points_A_range: " << N_points_A_range << "\n";
    std::cout << "12. N_steps_period: " << N_steps_period << "\n";
    std::cout << "13. N_periods: " << N_periods << "\n";
    std::cout << "14. N_periods_avg: " << N_periods_avg << "\n";
    std::cout << "15. N_samples_noise: " << N_samples_noise << "\n";

    std::cout << "16. alpha: " << alpha << "\n";
    std::cout << "17. nu: " << nu << "\n";
    std::cout << "18. eps0_target: " << eps0_target << "\n";
    std::cout << "19. A_target: " << A_target << "\n";

    std::cout << "20. rho00_init: " << rho00_init << "\n";
    std::cout << "21. rho11_init: " << rho11_init << "\n";
    std::cout << "22. rho22_init: " << rho22_init << "\n";
    std::cout << "23. rho33_init: " << rho33_init << "\n";

    std::cout << "24. path_output_csv: " << path_output_csv << "\n";
    std::cout << "25. path_output_bin_file_gridmode: " << path_output_bin_file_gridmode << "\n";
    std::cout << "26. path_output_bin_file_singlemode: " << path_output_bin_file_singlemode << "\n";
    std::cout << "27. path_dynamics_single_mode_output_csv: " << path_dynamics_single_mode_output_csv << "\n";

    std::cout << "28. delta_C: " << delta_C << "\n";
    std::cout << "29. delta_L: " << delta_L << "\n";
    std::cout << "30. delta_R: " << delta_R << "\n";

    std::cout << "31. g_en: " << g_en << "\n";
    std::cout << "32. g_phi: " << g_phi << "\n";
    std::cout << "33. gL_en: " << gL_en << "\n";
    std::cout << "34. gL_phi: " << gL_phi << "\n";
    std::cout << "35. gR_en: " << gR_en << "\n";
    std::cout << "36. gR_phi: " << gR_phi << "\n";

    std::cout << "37. Gamma_L0: " << host_Gamma_L0 << "\n";
    std::cout << "38. Gamma_R0: " << host_Gamma_R0 << "\n";
    std::cout << "39. muL: " << host_muL << "\n";
    std::cout << "40. muR: " << host_muR << "\n";
    std::cout << "41. T_K: " << T_K << "\n";

    std::cout << "42. Gamma_eg0: " << host_Gamma_eg0 << "\n";
    std::cout << "43. omega_c_norm: " << omega_c_norm << "\n";

    std::cout << "44. Gamma_phi0: " << host_Gamma_phi0 << "\n";
    std::cout << "45. sigma_eps: " << sigma_eps << "\n";

    std::cout << "46. single_mode_log_option: " << single_mode_log_option << "\n";
    std::cout << "47. path_dynamics_single_mode_output_log_csv: " << path_dynamics_single_mode_output_log_csv << "\n";
    std::cout << "48. path_dynamics_single_mode_output_log_hdf5: " << path_dynamics_single_mode_output_log_hdf5 << "\n";
    std::cout << "49. threads_per_traj_opt: " << threads_per_traj_opt << "\n";
    std::cout << "50. quasi_static_ensemble_dephasing_flag: " << quasi_static_ensemble_dephasing_flag << "\n";

    std::cout << "==============================================\n\n";

    // -------------------------
    // Validation
    // -------------------------

    if (!(grid_single_mode == "grid" ||
          grid_single_mode == "single" ||
          grid_single_mode == "grid_single")) {
        std::cerr << "ERROR: grid_single_mode must be 'grid' or 'single' or 'grid_single'. Got '" << grid_single_mode << "'\n";
        std::exit(EXIT_FAILURE);
    }

    if (!(avg_periods_ouput_option == "last" /*||
          avg_periods_ouput_option == "last_2last" ||
          avg_periods_ouput_option == "whole_last_2last_3last"*/)) {
        std::cerr << "ERROR: avg_periods_ouput_option must be 'last' or 'last_2last' or 'whole_last_2last_3last'. Got '" << avg_periods_ouput_option << "'\n";
        std::exit(EXIT_FAILURE);
    }

    if (!(ouput_option == "ssd_csv" ||
        ouput_option == "bin_file" ||
        ouput_option == "ram")) {
        std::cerr << "ERROR: ouput_option must be 'ssd_csv' or 'bin_file' or 'ram'. Got '" << ouput_option << "'\n";
        std::exit(EXIT_FAILURE);
    }

    if (!(unrolled_option == "as_arrays" ||
        unrolled_option == "unrolled")) {
        std::cerr << "ERROR: unrolled_option must be 'as_arrays' or 'unrolled'. Got '" << unrolled_option << "'\n";
        std::exit(EXIT_FAILURE);
    }

    if (ouput_option == "ram" && ram_shared_mmap_name == "null") {
        std::cerr << "ERROR: ram_shared_mmap_name = 'null', not defined.\n";
        std::exit(EXIT_FAILURE);
    }

    if (((grid_single_mode == "grid" || grid_single_mode == "grid_single") && (std::isnan(eps0_min) || std::isnan(eps0_max) || std::isnan(A_min) || std::isnan(A_max)
        || N_points_eps0_range == INT_MIN || N_points_A_range == INT_MIN)) ||
        ((grid_single_mode == "single") && !(std::isnan(eps0_min) && std::isnan(eps0_max) && std::isnan(A_min) && std::isnan(A_max)
            && N_points_eps0_range == INT_MIN && N_points_A_range == INT_MIN))
        ) {
        std::cerr << "ERROR: eps0_min, eps0_max, A_min, or A_max is NAN in grid or grid_single mode, or not NAN in single mode."
            << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (N_periods_avg > N_periods) {
        std::cerr << "ERROR: N_periods_avg must be less than or equal to N_periods. Got: "
            << N_periods_avg << " > " << N_periods << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (((grid_single_mode == "single" || grid_single_mode == "grid_single") && (std::isnan(eps0_target) || std::isnan(A_target))) ||
        ((grid_single_mode == "grid") && !(std::isnan(eps0_target) && std::isnan(A_target)))) {
        std::cerr << "ERROR: eps0_target or A_target is NAN in single or grid_single mode, or not NAN in grid mode."
            << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (delta_L != 0.0f || delta_R != 0.0f) {
        std::cerr << "ERROR: delta_L and delta_R are expected to be zero. Got: " << delta_L << " " << delta_R << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (!(threads_per_traj_opt == "one_thread_per_traj" ||
        threads_per_traj_opt == "thread_group_in_warp_per_traj_shuffle" ||
        threads_per_traj_opt == "thread_group_in_warp_per_traj_shmem")) {
        std::cerr << "ERROR: threads_per_traj_opt must be 'one_thread_per_traj'"
            << " or 'thread_group_in_warp_per_traj_shuffle'"
            << "or 'thread_group_in_warp_per_traj_shmem'.Got '" << threads_per_traj_opt << "'\n";
        std::exit(EXIT_FAILURE);
    }

    if (quasi_static_ensemble_dephasing_flag) {

        if (std::isnan(sigma_eps) || N_samples_noise == INT_MIN){
            std::cerr << "ERROR: sigma_eps or N_samples_noise is NAN with quasi_static_ensemble_dephasing_flag == 'True'."
                << std::endl;
            std::exit(EXIT_FAILURE);
        }

    }
    else if (!quasi_static_ensemble_dephasing_flag) {

        if (!std::isnan(sigma_eps) || N_samples_noise != INT_MIN) {
            std::cerr << "ERROR: sigma_eps or N_samples_noise is not NAN with quasi_static_ensemble_dephasing_flag == 'False'."
                << std::endl;
            std::exit(EXIT_FAILURE);
        }

    }
    else {
        std::cerr << "ERROR: Invalid argument. Expected 'True' or 'False'. Got: " << quasi_static_ensemble_dephasing_flag << std::endl;
        std::exit(EXIT_FAILURE);
    }


    // renormalization for safety

    float sum = rho00_init + rho11_init + rho22_init + rho33_init;

    if (sum != 0.0f) {
        rho00_init /= sum;
        rho11_init /= sum;
        rho22_init /= sum;
        rho33_init /= sum;
    }
    else {
        std::cerr << "ERROR: all initial rho_ii are zero. Aborting.\n";
        return 1;
    }



    //// physics constants
    //const float a = 0.1f;
    //const float m = 10.0f;
    //const float B = (a + 2.0f) * (m + 1.0f) / (a * (m - 1.0f));

    // compute dt so period length T has integer number of steps
    const float T = 1 / nu;
    const float dt = T / float(N_steps_period);



    // Thermal scale
    const float kB_mu_eV = 86.173324f; // µeV/K
    float host_kT = kB_mu_eV * T_K;
    if (host_kT < 1e-9f) host_kT = 1e-9f;

    if (fabs(host_kT - kT) > 1e-6f) {
        fprintf(stderr, "Error: kT mismatch! Expected %.6f, got %.6f\n", kT, host_kT);
        exit(1);
    }

    if (fabs(host_muL - muL) > 1e-6f) {
        fprintf(stderr, "Error: muL mismatch! Expected %.6f, got %.6f\n", muL, host_muL);
        exit(1);
    }

    if (fabs(host_muR - muR) > 1e-6f) {
        fprintf(stderr, "Error: muL mismatch! Expected %.6f, got %.6f\n", muR, host_muR);
        exit(1);
    }






    // Runtime parameters
    std::cout << "Runtime params:\n";
    std::cout << " steps/period: " << N_steps_period
        << "  N_periods: " << N_periods
        << "  N_periods_avg: " << N_periods_avg
        << "  dt: " << dt << std::endl;
    std::cout << " N_samples_noise: " << N_samples_noise << std::endl;
    std::cout << " quasi_static_ensemble_dephasing_flag: " << quasi_static_ensemble_dephasing_flag << std::endl;
    std::cout << " total steps: " << N_steps_period * N_periods << std::endl;
    std::cout << " delta_C: " << delta_C << std::endl;
    std::cout << " delta_L: " << delta_L << std::endl;
    std::cout << " delta_R: " << delta_R << std::endl;
    std::cout << " g_en: " << g_en << std::endl;
    std::cout << " g_phi: " << g_phi << std::endl;
    std::cout << " gL_en: " << gL_en << std::endl;
    std::cout << " gL_phi: " << gL_phi << std::endl;
    std::cout << " gR_en: " << gR_en << std::endl;
    std::cout << " gR_phi: " << gR_phi << std::endl;

    std::cout << " alpha: " << alpha << "  nu: " << nu << std::endl;

    std::cout << " rho00_init: " << rho00_init << "  rho11_init: " << rho11_init << std::endl;
    std::cout << " rho22_init: " << rho22_init << "  rho33_init: " << rho33_init << std::endl;


    float* eps_offsets;

    if (quasi_static_ensemble_dephasing_flag){

        if (N_samples_noise > MAX_NOISE_SAMPLES) {
            fprintf(stderr, "Error: N_samples_noise (%d) > MAX_NOISE_SAMPLES (%d)\n",
                N_samples_noise, MAX_NOISE_SAMPLES);
            std::exit(EXIT_FAILURE); 
            exit(1);
        }

        
        cudaMallocManaged(&eps_offsets, N_samples_noise * sizeof(float));

        // Fill with noise
        std::mt19937 rng(12345);
        std::normal_distribution<float> dist(0.0f, sigma_eps);
        for (int i = 0; i < N_samples_noise; ++i) {
            eps_offsets[i] = dist(rng);
        }

        // Copy to GPU constant memory
        //gpuCheck(cudaMemcpyToSymbol(c_eps_offsets, host_eps_offsets.data(), N_samples_noise * sizeof(float), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol c_eps_offsets");

    }

    if (grid_single_mode == "grid" || grid_single_mode == "grid_single") {
        run_grid_mode(
            eps0_min, eps0_max, A_min, A_max,
            N_points_eps0_range, N_points_A_range,
            N_steps_period, N_periods, N_periods_avg,
            N_samples_noise, quasi_static_ensemble_dephasing_flag, eps_offsets,
            dt, nu, alpha,
            path_output_csv, path_output_bin_file_gridmode, avg_periods_ouput_option, ouput_option, unrolled_option,
            ram_shared_mmap_name, threads_per_traj_opt,
            rho00_init, rho11_init, rho22_init, rho33_init,
            delta_C, delta_L, delta_R, g_en, g_phi, gL_en, gL_phi, gR_en, gR_phi,
            host_Gamma_L0, host_Gamma_R0, host_Gamma_eg0, omega_c_norm, host_Gamma_phi0
        );
    }
    if (grid_single_mode == "single" || grid_single_mode == "grid_single") {
        run_single_mode(
            eps0_target, A_target, N_steps_period, N_periods,
            N_samples_noise, quasi_static_ensemble_dephasing_flag, eps_offsets,
            dt, nu, alpha,
            rho00_init, rho11_init, rho22_init, rho33_init,
            delta_C, delta_L, delta_R, g_en, g_phi, gL_en, gL_phi, gR_en, gR_phi,
            path_dynamics_single_mode_output_csv, path_output_bin_file_singlemode, ouput_option, unrolled_option,
            single_mode_log_option, path_dynamics_single_mode_output_log_csv,
            path_dynamics_single_mode_output_log_hdf5,
            threads_per_traj_opt,
            host_Gamma_L0, host_Gamma_R0, host_Gamma_eg0, omega_c_norm, host_Gamma_phi0
        );
    }




    cudaDeviceReset();

    return 0;
}









