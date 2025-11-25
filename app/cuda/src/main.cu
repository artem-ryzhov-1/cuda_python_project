////////////////////////////////////////
// app/cuda/src/main.cu
////////////////////////////////////////

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
    const std::string ouput_option = config["ouput_option"];
    const std::string unrolled_option = config["unrolled_option"];
    const std::string ram_shared_mmap_name = config["ram_shared_mmap_name"];

    const float eps0_min = safe_get<float>(config, "eps0_min", std::nanf(""));
    const float eps0_max = safe_get<float>(config, "eps0_max", std::nanf(""));
    const float A_min = safe_get<float>(config, "A_min", std::nanf(""));
    const float A_max = safe_get<float>(config, "A_max", std::nanf(""));

    const int N_points_eps0_range = safe_get<int>(config, "N_points_eps0_range", INT_MIN);
    const int N_points_A_range = safe_get<int>(config, "N_points_A_range", INT_MIN);

    const int host_N_steps_per_period = config["N_steps_period"];
    const int host_N_periods = config["N_periods"];
    const int N_periods_avg = config["N_periods_avg"];
    const int N_samples_noise = safe_get<int>(config, "N_samples_noise", INT_MIN);

    const float nu_phys = config["nu"];
    const float E_C = config["E_C"];

    const float eps0_target = safe_get<float>(config, "eps0_target_singlepoint", std::nanf(""));
    const float A_target = safe_get<float>(config, "A_target_singlepoint", std::nanf(""));

    float host_rho00_init = config["rho00_init"];
    float host_rho11_init = config["rho11_init"];
    float host_rho22_init = config["rho22_init"];
    float host_rho33_init = config["rho33_init"];

    const std::string path_output_csv = config["path_output_csv"];
    const std::string path_output_bin_file_gridmode = config["path_output_bin_file_gridmode"];
    const std::string path_output_bin_file_singlemode = config["path_output_bin_file_singlemode"];
    const std::string path_dynamics_single_mode_output_csv = config["path_dynamics_single_mode_output_csv"];

    const float host_delta_C = config["delta_C"];

    const float Gamma_L0_phys = config["GammaL0"];     // prefactor (GHz)
    const float Gamma_R0_phys = config["GammaR0"];     // prefactor (GHz)
    // const float host_muL = config["muL"];    // E_C
    // const float host_muR = config["muR"];    // E_C
    // const float T_K = config["T_K"];    // Kelvin

    const float Gamma_eg0_phys = config["Gamma_eg0"];    // prefactor (GHz)
    const float omega_c_norm_phys = config["omega_c"];    // high-frequency cutoff

    const float Gamma_phi0_phys = safe_get<float>(config, "Gamma_phi0", std::nanf(""));    // prefactor (GHz)
    const float host_sigma_eps = safe_get<float>(config, "sigma_eps", std::nanf(""));

    const bool single_mode_log_option = config["single_mode_log_option"];
    const std::string path_dynamics_single_mode_output_log_csv = config["path_dynamics_single_mode_output_log_csv"];
    const std::string path_dynamics_single_mode_output_log_hdf5 = config["path_dynamics_single_mode_output_log_hdf5"];
    
    const std::string threads_per_traj_opt = config["threads_per_traj_opt"];
    const std::string quasi_static_ensemble_dephasing_opt = config["quasi_static_ensemble_dephasing_opt"];

    std::cout << "Parameters extracted successfully.\n\n";
    std::cout << "==============================================\n";

    std::cout << "\n--- Listing received arguments ---\n";

    std::cout << "1.  grid_single_mode: " << grid_single_mode << "\n";
    std::cout << "3.  ouput_option: " << ouput_option << "\n";
    std::cout << "4.  unrolled_option: " << unrolled_option << "\n";
    std::cout << "5.  ram_shared_mmap_name: " << ram_shared_mmap_name << "\n";

    std::cout << "6.  eps0_min: " << eps0_min << " (E_C)\n";
    std::cout << "7.  eps0_max: " << eps0_max << " (E_C)\n";
    std::cout << "8.  A_min: " << A_min << " (E_C)\n";
    std::cout << "9.  A_max: " << A_max << " (E_C)\n";

    std::cout << "10. N_points_eps0_range: " << N_points_eps0_range << "\n";
    std::cout << "11. N_points_A_range: " << N_points_A_range << "\n";
    std::cout << "12. N_steps_period: " << host_N_steps_per_period << "\n";
    std::cout << "13. N_periods: " << host_N_periods << "\n";
    std::cout << "14. N_periods_avg: " << N_periods_avg << "\n";
    std::cout << "15. N_samples_noise: " << N_samples_noise << "\n";

    std::cout << "17. nu_phys: " << nu_phys << " (GHz)\n";
    std::cout << "17. E_C: " << E_C << " (eV)\n";
    std::cout << "18. eps0_target: " << eps0_target << " (E_C)\n";
    std::cout << "19. A_target: " << A_target << " (E_C)\n";

    std::cout << "20. rho00_init: " << host_rho00_init << "\n";
    std::cout << "21. rho11_init: " << host_rho11_init << "\n";
    std::cout << "22. rho22_init: " << host_rho22_init << "\n";
    std::cout << "23. rho33_init: " << host_rho33_init << "\n";

    std::cout << "24. path_output_csv: " << path_output_csv << "\n";
    std::cout << "25. path_output_bin_file_gridmode: " << path_output_bin_file_gridmode << "\n";
    std::cout << "26. path_output_bin_file_singlemode: " << path_output_bin_file_singlemode << "\n";
    std::cout << "27. path_dynamics_single_mode_output_csv: " << path_dynamics_single_mode_output_csv << "\n";

    std::cout << "28. delta_C: " << host_delta_C << "\n";

    std::cout << "37. Gamma_L0_phys: " << Gamma_L0_phys << " (GHz)\n";
    std::cout << "38. Gamma_R0_phys: " << Gamma_R0_phys << " (GHz)\n";
    // std::cout << "39. muL: " << host_muL << "\n";
    // std::cout << "40. muR: " << host_muR << "\n";
    // std::cout << "41. T_K: " << T_K << "\n";

    std::cout << "42. Gamma_eg0_phys: " << Gamma_eg0_phys << " (GHz)\n";
    std::cout << "43. omega_c_norm: " << omega_c_norm_phys << "\n";

    std::cout << "44. Gamma_phi0_phys: " << Gamma_phi0_phys << " (GHz)\n";
    std::cout << "45. sigma_eps: " << host_sigma_eps << " (E_C)\n";

    std::cout << "46. single_mode_log_option: " << single_mode_log_option << "\n";
    std::cout << "47. path_dynamics_single_mode_output_log_csv: " << path_dynamics_single_mode_output_log_csv << "\n";
    std::cout << "48. path_dynamics_single_mode_output_log_hdf5: " << path_dynamics_single_mode_output_log_hdf5 << "\n";
    std::cout << "49. threads_per_traj_opt: " << threads_per_traj_opt << "\n";
    std::cout << "50. quasi_static_ensemble_dephasing_otp: " << quasi_static_ensemble_dephasing_opt << "\n";

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

    if (N_periods_avg > host_N_periods) {
        std::cerr << "ERROR: N_periods_avg must be less than or equal to N_periods. Got: "
            << N_periods_avg << " > " << host_N_periods << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (((grid_single_mode == "single" || grid_single_mode == "grid_single") && (std::isnan(eps0_target) || std::isnan(A_target))) ||
        ((grid_single_mode == "grid") && !(std::isnan(eps0_target) && std::isnan(A_target)))) {
        std::cerr << "ERROR: eps0_target or A_target is NAN in single or grid_single mode, or not NAN in grid mode."
            << std::endl;
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

    if (quasi_static_ensemble_dephasing_opt == "sequential" ||
        quasi_static_ensemble_dephasing_opt == "parallel") {

        if (std::isnan(host_sigma_eps) || N_samples_noise == INT_MIN){
            std::cerr << "ERROR: sigma_eps or N_samples_noise is NAN with quasi_static_ensemble_dephasing_opt == 'sequential' or 'parallel'."
                << std::endl;
            std::exit(EXIT_FAILURE);
        }

    }
    else if (quasi_static_ensemble_dephasing_opt == "false") {

        if (!std::isnan(host_sigma_eps) || N_samples_noise != INT_MIN) {
            std::cerr << "ERROR: sigma_eps or N_samples_noise is not NAN with quasi_static_ensemble_dephasing_opt == 'false'."
                << std::endl;
            std::exit(EXIT_FAILURE);
        }

    }
    else {
        std::cerr << "ERROR: Invalid argument. Expected 'false' or 'sequential' or 'parallel'. Got: " << quasi_static_ensemble_dephasing_opt << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // -------------------------
    // Scaling and renormalization
    // -------------------------

    const float hbar_div_E_C = M_HBARf / E_C; // hbar (eV*s) / E_C (eV) = s

    const float omega_phys = 2.0f * M_PIf * nu_phys * 1e9f;
    const float host_omega = hbar_div_E_C * omega_phys; // new
    // const float host_omega = 2.0f * M_PIf * nu_phys; //old


    const float host_Gamma_L0 = hbar_div_E_C * Gamma_L0_phys * 1e9f;
    const float host_Gamma_R0 = hbar_div_E_C * Gamma_R0_phys * 1e9f;
    const float host_Gamma_eg0 = hbar_div_E_C * Gamma_eg0_phys * 1e9f;
    // const float omega_c_norm = hbar_div_E_C * omega_c_norm_phys * 1e9f;
    const float host_Gamma_phi0 = std::isnan(Gamma_phi0_phys) ? std::nanf("") : hbar_div_E_C * Gamma_phi0_phys * 1e9f;


    const float omega_c_norm = omega_c_norm_phys; // #TBD


    // compute dt so period length T has integer number of steps

    // period in physical units (s)
    const float T_phys = 1.0f / (nu_phys * 1e9f);  // seconds

    // convert to dimensionless time units
    const float T_dimless = T_phys / hbar_div_E_C;

    // per-step timestep in dimensionless units
    const float host_dt = T_dimless / float(host_N_steps_per_period); // new

    // const float T = 1 / nu_phys;
    // const float host_dt = T / float(host_N_steps_per_period); // old

    std::cout << "T_phys: " << T_phys << " (s)\n";

    std::cout << "--- Scaled parameters ---\n";

    std::cout << "T_prime: " << T_dimless << " ([1])\n";
    std::cout << "dt: " << host_dt << " ([1])\n";  
    std::cout << "omega_prime: " << host_omega << " ([1])\n";
    std::cout << "Gamma_L0_prime: " << host_Gamma_L0 << " ([1])\n";
    std::cout << "Gamma_R0_prime: " << host_Gamma_R0 << " ([1])\n";
    std::cout << "Gamma_eg0_prime: " << host_Gamma_eg0 << " ([1])\n";
    std::cout << "omega_c_norm_prime: " << omega_c_norm << " ([1])\n";  // #TBD
    std::cout << "Gamma_phi0_prime: " << host_Gamma_phi0 << " ([1])\n";
    std::cout << "sigma_eps_prime: " << host_sigma_eps << " ([1])\n";

    std::cout << "==============================================\n\n";

    // renormalization for safety

    float sum = host_rho00_init + host_rho11_init + host_rho22_init + host_rho33_init;

    if (sum != 0.0f) {
        host_rho00_init /= sum;
        host_rho11_init /= sum;
        host_rho22_init /= sum;
        host_rho33_init /= sum;
    }
    else {
        std::cerr << "ERROR: all initial rho_ii are zero. Aborting.\n";
        std::exit(EXIT_FAILURE);
    }



    //// physics constants
    //const float a = 0.1f;
    //const float m = 10.0f;
    //const float B = (a + 2.0f) * (m + 1.0f) / (a * (m - 1.0f));




    // Thermal scale
    // const float kB_mu_eV = 86.173324f; // µeV/K
    // float host_kT = kB_mu_eV * T_K;
    // if (host_kT < 1e-9f) host_kT = 1e-9f;


    /////////////////////////////////////////////////////////////////


    const float host_GammaLR0 = host_Gamma_L0 + host_Gamma_R0;
    


    const float radical = std::sqrt(1 + m * m * (B * B - 1) * host_delta_C * host_delta_C);
    const float host_epsilon_L = (B + radical) / (m * (B * B - 1));
    const float host_epsilon_R = (B - radical) / (m * (B * B - 1));

    //const float host_one_div_m = 1.f / host_m;


    const float host_beta = host_delta_C * host_delta_C / (omega_c_norm * omega_c_norm);
    const float host_Gamma_eg0_norm = host_Gamma_eg0 * expf(host_beta);

    std::cout << "debugging: omega_c_norm   : " << omega_c_norm << std::endl;
    std::cout << "debugging: host_beta   : " << host_beta << std::endl;
    std::cout << "debugging: host_Gamma_eg0   : " << host_Gamma_eg0 << std::endl;
    std::cout << "debugging: host_Gamma_eg0_norm   : " << host_Gamma_eg0_norm << std::endl;

    std::cout << "==============================================\n\n";



    gpuCheck(cudaMemcpyToSymbol(Gamma_LR0, &host_GammaLR0, sizeof(float)), "cudaMemcpyToSymbol Gamma_LR0");
    gpuCheck(cudaMemcpyToSymbol(Gamma_L0,  &host_Gamma_L0, sizeof(float)), "cudaMemcpyToSymbol Gamma_L0");
    gpuCheck(cudaMemcpyToSymbol(Gamma_R0,  &host_Gamma_R0, sizeof(float)), "cudaMemcpyToSymbol Gamma_R0");
    //gpuCheck(cudaMemcpyToSymbol(muL, &host_muL, sizeof(float)), "cudaMemcpyToSymbol muL");
    //gpuCheck(cudaMemcpyToSymbol(muR, &host_muR, sizeof(float)), "cudaMemcpyToSymbol muR");
    //gpuCheck(cudaMemcpyToSymbol(kT, &host_kT, sizeof(float)), "cudaMemcpyToSymbol kT");

    gpuCheck(cudaMemcpyToSymbol(omega, &host_omega, sizeof(float)), "cudaMemcpyToSymbol omega");


    gpuCheck(cudaMemcpyToSymbol(delta_C, &host_delta_C, sizeof(float)), "cudaMemcpyToSymbol delta_C");

    gpuCheck(cudaMemcpyToSymbol(epsilon_R, &host_epsilon_R, sizeof(float)), "cudaMemcpyToSymbol epsilon_R");
    gpuCheck(cudaMemcpyToSymbol(epsilon_L, &host_epsilon_L, sizeof(float)), "cudaMemcpyToSymbol epsilon_L");

    gpuCheck(cudaMemcpyToSymbol(beta,           &host_beta,           sizeof(float)), "cudaMemcpyToSymbol beta");
    gpuCheck(cudaMemcpyToSymbol(Gamma_eg0_norm, &host_Gamma_eg0_norm, sizeof(float)), "cudaMemcpyToSymbol Gamma_eg0_norm");


    gpuCheck(cudaMemcpyToSymbol(rho00_init, &host_rho00_init, sizeof(float)), "cudaMemcpyToSymbol rho00_init");
    gpuCheck(cudaMemcpyToSymbol(rho11_init, &host_rho11_init, sizeof(float)), "cudaMemcpyToSymbol rho11_init");
    gpuCheck(cudaMemcpyToSymbol(rho22_init, &host_rho22_init, sizeof(float)), "cudaMemcpyToSymbol rho22_init");
    gpuCheck(cudaMemcpyToSymbol(rho33_init, &host_rho33_init, sizeof(float)), "cudaMemcpyToSymbol rho33_init");

    gpuCheck(cudaMemcpyToSymbol(N_steps_per_period, &host_N_steps_per_period, sizeof(int)), "cudaMemcpyToSymbol N_steps_per_period");
    gpuCheck(cudaMemcpyToSymbol(N_periods, &host_N_periods, sizeof(int)), "cudaMemcpyToSymbol N_periods");
    gpuCheck(cudaMemcpyToSymbol(dt, &host_dt, sizeof(float)), "cudaMemcpyToSymbol dt");

    gpuCheck(cudaMemcpyToSymbol(Gamma_eg0,  &host_Gamma_eg0,  sizeof(float)), "cudaMemcpyToSymbol Gamma_eg0");
    gpuCheck(cudaMemcpyToSymbol(Gamma_phi0, &host_Gamma_phi0, sizeof(float)), "cudaMemcpyToSymbol Gamma_phi0");


    float* eps_offsets;

    if (quasi_static_ensemble_dephasing_opt == "sequential" ||
        quasi_static_ensemble_dephasing_opt == "parallel") {

        if (N_samples_noise > MAX_NOISE_SAMPLES) {
            fprintf(stderr, "Error: N_samples_noise (%d) > MAX_NOISE_SAMPLES (%d)\n",
                N_samples_noise, MAX_NOISE_SAMPLES);
            std::exit(EXIT_FAILURE);
        }

        
        cudaMallocManaged(&eps_offsets, N_samples_noise * sizeof(float));

        // Fill with noise
        std::mt19937 rng(12345);
        std::normal_distribution<float> dist(0.0f, host_sigma_eps);
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
            host_N_steps_per_period, host_N_periods, N_periods_avg,
            N_samples_noise, quasi_static_ensemble_dephasing_opt, eps_offsets,
            path_output_csv, path_output_bin_file_gridmode, ouput_option, unrolled_option,
            ram_shared_mmap_name, threads_per_traj_opt
        );
    }
    if (grid_single_mode == "single" || grid_single_mode == "grid_single") {
        run_single_mode(
            eps0_target, A_target, host_N_steps_per_period, host_N_periods,
            N_samples_noise, quasi_static_ensemble_dephasing_opt, eps_offsets,
            path_dynamics_single_mode_output_csv, path_output_bin_file_singlemode, ouput_option, unrolled_option,
            single_mode_log_option, path_dynamics_single_mode_output_log_csv,
            path_dynamics_single_mode_output_log_hdf5,
            threads_per_traj_opt
        );
    }
    
    
    if (quasi_static_ensemble_dephasing_opt == "sequential" ||
        quasi_static_ensemble_dephasing_opt == "parallel") {
        cudaFree(eps_offsets);
    }

    cudaDeviceReset();

    return 0;
}









