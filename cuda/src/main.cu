// src/main.cu

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <string>
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







// -------------------------
//   main
// -------------------------
int main(int argc, char** argv)
{

    std::cout << "\n--- Listing received arguments ---\n";
    if (argc > 1) std::cout << "1.  single_mode_flag: " << argv[1] << "\n";
    if (argc > 2) std::cout << "2.  avg_periods_ouput_option: " << argv[2] << "\n";
    if (argc > 3) std::cout << "3.  ouput_option: " << argv[3] << "\n";
    if (argc > 4) std::cout << "4.  unrolled_option: " << argv[4] << "\n";
    if (argc > 5) std::cout << "5.  ram_shared_mmap_name: " << argv[5] << "\n";

    if (argc > 6) std::cout << "6.  eps0_min: " << argv[6] << "\n";
    if (argc > 7) std::cout << "7.  eps0_max: " << argv[7] << "\n";
    if (argc > 8) std::cout << "8.  A_min: "    << argv[8] << "\n";
    if (argc > 9) std::cout << "9.  A_max: "    << argv[9] << "\n";

    if (argc > 10) std::cout << "10. N_points_eps0_range: " << argv[10] << "\n";
    if (argc > 11) std::cout << "11. N_points_A_range: " << argv[11] << "\n";
    if (argc > 12) std::cout << "12. N_steps_period: " << argv[12] << "\n";
    if (argc > 13) std::cout << "13. N_periods: " << argv[13] << "\n";
    if (argc > 13) std::cout << "14. N_periods_avg: " << argv[14] << "\n";

    if (argc > 14) std::cout << "15. alpha: " << argv[15] << "\n";
    if (argc > 15) std::cout << "16. nu: "    << argv[16] << "\n";
    if (argc > 16) std::cout << "17. eps0_target_singlepoint: " << argv[17] << "\n";
    if (argc > 17) std::cout << "18. A_target_singlepoint: "    << argv[18] << "\n";

    if (argc > 19) std::cout << "19. rho00_init: " << argv[19] << "\n";
    if (argc > 20) std::cout << "20. rho11_init: " << argv[20] << "\n";
    if (argc > 21) std::cout << "21. rho22_init: " << argv[21] << "\n";
    if (argc > 22) std::cout << "22. rho33_init: " << argv[22] << "\n";

    if (argc > 23) std::cout << "23. path_output_csv: " << argv[23] << "\n";
    if (argc > 24) std::cout << "24. path_output_bin_file: " << argv[24] << "\n";
    if (argc > 25) std::cout << "25. path_dynamics_single_mode_output_csv: " << argv[25] << "\n";

    if (argc > 26) std::cout << "26. delta_C: " << argv[26] << "\n";
    if (argc > 27) std::cout << "27. delta_L: " << argv[27] << "\n";
    if (argc > 28) std::cout << "28. delta_R: " << argv[28] << "\n";

    if (argc > 29) std::cout << "29. g_en: "   << argv[29] << "\n";
    if (argc > 30) std::cout << "30. g_phi: "  << argv[30] << "\n";
    if (argc > 31) std::cout << "31. gL_en: "  << argv[31] << "\n";
    if (argc > 32) std::cout << "32. gL_phi: " << argv[32] << "\n";
    if (argc > 33) std::cout << "33. gR_en: "  << argv[33] << "\n";
    if (argc > 34) std::cout << "34. gR_phi: " << argv[34] << "\n";

    if (argc > 35) std::cout << "35. Gamma_L0: " << argv[35] << "\n";
    if (argc > 36) std::cout << "36. Gamma_R0: " << argv[36] << "\n";
    if (argc > 37) std::cout << "37. muL: " << argv[37] << "\n";
    if (argc > 38) std::cout << "38. muR: " << argv[38] << "\n";
    if (argc > 39) std::cout << "39. T_K: " << argv[39] << "\n";

    if (argc > 40) std::cout << "40. Gamma_eg0: "  << argv[40] << "\n";
    if (argc > 41) std::cout << "41. omega_c: "    << argv[41] << "\n";

    if (argc > 42) std::cout << "42. Gamma_phi0: " << argv[42] << "\n";

    if (argc > 43) std::cout << "43. single_mode_log_option: "                    << argv[43] << "\n";
    if (argc > 44) std::cout << "44. path_dynamics_single_mode_output_log_csv: "  << argv[44] << "\n";
    if (argc > 45) std::cout << "45. path_dynamics_single_mode_output_log_hdf5: " << argv[45] << "\n";
    if (argc > 46) std::cout << "46. threads_per_traj_opt: "                      << argv[46] << "\n";

    std::cout << "==================================================" << std::endl;





    if (argc != 47) {
        /*std::cout << "Usage: ./lindblad_gpu single_mode_flag "
            << "avg_periods_ouput_option ouput_option unrolled_option ram_shared_mmap_name "
            << "eps0_min eps0_max A_min A_max N_points_eps0_range N_points_A_range "
            << "N_steps_period N_periods alpha nu eps0_target_singlepoint A_target_singlepoint "
            << "rho00_init rho11_init rho22_init rho33_init "
            << "path_output_csv path_dynamics_single_mode_output_csv "
            << "delta_C delta_L delta_R g_en g_phi gL_en gL_phi gR_en gR_phi "
            << "host_Gamma_L0 host_Gamma_R0 host_muL host_muR T_K "
            << "single_mode_log_option path_dynamics_single_mode_output_log_csv "
            << "path_dynamics_single_mode_output_log_hdf5 "
            << "trajectory_threading_option_str" << std::endl;*/


        std::cout << "ERROR: Expected 46 arguments (including program path).\n";
        std::cout << "Received " << argc << " arguments. Exiting program.\n";

        //std::cout << "Listing received arguments:\n\n";
        //for (int i = 0; i < argc; ++i) {
        //    std::cout << "argv[" << i << "]: " << argv[i] << "\n";
        //}


        std::exit(EXIT_FAILURE);
        return 1;
    }


#ifdef _WIN32
    // Allow this process to receive CTRL+C and CTRL+BREAK
    SetConsoleCtrlHandler(NULL, FALSE);
#endif


    bool single_mode;
    if (argv[1] == std::string("single")) {
        single_mode = true;
    }
    else if (argv[1] == std::string("grid")) {
        single_mode = false;
    }
    else {
        std::cerr << "ERROR: first argument must be 'single' or 'grid'. Got '" << argv[1] << "'\n";
        std::exit(EXIT_FAILURE);
        return 1;
    }

    const std::string avg_periods_ouput_option = argv[2];
    if (!(avg_periods_ouput_option == "last" /*||
          avg_periods_ouput_option == "last_2last" ||
          avg_periods_ouput_option == "whole_last_2last_3last"*/)) {
        std::cerr << "ERROR: third argument must be 'last' or 'last_2last' or 'whole_last_2last_3last'. Got '" << argv[2] << "'\n";
        std::exit(EXIT_FAILURE);
        return 1;
    }

    const std::string ouput_option = argv[3];
    if (!(ouput_option == "ssd_csv" ||
          ouput_option == "bin_file" ||
          ouput_option == "ram")) {
        std::cerr << "ERROR: fourth argument must be 'ssd_csv' or 'bin_file' or 'ram'. Got '" << argv[3] << "'\n";
        std::exit(EXIT_FAILURE);
        return 1;
    }

    const std::string unrolled_option = argv[4];
    if (!(unrolled_option == "as_arrays" ||
          unrolled_option == "unrolled")) {
        std::cerr << "ERROR: fifth argument must be 'as_arrays' or 'unrolled'. Got '" << argv[4] << "'\n";
        std::exit(EXIT_FAILURE);
        return 1;
    }

    const std::string ram_shared_mmap_name = argv[5];
    if (ouput_option == "ram" && ram_shared_mmap_name == "null") {
        std::cerr << "ERROR: sixth argument ram_shared_mmap_name = 'null', not defined.\n";
        std::exit(EXIT_FAILURE);
        return 1;
    }


    const float eps0_min = safe_stof(argv[6]);
    const float eps0_max = safe_stof(argv[7]);
    const float A_min = safe_stof(argv[8]);
    const float A_max = safe_stof(argv[9]);

    const int   N_points_eps0_range = safe_stoi(argv[10]);
    const int   N_points_A_range    = safe_stoi(argv[11]);

    if ((!single_mode && (std::isnan(eps0_min) || std::isnan(eps0_max) || std::isnan(A_min) || std::isnan(A_max)
            || N_points_eps0_range == INT_MIN || N_points_A_range == INT_MIN)) ||
        (single_mode && !(std::isnan(eps0_min) && std::isnan(eps0_max) && std::isnan(A_min) && std::isnan(A_max)
            && N_points_eps0_range == INT_MIN && N_points_A_range == INT_MIN))) {
        std::cerr << "ERROR: eps0_min, eps0_max, A_min, or A_max is NAN in grid mode, or not NAN in single mode."
            << std::endl;
        std::exit(EXIT_FAILURE);
    }

    
    const int   N_steps_period = std::stoi(argv[12]);
    const int   N_periods      = std::stoi(argv[13]);
    const int   N_periods_avg  = std::stoi(argv[14]);

    if (N_periods_avg > N_periods) {
        std::cerr << "ERROR: N_periods_avg must be less than or equal to N_periods. Got: "
            << N_periods_avg << " > " << N_periods << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const float alpha = std::stof(argv[15]);
    const float nu = std::stof(argv[16]);

    // for singlepoint mode
    const float eps0_target = safe_stof(argv[17]);
    const float A_target    = safe_stof(argv[18]);

    if ((single_mode && (std::isnan(eps0_target) || std::isnan(A_target))) || 
        (!single_mode && !(std::isnan(eps0_target) && std::isnan(A_target)))){
        std::cerr << "ERROR: eps0_target or A_target is NAN in single mode, or not NAN in grid mode."
            << std::endl;
        std::exit(EXIT_FAILURE);
        }


    // rho_0

    float rho00_init = std::stof(argv[19]);
    float rho11_init = std::stof(argv[20]);
    float rho22_init = std::stof(argv[21]);
    float rho33_init = std::stof(argv[22]);

    const std::string path_output_csv = argv[23];
    const std::string path_output_bin_file = argv[24];
    const std::string path_dynamics_single_mode_output_csv = argv[25];

    const float delta_C = std::stof(argv[26]);
    const float delta_L = std::stof(argv[27]);
    const float delta_R = std::stof(argv[28]);

    if (delta_L != 0.0f || delta_R != 0.0f) {
        std::cerr << "ERROR: delta_L and delta_R are expected to be zero. Got: " << delta_L << delta_R << std::endl;
        std::exit(EXIT_FAILURE);
        return 1;
    }


    // these Gammas are only for printing into CSV. not used elsewhere in program
    const float g_en   = std::stof(argv[29]);
    const float g_phi  = std::stof(argv[30]);
    const float gL_en  = std::stof(argv[31]);
    const float gL_phi = std::stof(argv[32]);
    const float gR_en  = std::stof(argv[33]);
    const float gR_phi = std::stof(argv[34]);

    
    const float host_Gamma_L0    = std::stof(argv[35]);    // prefactor (GHz etc.)
    const float host_Gamma_R0    = std::stof(argv[36]);    // prefactor (GHz etc.)
    const float host_muL         = std::stof(argv[37]);    // µeV
    const float host_muR         = std::stof(argv[38]);    // µeV
    const float T_K              = std::stof(argv[39]);    // Kelvin
    
    const float host_Gamma_eg0   = std::stof(argv[40]);    // prefactor (GHz etc.)
    const float omega_c_norm     = std::stof(argv[41]);    // high-frequency cutoff

    const float host_Gamma_phi0  = std::stof(argv[42]);    // prefactor (GHz etc.)
    /////// have not implemented the output of these vars.


    std::string single_mode_log_option_str = argv[43];
    bool single_mode_log_option;

    if (single_mode_log_option_str == "True") {
        single_mode_log_option = true;
    }
    else if (single_mode_log_option_str == "False") {
        single_mode_log_option = false;
    }
    else {
        std::cerr << "ERROR: Invalid argument. Expected 'True' or 'False'. Got: " << single_mode_log_option_str << std::endl;
        std::exit(EXIT_FAILURE);
        return 1;
    }

    const std::string path_dynamics_single_mode_output_log_csv = argv[44];

    const std::string path_dynamics_single_mode_output_log_hdf5 = argv[45];

    std::string threads_per_traj_opt = argv[46];


    if (!(threads_per_traj_opt == "one_thread_per_traj" ||
          threads_per_traj_opt == "thread_group_in_warp_per_traj_shuffle" ||
          threads_per_traj_opt == "thread_group_in_warp_per_traj_shmem")) {
        std::cerr << "ERROR: threads_per_traj_opt must be 'one_thread_per_traj'"
            << " or 'thread_group_in_warp_per_traj_shuffle'"
            << "or 'thread_group_in_warp_per_traj_shmem'.Got '" << threads_per_traj_opt << "'\n";
        std::exit(EXIT_FAILURE);
        return 1;
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




    
    if (single_mode) {
        run_single_mode(
            eps0_target, A_target, N_steps_period, N_periods,
            dt, nu, alpha,
            rho00_init, rho11_init, rho22_init, rho33_init,
            delta_C, delta_L, delta_R, g_en, g_phi, gL_en, gL_phi, gR_en, gR_phi,
            path_dynamics_single_mode_output_csv, path_output_bin_file, ouput_option, unrolled_option,
            single_mode_log_option, path_dynamics_single_mode_output_log_csv,
            path_dynamics_single_mode_output_log_hdf5,
            threads_per_traj_opt,
            host_Gamma_L0, host_Gamma_R0, host_Gamma_eg0, omega_c_norm, host_Gamma_phi0
        );
    }
    else {
        run_grid_mode(
            eps0_min, eps0_max, A_min, A_max,
            N_points_eps0_range, N_points_A_range,
            N_steps_period, N_periods, N_periods_avg, dt, nu, alpha,
            path_output_csv, path_output_bin_file, avg_periods_ouput_option, ouput_option, unrolled_option,
            ram_shared_mmap_name, threads_per_traj_opt,
            rho00_init, rho11_init, rho22_init, rho33_init,
            delta_C, delta_L, delta_R, g_en, g_phi, gL_en, gL_phi, gR_en, gR_phi,
            host_Gamma_L0, host_Gamma_R0, host_Gamma_eg0, omega_c_norm, host_Gamma_phi0
        );
    }

    


    cudaDeviceReset();

    return 0;
}









