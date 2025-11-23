// src/log_writer.cuh

#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include "constants.cuh"

// ===========================
// Binary Logger Function
// ===========================
__host__ inline void write_log_entries_to_binary(
    const std::vector<LogEntry>& h_log_buffer,
    const std::string& filename,

    const float host_pi_alpha,
    const float host_pi_alpha_delta_C,
    const float host_delta_C,
    const float host_Gamma_L0,
    const float host_Gamma_R0,
    const float host_Gamma_eg0,
    const float host_Gamma_eg0_norm,
    const float host_beta,
    const float host_Gamma_phi0,
    const float host_epsilon_L,
    const float host_epsilon_R
) {
    try {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        // Write header with metadata
        // Format: [num_entries(int)] [num_scalar_attrs(int)] [scalar attributes...] [log entries...]
        
        int num_entries = static_cast<int>(h_log_buffer.size());
        file.write(reinterpret_cast<const char*>(&num_entries), sizeof(int));

        // Write number of scalar attributes (13 in this case)
        int num_scalar_attrs = 13;
        file.write(reinterpret_cast<const char*>(&num_scalar_attrs), sizeof(int));

        // Write scalar attributes
        file.write(reinterpret_cast<const char*>(&one_div_m), sizeof(float));
        file.write(reinterpret_cast<const char*>(&B), sizeof(float));
        file.write(reinterpret_cast<const char*>(&host_pi_alpha), sizeof(float));
        file.write(reinterpret_cast<const char*>(&host_pi_alpha_delta_C), sizeof(float));
        file.write(reinterpret_cast<const char*>(&host_delta_C), sizeof(float));
        file.write(reinterpret_cast<const char*>(&host_Gamma_L0), sizeof(float));
        file.write(reinterpret_cast<const char*>(&host_Gamma_R0), sizeof(float));
        file.write(reinterpret_cast<const char*>(&host_Gamma_eg0), sizeof(float));
        file.write(reinterpret_cast<const char*>(&host_Gamma_eg0_norm), sizeof(float));
        file.write(reinterpret_cast<const char*>(&host_beta), sizeof(float));
        file.write(reinterpret_cast<const char*>(&host_Gamma_phi0), sizeof(float));
        file.write(reinterpret_cast<const char*>(&host_epsilon_L), sizeof(float));
        file.write(reinterpret_cast<const char*>(&host_epsilon_R), sizeof(float));

        // Write log entries as raw binary data
        file.write(reinterpret_cast<const char*>(h_log_buffer.data()), 
                   num_entries * sizeof(LogEntry));

        file.close();

        std::cout << "Log results saved to binary file: " << filename << std::endl;
        std::cout << "Total entries written: " << num_entries << std::endl;
        std::cout << "File size: " << (num_entries * sizeof(LogEntry) + 2 * sizeof(int) + 13 * sizeof(float)) / 1048576.0 
                  << " MB" << std::endl;

    }
    catch (std::exception& e) {
        std::cerr << "Binary write error: " << e.what() << std::endl;
    }
}