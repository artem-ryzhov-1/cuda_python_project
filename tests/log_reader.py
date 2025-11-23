import numpy as np
import pandas as pd
import struct
from pathlib import Path

def read_log_binary(filename):
    """
    Read binary log file written by write_log_entries_to_binary
    
    Returns:
        dict: Dictionary containing:
            - 'attributes': dict of scalar attributes
            - 'data': structured numpy array with all log entries
    """
    
    with open(filename, 'rb') as f:
        # Read header
        num_entries = struct.unpack('i', f.read(4))[0]
        num_scalar_attrs = struct.unpack('i', f.read(4))[0]
        
        # Read scalar attributes
        attributes = {}
        attr_names = [
            'one_div_m', 'B', 'pi_alpha', 'pi_alpha_delta_C', 'delta_C',
            'Gamma_L0', 'Gamma_R0', 'Gamma_eg0', 'Gamma_eg0_norm', 
            'beta', 'Gamma_phi0', 'epsilon_L', 'epsilon_R'
        ]
        
        for name in attr_names:
            attributes[name] = struct.unpack('f', f.read(4))[0]
        
        # Define the dtype for LogEntry structure
        # This must match the C++ struct exactly
        log_dtype = np.dtype([
            ('t_idx_step', np.int32),
            ('t_idx_substep', np.int32),
            ('t_step', np.float32),
            ('t_substep', np.float32),
            ('substep_num', np.int32),
            ('eps_t_substep', np.float32),
            ('gp_sqr', np.float32),
            ('gm_sqr', np.float32),
            ('gp_gm', np.float32),
            ('Gamma_lprm', np.float32),
            ('Gamma_lmrp', np.float32),
            # ('Gamma_10', np.float32),
            # ('Gamma_20', np.float32),
            # ('Gamma_30', np.float32),
            # ('Gamma_21', np.float32),
            # ('Gamma_31', np.float32),
            # ('Gamma_32', np.float32),
            # ('Gamma_L0_log', np.float32),
            # ('Gamma_R0_log', np.float32),
            ('Gamma_eg', np.float32),
            ('debug_eps_t_substep', np.float32),
            ('debug_delta_C', np.float32),
            ('debug_radical', np.float32),
            ('debig_radical_div_delta_C', np.float32),
            ('debug_Gamma_eg0_norm', np.float32),
            ('debug_beta', np.float32),
            ('debug_Gamma_eg_loc', np.float32),
            ('Gamma_phi', np.float32),
            # ('interval_diagonalizer', np.int32),
            ('interval_dissipator', np.int32),
            # ('W_L_1_0', np.float32),
            # ('W_R_1_0', np.float32),
            # ('W_L_2_0', np.float32),
            # ('W_R_2_0', np.float32),
            # ('W_L_3_0', np.float32),
            # ('W_R_3_0', np.float32),
            # ('W_L_2_1', np.float32),
            # ('W_R_2_1', np.float32),
            # ('W_L_3_1', np.float32),
            # ('W_R_3_1', np.float32),
            # ('W_L_3_2', np.float32),
            # ('W_R_3_2', np.float32),
            # ('U00', np.float32),
            # ('U01', np.float32),
            # ('U02', np.float32),
            # ('U03', np.float32),
            # ('U10', np.float32),
            # ('U11', np.float32),
            # ('U12', np.float32),
            # ('U13', np.float32),
            # ('U20', np.float32),
            # ('U21', np.float32),
            # ('U22', np.float32),
            # ('U23', np.float32),
            # ('U30', np.float32),
            # ('U31', np.float32),
            # ('U32', np.float32),
            # ('U33', np.float32),
            # ('E0', np.float32),
            # ('E1', np.float32),
            # ('E2', np.float32),
            # ('E3', np.float32),
            # ('rule_col_0', np.int32),
            # ('rule_col_1', np.int32),
            # ('rule_col_2', np.int32),
            # ('rule_col_3', np.int32),
            ('rho_in_0', np.float32),
            ('rho_in_1', np.float32),
            ('rho_in_2', np.float32),
            ('rho_in_3', np.float32),
            ('rho_in_4', np.float32),
            ('rho_in_5', np.float32),
            ('rho_in_6', np.float32),
            ('rho_in_7', np.float32),
            ('rho_in_8', np.float32),
            ('rho_in_9', np.float32),
            ('rho_in_10', np.float32),
            ('rho_in_11', np.float32),
            ('rho_in_12', np.float32),
            ('rho_in_13', np.float32),
            ('rho_in_14', np.float32),
            ('rho_in_15', np.float32),
            ('drho_out_comm_0', np.float32),
            ('drho_out_comm_1', np.float32),
            ('drho_out_comm_2', np.float32),
            ('drho_out_comm_3', np.float32),
            ('drho_out_comm_4', np.float32),
            ('drho_out_comm_5', np.float32),
            ('drho_out_comm_6', np.float32),
            ('drho_out_comm_7', np.float32),
            ('drho_out_comm_8', np.float32),
            ('drho_out_comm_9', np.float32),
            ('drho_out_comm_10', np.float32),
            ('drho_out_comm_11', np.float32),
            ('drho_out_comm_12', np.float32),
            ('drho_out_comm_13', np.float32),
            ('drho_out_comm_14', np.float32),
            ('drho_out_comm_15', np.float32),
            ('drho_out_D_dl_r00', np.float32),
            ('drho_out_D_dl_r11', np.float32),
            ('drho_out_D_dl_r22', np.float32),
            ('drho_out_D_dl_r33', np.float32),
            ('drho_out_D_dl_r01', np.float32),
            ('drho_out_D_dl_i01', np.float32),
            ('drho_out_D_dl_r02', np.float32),
            ('drho_out_D_dl_i02', np.float32),
            ('drho_out_D_dl_r03', np.float32),
            ('drho_out_D_dl_i03', np.float32),
            ('drho_out_D_dl_r12', np.float32),
            ('drho_out_D_dl_i12', np.float32),
            ('drho_out_D_dl_r13', np.float32),
            ('drho_out_D_dl_i13', np.float32),
            ('drho_out_D_dl_r23', np.float32),
            ('drho_out_D_dl_i23', np.float32),
            ('drho_out_D_eg_r11', np.float32),
            ('drho_out_D_eg_r22', np.float32),
            ('drho_out_D_eg_r01', np.float32),
            ('drho_out_D_eg_i01', np.float32),
            ('drho_out_D_eg_r02', np.float32),
            ('drho_out_D_eg_i02', np.float32),
            ('drho_out_D_eg_r12', np.float32),
            ('drho_out_D_eg_i12', np.float32),
            ('drho_out_D_eg_r13', np.float32),
            ('drho_out_D_eg_i13', np.float32),
            ('drho_out_D_eg_r23', np.float32),
            ('drho_out_D_eg_i23', np.float32),
            ('drho_out_D_phi_r11', np.float32),
            ('drho_out_D_phi_r22', np.float32),
            ('drho_out_D_phi_r01', np.float32),
            ('drho_out_D_phi_i01', np.float32),
            ('drho_out_D_phi_r02', np.float32),
            ('drho_out_D_phi_i02', np.float32),
            ('drho_out_D_phi_r12', np.float32),
            ('drho_out_D_phi_i12', np.float32),
            ('drho_out_D_phi_r13', np.float32),
            ('drho_out_D_phi_i13', np.float32),
            ('drho_out_D_phi_r23', np.float32),
            ('drho_out_D_phi_i23', np.float32),
            # ('drho_out_D_egphi_r11', np.float32),
            # ('drho_out_D_egphi_r22', np.float32),
            # ('drho_out_D_egphi_r01', np.float32),
            # ('drho_out_D_egphi_i01', np.float32),
            # ('drho_out_D_egphi_r02', np.float32),
            # ('drho_out_D_egphi_i02', np.float32),
            # ('drho_out_D_egphi_r12', np.float32),
            # ('drho_out_D_egphi_i12', np.float32),
            # ('drho_out_D_egphi_r13', np.float32),
            # ('drho_out_D_egphi_i13', np.float32),
            # ('drho_out_D_egphi_r23', np.float32),
            # ('drho_out_D_egphi_i23', np.float32),
            ('drho_out_total_0', np.float32),
            ('drho_out_total_1', np.float32),
            ('drho_out_total_2', np.float32),
            ('drho_out_total_3', np.float32),
            ('drho_out_total_4', np.float32),
            ('drho_out_total_5', np.float32),
            ('drho_out_total_6', np.float32),
            ('drho_out_total_7', np.float32),
            ('drho_out_total_8', np.float32),
            ('drho_out_total_9', np.float32),
            ('drho_out_total_10', np.float32),
            ('drho_out_total_11', np.float32),
            ('drho_out_total_12', np.float32),
            ('drho_out_total_13', np.float32),
            ('drho_out_total_14', np.float32),
            ('drho_out_total_15', np.float32),
        ])
        
        # Read all log entries at once
        data = np.fromfile(f, dtype=log_dtype, count=num_entries)
    
    print(f"Loaded {num_entries} log entries from {filename}")
    print(f"Attributes: {list(attributes.keys())}")
    
    return {
        'attributes': attributes,
        'data': data
    }


# Example usage:
#if __name__ == "__main__":
    
repo_path = Path(__file__).resolve().parent.parent

log_file = repo_path / "cuda" / "output" / "rho_dynamics_single_mode_log_out.bin"

# Read the binary file
result = read_log_binary(log_file)

# Access attributes
print("\nScalar Attributes:")
for key, value in result['attributes'].items():
    print(f"  {key}: {value}")

# Access log data
data = result['data']
print(f"\nLog data shape: {data.shape}")
print(f"Available fields: {data.dtype.names}")

# Example: access specific fields
print(f"\nFirst entry t_step: {data['t_step'][0]}")
print(f"First entry eps_t_substep: {data['eps_t_substep'][0]}")


df = pd.DataFrame(data)








'''
# Example: plot something
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(data['t_substep'], data['eps_t_substep'])
plt.xlabel('Time')
plt.ylabel('eps_t_substep')
plt.title('Driving')

plt.subplot(1, 3, 2)
plt.plot(data['t_substep'], data['rho_in_0'], label='rho00')
plt.plot(data['t_substep'], data['rho_in_5'], label='rho11')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Populations')

plt.subplot(1, 3, 3)
plt.plot(data['t_substep'], data['Gamma_eg'])
plt.xlabel('Time')
plt.ylabel('Gamma_eg')
plt.title('Energy Damping')

plt.tight_layout()
plt.show()
'''