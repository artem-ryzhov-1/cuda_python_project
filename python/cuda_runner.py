
from pathlib import Path
import platform

import time
import subprocess
import threading
import signal

from config import SimulationConfig


def run_gpu_lindblad_program(c: SimulationConfig):
    
    """
    Runs the CUDA Lindblad executable with parameters from SimulationConfig `c`.
    
    Streams output live to console, supports graceful termination via ENTER or q.
    
    Raises subprocess.CalledProcessError on non-zero exit code, including full output.
    
    Returns:
        int: subprocess return code.
    """
    
    if c.environment == 'Windows':
        import msvcrt
    
    print("\n==============================================")
    
    # Build argument list (convert numbers to strings)
    args = [
        str(c.cuda_program_path),
        str(c.single_point_mode_flag),
        str(c.avg_periods_ouput_option),
        str(c.ouput_option),
        str(c.unrolled_option),
        str(c.ram_shared_mmap_name),
        str(c.eps0_min), str(c.eps0_max),
        str(c.A_min), str(c.A_max),
        str(c.N_points_eps0_range),
        str(c.N_points_A_range),
        str(c.N_steps_period),
        str(c.N_periods),
        str(c.N_periods_avg),
        str(c.alpha),
        str(c.nu),
        str(c.eps0_target_singlepoint),
        str(c.A_target_singlepoint),
        str(c.rho00_init),
        str(c.rho11_init),
        str(c.rho22_init),
        str(c.rho33_init),
        str(c.path_output_csv),
        str(c.path_output_bin_file),
        str(c.path_dynamics_single_mode_output_csv),
        str(c.delta_C),
        str(c.delta_L),
        str(c.delta_R),
        str(c.g_en),
        str(c.g_phi),
        str(c.gL_en),
        str(c.gL_phi),
        str(c.gR_en),
        str(c.gR_phi),
        
        str(c.GammaL0),
        str(c.GammaR0),
        str(c.muL),
        str(c.muR),
        str(c.T_K),
        str(c.Gamma_eg0),
        str(c.omega_c),
        str(c.Gamma_phi0),
        
        str(c.single_mode_log_option),
        str(c.path_dynamics_single_mode_output_log_csv),
        str(c.path_dynamics_single_mode_output_log_hdf5),
        str(c.threads_per_traj_opt)
        
    ]
    
    args_str = ' '.join(args)    
    
    print("Args being passed to CUDA program:", args)
    print()
    print(args_str)
    print()
    print(f"Total arguments (including program path): {len(args)}")
    print()
    print("==============================================")
    print("CUDA program run started. Console output:")
    print()
    
    
    if c.environment == "Windows":
        cwd = Path(r"C:\Users\E-Store\Documents\projects\repos\lindblad_cuda3\x64\Release")
    else:
        script_dir = Path(__file__).resolve().parent
        cwd = script_dir.parent / "cuda"
    
    # Launch CUDA program with output capture
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        bufsize=1,  # line-buffered
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if c.environment == "Windows" else 0  # Allow sending CTRL_C_EVENT
    )
    
    output_lines = []

    
    # previous
    def output_reader():
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            print(line, end='', flush=True)
            output_lines.append(line)



    
    
    output_thread = threading.Thread(target=output_reader)
    output_thread.daemon = True
    output_thread.start()
     

    # Optional: handle user input for stopping
    try:
        print("CUDA program running. Press 'q' to stop...")
        while True:
            if proc.poll() is not None:
                break
            
            if platform.system() == 'Windows':
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    #print(f"Key pressed: {key}")  # Debug print
                    if key.lower() == b'q':
                        print("\n'q' pressed. Stopping CUDA program. Sending CTRL_C_EVENT...")
                        proc.send_signal(signal.CTRL_C_EVENT)
                        break
            else:
                key = input()  # Linux/WSL2/Colab
                if key.lower() == 'q':
                    print("\n'q' pressed. Stopping CUDA program. Sending SIGINT...")
                    proc.send_signal(signal.SIGINT)
                    break

            time.sleep(0.001)  # sleep to reduce CPU hogging
            
            
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received in Python. Stopping CUDA program. Sending CTRL_C_EVENT...")
        if c.environment == "Windows":
            proc.send_signal(signal.CTRL_C_EVENT)
        else:
            proc.send_signal(signal.SIGINT)


    # Finalize
    proc.wait()
    output_thread.join()
    proc.stdout.close()  
    
    
    # Raise an exception if the program failed
    if proc.returncode == 0:
        print("CUDA program run finished successfully. Console output closed.")
    elif proc.returncode == 0xC000013A:  # CTRL+C
        print("CUDA program was interrupted by CTRL+C or signal. Exiting gracefully.")
    else:
        print(f"CUDA program exited with error code {proc.returncode}")
        full_output = "".join(output_lines)
        raise subprocess.CalledProcessError(proc.returncode, args, output=full_output)

    
    print("==============================================")
    print()
    
    return proc.returncode








