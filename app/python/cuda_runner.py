########################################
# app/python/cuda_runner.py
########################################

from pathlib import Path
import json
from dataclasses import asdict

import time
import subprocess
import threading
import signal

from app.python.config import SimulationConfig


def run_gpu_lindblad_program(c: SimulationConfig):
    
    """
    Runs the CUDA Lindblad executable with parameters from SimulationConfig `c`.
    
    Uses JSON config file instead of command-line arguments.
    Streams output live to console, supports graceful termination via ENTER or q.
    
    Raises subprocess.CalledProcessError on non-zero exit code, including full output.
    
    Returns:
        int: subprocess return code.
    """
        
    if c.platform_type == 'local_windows':
        import msvcrt
    
    print("\n==============================================")
    
    # Convert dataclass to dictionary
    config = asdict(c)
    
    # Add version field
    config["version"] = "1.0"
    
    # Remove fields that shouldn't go to CUDA program
    fields_to_remove = [
        "cuda_cwd",
        "cuda_program_path", 
        "platform_type",
        "path_dynamics_grid_mode_output_hdf5_after_ram"  # Not used in CUDA
    ]
    
    for field in fields_to_remove:
        config.pop(field, None)
    
    # Convert Path objects to strings (if any remain)
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    
    # Write config to JSON file
    config_path = Path(c.cuda_cwd).parent/ "input" / "run_config.json" #TBD to improve
    print(f"Writing configuration to: {config_path}")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved with {len(config)} parameters")
    print("==============================================")
    
    # Build argument list (now just program path + config file)
    args = [
        str(c.cuda_program_path),
        str(config_path)
    ]
    
    print("Args being passed to CUDA program:", args)
    print(f"Total arguments: {len(args)}")
    print("==============================================")
    print("CUDA program run started. Console output:")
    print()
    
  
    # Launch CUDA program with output capture
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=c.cuda_cwd,
        bufsize=1,  # line-buffered
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if c.platform_type == 'local_windows' else 0  # Allow sending CTRL_C_EVENT
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
        
    if c.platform_type in ['local_windows', 'local_linux', 'local_wsl2']:  
        # Optional: handle user input for stopping
        try:
            print("CUDA program running. Press 'q' to stop...")
            while True:
                if proc.poll() is not None:
                    break
                
                if c.platform_type == 'local_windows':
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
            if c.platform_type == 'local_windows':
                proc.send_signal(signal.CTRL_C_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
    
    elif c.platform_type == "colab_linux":
        print("CUDA program running (non-interactive mode)...")

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








