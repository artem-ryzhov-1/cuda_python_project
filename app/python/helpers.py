########################################
# app/python/helpers.py
########################################

def run_build_script(repo_path, platform_type):

    import subprocess
    import sys

    # Full paths to the build/setup scripts
    setup_colab_script = repo_path / "app" / 'build_scripts' / 'setup_colab.sh'
    linux_build_script = repo_path / "app" / 'build_scripts' / 'build_local_linux.sh'
    windows_build_script = repo_path / "app" / 'build_scripts' / 'build_local_windows.bat'

    # Password for sudo (for WSL2 or Local Linux)
    sudo_password = '...'  # Replace with your actual password

    try:
        if platform_type in ['local_wsl2', 'local_linux']:
            # For WSL2 or Local Linux, use sudo to run the build script
            print("Building for Linux/WSL2...")
            result = subprocess.run(
                f'echo {sudo_password} | sudo -S bash {linux_build_script}',
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:", result.stderr)
            print("\n✓ Build script executed successfully for local Linux/WSL2 environment.")

        elif platform_type == 'colab_linux':
            # For Colab, just run the setup script without sudo
            print("Building for Google Colab...")
            result = subprocess.run(
                f'bash {setup_colab_script}',
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:", result.stderr)
            print("\n✓ Build script executed successfully in Colab.")

        elif platform_type == 'local_windows':
            # For Windows, run the batch script
            print("Building for Windows...")
            if not windows_build_script.exists():
                print(f"Error: Build script not found at {windows_build_script}")
                sys.exit(1)

            # Run the batch script from its directory to ensure relative paths work
            result = subprocess.run(
                [str(windows_build_script)],
                cwd=windows_build_script.parent,
                check=True,
                capture_output=True,
                text=True,
                shell=True  # Needed for batch files on Windows
            )
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:", result.stderr)
            print("\n✓ Build script executed successfully for Windows environment.")

        else:
            print(f"Error: Unknown platform_type '{platform_type}'")
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: Required file or command not found: {e}")
        sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"Error: Build script failed with exit code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def run_test_cuda_program(repo_path):
    # ===============================================
    # RUN PRECOMPILED CUDA PROGRAM WITH JSON CONFIG
    # ===============================================

    import os, json
    import subprocess

    # -------------------------------
    # 1. PROJECT ROOT
    # -------------------------------
    print("Repo path:", repo_path)

    # Ensure output folder exists
    output_dir = repo_path / "app" / "cuda" / "output"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------
    # 2. CREATE JSON CONFIG
    # -------------------------------
    config = {
        "grid_single_mode": "grid_single",
        "ouput_option": "bin_file",
        "unrolled_option": "unrolled",
        "ram_shared_mmap_name": "MySimSharedMemory",
        "single_mode_log_option": False,
        "threads_per_traj_opt": "one_thread_per_traj",

        "eps0_target_singlepoint": -0.00018929930075147695,
        "A_target_singlepoint": 0.005371448934150004,
        "eps0_min": -0.006,
        "eps0_max": 0.006,
        "A_min": 0.0,
        "A_max": 0.01,
        "N_points_eps0_range": 774,
        "N_points_A_range": 645,
        "N_steps_period": 1000,
        "N_periods": 10,
        "N_periods_avg": 1,
        "N_samples_noise": None,

        "delta_C": 0.0001208,
        "nu": 21.0,

        "rho00_init": 0.25,
        "rho11_init": 0.25,
        "rho22_init": 0.25,
        "rho33_init": 0.25,

        # Use repo_path instead of absolute paths
        "path_output_csv": (output_dir / "rho_avg_out.csv").as_posix(),
        "path_output_bin_file_gridmode": (output_dir / "rho_avg_out.bin").as_posix(),
        "path_output_bin_file_singlemode": (output_dir / "rho_dynamics_single_mode_out.bin").as_posix(),
        "path_dynamics_single_mode_output_csv": (output_dir / "rho_dynamics_single_mode_out.csv").as_posix(),
        "path_dynamics_single_mode_output_log_csv": (output_dir / "rho_dynamics_single_mode_log_out.csv").as_posix(),
        "path_dynamics_single_mode_output_log_bin": (output_dir / "rho_dynamics_single_mode_log_out.bin").as_posix(),

        "GammaL0": 420.0,
        "GammaR0": 68.0,
        #"muL": 0,
        #"muR": 0,
        #"T_K": 0,
        "Gamma_eg0": 10.0,
        "omega_c": 0.0015731484686413405,
        "Gamma_phi0": 36.6,

        "quasi_static_ensemble_dephasing_opt": "false",
        "sigma_eps": None,

        "version": "2.0"
    }

    # Save JSON
    config_path = repo_path / "app" / "cuda" / "input" / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print("Config written to:", config_path)

    # -------------------------------
    # 3. RUN THE PROGRAM
    # -------------------------------
    binary_path = repo_path / "app" / "cuda" / "bin" / "lindblad_gpu"
    cmd = f"{binary_path.as_posix()} {config_path.as_posix()}"
    print("Executing:", cmd)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    print("\n=== STDOUT ===")
    print(result.stdout)
    print("\n=== STDERR ===")
    print(result.stderr)
    print("\n=== program executed ===")

    return result


def install_dependencies(repo_path, platform_type):
    import subprocess
    if platform_type == 'colab_linux':
        print("Installing Python dependencies in Colab...")
        requirements_path = repo_path / "app" / 'requirements.txt'
        subprocess.run(['pip', 'install', '-r', str(requirements_path)], check=True)
        print("Dependencies installed")

def verify_required_files(repo_path):
    # =============================================================================
    # Verify required files exist
    # =============================================================================
    required_files = [
        'app/python/app_class_dynamics_plot.py',
        'app/python/app_class_interactive_interferogram_dynamics.py',
        'app/python/app_class_interferogram_plot.py',
        'app/python/app_class_simulation_parameters.py',
        'app/python/simulation.py',
        'app/python/config.py',
        'app/python/cuda_runner.py',
        'app/python/file_io.py',
        'app/python/simulation.py',
        'app/python/helpers.py'
    ]

    missing_files = []
    for file in required_files:
        if not (repo_path / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"⚠️ Missing files: {missing_files}")
        print("Please upload these files to your repository")
    else:
        print("✅ All required files present")

def detect_cupy_cudf():
    import os

    # GPU detection and configuration
    CUPY_AVAILABLE = False
    CUDF_AVAILABLE = False

    try:
        import cupy as cp
        CUPY_AVAILABLE = cp.cuda.is_available()
        if CUPY_AVAILABLE:
            print(f"[GPU] CuPy detected - {cp.cuda.runtime.getDeviceCount()} CUDA device(s) found")
        else:
            print("[GPU] CuPy not available - Datashader will use CPU")
    except ImportError:
        print("[GPU] CuPy not available - Datashader will use CPU")
        #print("[GPU] CuPy not available - install cupy for GPU acceleration")
    except Exception as e:
        print(f"[GPU] CuPy error: {e}")

    # Check for cuDF (required for Datashader GPU acceleration)
    try:
        import cudf
        CUDF_AVAILABLE = True
        print("[GPU] cuDF detected - Datashader GPU acceleration available")
        os.environ['DATASHADER_USE_CUPY'] = '1'
    except ImportError:
        print("[GPU] cuDF not available - Datashader will use CPU")
        #print("[GPU] Install with: conda install -c rapidsai -c conda-forge cudf")
    except Exception as e:
        print(f"[GPU] cuDF error: {e}")

    CUPY_CUDF_AVAILABLE = CUPY_AVAILABLE and CUDF_AVAILABLE
    
    return CUPY_CUDF_AVAILABLE

# Convert to GPU arrays if GPU enabled
#if self.gpu_enabled and CUPY_AVAILABLE:
#    eps0_array = cp.asarray(self.eps0_grid)
#    A_array = cp.asarray(self.A_grid)
#    data_array = cp.asarray(data)
#else:
#    eps0_array = self.eps0_grid
#    A_array = self.A_grid
#    data_array = data

def modify_render_mode(render_mode, CUPY_CUDF_AVAILABLE):
    # Auto-fallback if GPU requested but not available
    if render_mode in ['raster_static_gpu', 'raster_dynamic_gpu'] and not cupy_available:
        print("[GPU] GPU mode requested but CuPy not available - falling back to CPU version")
        render_mode_modified = render_mode.replace('_gpu', '')
        print(f"Render mode: {render_mode_modified}")
    else:
        render_mode_modified = render_mode
    return render_mode_modified

def launch_app_colab(dashboard):
    import panel as pn
    from IPython.display import display, HTML, IFrame
    from google.colab.output import eval_js

    print("\n" + "="*70)
    print("🚀 LAUNCHING IN COLAB MODE")
    print("="*70)
    
    import threading
    import time
    import socket
    
    def find_free_port(start=5006):
        """Find an available port"""
        for port in range(start, start + 20):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('0.0.0.0', port))
                sock.close()
                return port
            except OSError:
                continue
        raise RuntimeError("No free ports found")
    
    PORT = find_free_port()
    print(f"✅ Using port: {PORT}")
    
    # Start server in background thread
    server_started = threading.Event()
    
    def start_server():
        try:
            pn.serve(
                {'/': dashboard},
                port=PORT,
                address='127.0.0.1',
                show=False,
                allow_websocket_origin=['*'],
                threaded=True,
                verbose=True
            )
            server_started.set()
        except Exception as e:
            print(f"❌ Server error: {e}")
    
    print("🔄 Starting Bokeh server...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    if not server_started.wait(timeout=15):
        print("⚠️ Server startup timeout - trying anyway...")
    
    time.sleep(3)  # Give server extra time to initialize
    
    # Get Colab proxy URL
    try:
        colab_url = eval_js(f"google.colab.kernel.proxyPort({PORT})")
        
        print("\n" + "="*70)
        print("✅ SERVER READY!")
        print("="*70)
        print(f"📍 URL: {colab_url}")
        
        # Display clickable link and iframe
        display(HTML(f'''
            <div style="background: #f0f8ff; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h2 style="color: #1a73e8; margin-top: 0;">
                    🚀 <a href="{colab_url}" target="_blank" style="color: #1a73e8; text-decoration: none;">
                        Click here to open Interactive Interferogram Dashboard
                    </a>
                </h2>
                <ul style="line-height: 1.8;">
                    <li><strong>✅ Click on interferogram to select coordinates</strong></li>
                    <li><strong>✅ Blue crosshairs show selected point</strong></li>
                    <li><strong>✅ Dynamics plot updates automatically</strong></li>
                    <li><strong>💡 Enable "Show Dynamics Plot" first</strong></li>
                    <li><strong>🎛️ Use sliders to adjust parameters</strong></li>
                    <li><strong>🔄 Click "Regenerate" buttons to recompute</strong></li>
                </ul>
                <p style="color: #666; font-size: 14px; margin-bottom: 0;">
                    ⚠️ If the app doesn't load, click the link above to open in a new tab
                </p>
            </div>
        '''))
        
        # Embed in iframe (may not work in all browsers)
        display(IFrame(src=colab_url, width=1400, height=1200))
        
        print("\n💡 USAGE INSTRUCTIONS:")
        print("   1. Enable 'Show Dynamics Plot' toggle")
        print("   2. Click anywhere on the interferogram colormap")
        print("   3. Blue cross will appear and dynamics will compute")
        print("   4. Adjust parameters with sliders")
        print("   5. Use regenerate buttons to update plots")
        print("\n🛑 To stop server: Interrupt this cell (Runtime → Interrupt)\n")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n✅ Server stopped")
            
    except Exception as e:
        print(f"\n❌ Failed to get Colab URL: {e}")
        print(f"💡 Try accessing manually: http://localhost:{PORT}")





