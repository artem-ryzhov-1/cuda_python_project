########################################
# python/app_local.py
########################################

from pathlib import Path
import sys
import os

# Determine the platform type (local or colab)
if 'COLAB_GPU' in os.environ:  # Check if running in Google Colab
    platform_type = 'colab_linux'
else:
    if sys.platform.startswith('linux'):
        # Check for WSL2
        # Check if running under WSL (by looking for /mnt/c/)
        if Path('/mnt/c/').exists():
            platform_type = 'local_wsl2'
        else:
            platform_type = 'local_linux'
    elif sys.platform.startswith('win'):
        platform_type = 'local_windows'
    elif sys.platform.startswith('darwin'):
        raise RuntimeError("CUDA is not natively supported on macOS. Execution is not implemented.")
    else:
        raise RuntimeError("Unsupported platform")

# Set the repo path based on platform
if platform_type == 'colab_linux':
    repo_path = Path('/content/cuda_python_project')  # Colab default path
elif platform_type in ['local_linux', 'local_wsl2']:
    repo_path = Path('~/cuda_python_project').expanduser()  # Shared path for WSL2 and Linux
elif platform_type == 'local_windows':
    repo_path = Path(os.path.expandvars(r'%USERPROFILE%\cuda_python_project'))  # Windows local path


print(f"Platform: {platform_type}")
print(f"Repo directory: {repo_path}")


import panel as pn
import holoviews as hv
from app_interferogram_dynamics_class import InteractiveInterferogramDynamics


# GPU detection and configuration
CUPY_AVAILABLE = False
CUDF_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
    if CUPY_AVAILABLE:
        print(f"[GPU] CuPy detected - {cp.cuda.runtime.getDeviceCount()} CUDA device(s) found")
    else:
        print("[GPU] CUDA not available")
except ImportError:
    print("[GPU] CuPy not available - install cupy for GPU acceleration")
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
    print("[GPU] Install with: conda install -c rapidsai -c conda-forge cudf")
except Exception as e:
    print(f"[GPU] cuDF error: {e}")

CUPY_CUDF_AVAILABLE = CUPY_AVAILABLE and CUDF_AVAILABLE

        

render_mode = 'raster_dynamic' # ['vector', 'raster_static', 'raster_static_gpu', 'raster_dynamic', 'raster_dynamic_gpu']


# Auto-fallback if GPU requested but not available
if render_mode in ['raster_static_gpu', 'raster_dynamic_gpu'] and not CUPY_AVAILABLE:
    print("[GPU] GPU mode requested but CuPy not available - falling back to CPU version")
    render_mode = render_mode.replace('_gpu', '')


# Enable Panel extension - CRITICAL: Must be called before creating any Panel objects
pn.extension()
hv.extension('bokeh')


# Define your app parameters
app_interferogram_dynamics = InteractiveInterferogramDynamics(
    eps0_min=-0.006,
    eps0_max=0.006,
    A_min=0.0,
    A_max=0.01,
    N_points_target=500_000,
    delta_C_range=(0, 0.001),
    GammaL0_range=(0, 1000),
    GammaR0_range=(0, 150),
    Gamma_eg0_range=(0, 50),
    Gamma_phi0_range=(0, 100),
    sigma_eps_range=(1, 10),
    N_steps_period_array=(100, 2000),
    N_periods_array=(1, 20),
    N_periods_avg_array=(1, 10),
    N_samples_noise_array=(0, 1000),
    delta_C_default=0.0003,
    GammaL0_default=420,
    GammaR0_default=68,
    Gamma_eg0_default=10,
    Gamma_phi0_default=3.6,
    sigma_eps_default=2.0,
    N_steps_period_default=1000,
    N_periods_default=10,
    N_periods_avg_default=1,
    N_samples_noise_default=100,
    dC_default_thresholds=(-3000, 1000),
    
    nu=21,
    m=10,
    B=25,
    
    platform_type=platform_type,
    repo_path=repo_path,
    cmap_name='fire',
    render_mode=render_mode
)


# Create the dashboard
dashboard = app_interferogram_dynamics.create_dashboard()

# For panel serve: wrap in a template for better layout
template = pn.template.FastListTemplate(
    title="Interactive Interferogram Dynamics",
    sidebar=[],
    main=[dashboard],
    header_background="#2E86AB",
)

# Make it servable
template.servable()