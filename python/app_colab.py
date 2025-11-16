########################################
# python/app_colab.py
########################################

"""
app_colab.py - Colab-compatible launcher for your interferogram app

Add this to your repository and run it in Colab instead of app.py
"""

from pathlib import Path
import sys
import os
import panel as pn
import holoviews as hv

# Detect Colab
try:
    import google.colab
    IN_COLAB = True
    from google.colab.output import eval_js
    from IPython.display import HTML, IFrame, display
except ImportError:
    IN_COLAB = False

# Platform detection
if 'COLAB_GPU' in os.environ or IN_COLAB:
    platform_type = 'colab_linux'
else:
    if sys.platform.startswith('linux'):
        if Path('/mnt/c/').exists():
            platform_type = 'local_wsl2'
        else:
            platform_type = 'local_linux'
    elif sys.platform.startswith('win'):
        platform_type = 'local_windows'
    else:
        raise RuntimeError("Unsupported platform")

# Set repo path
if platform_type == 'colab_linux':
    repo_path = Path('/content/cuda_python_project')
else:
    repo_path = Path('~/cuda_python_project').expanduser() if platform_type in ['local_linux', 'local_wsl2'] else Path(os.path.expandvars(r'%USERPROFILE%\cuda_python_project'))

print(f"Platform: {platform_type}")
print(f"Repo directory: {repo_path}")

# GPU detection
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
    print("[GPU] CuPy not available")
except Exception as e:
    print(f"[GPU] CuPy error: {e}")

try:
    import cudf
    CUDF_AVAILABLE = True
    print("[GPU] cuDF detected - Datashader GPU acceleration available")
    os.environ['DATASHADER_USE_CUPY'] = '1'
except ImportError:
    print("[GPU] cuDF not available - Datashader will use CPU")
except Exception as e:
    print(f"[GPU] cuDF error: {e}")

CUPY_CUDF_AVAILABLE = CUPY_AVAILABLE and CUDF_AVAILABLE

# Render mode
render_mode = 'raster_dynamic'

# Auto-fallback
if render_mode in ['raster_static_gpu', 'raster_dynamic_gpu'] and not CUPY_AVAILABLE:
    print("[GPU] GPU mode requested but CuPy not available - falling back to CPU")
    render_mode = render_mode.replace('_gpu', '')

# Enable Panel
pn.extension()
hv.extension('bokeh')

# Import your app
from app_interferogram_dynamics_class import InteractiveInterferogramDynamics

# Create app instance
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

# Create dashboard
dashboard = app_interferogram_dynamics.create_dashboard()

# CRITICAL: Different handling for Colab vs local
if IN_COLAB:
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

else:
    # Local mode - use template
    print("\n🖥️ LOCAL MODE - Use 'panel serve app_colab.py --show' instead")
    
    template = pn.template.FastListTemplate(
        title="Interactive Interferogram Dynamics",
        sidebar=[],
        main=[dashboard],
        header_background="#2E86AB",
    )
    
    template.servable()