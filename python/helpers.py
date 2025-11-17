########################################
# python/helpers.py
########################################


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





