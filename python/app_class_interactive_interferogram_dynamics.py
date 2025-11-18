########################################
# python/app_class_interactive_interferogram_dynamics.py
########################################

import numpy as np
import panel as pn
import holoviews as hv
import time
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

import matplotlib.pyplot as plt
# Get default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Enable Panel extension for Jupyter
pn.extension()
hv.extension('bokeh')

# Import simulation modules
from simulation import run_simulation
from config import SimRunGridMode, SimRunSingleMode, SimRunGridSingleMode

# Import app modules
from app_class_simulation_parameters import SimulationParameters
from app_class_interferogram_plot import InterferogramPlot
from app_class_dynamics_plot import DynamicsPlot






class InteractiveInterferogramDynamics:
    """Main coordinator class for the interactive dashboard."""
    
    def __init__(self, eps0_min, eps0_max, A_min, A_max, N_points_target,
                 delta_C_range, GammaL0_range, GammaR0_range, Gamma_eg0_range, Gamma_phi0_range, sigma_eps_range,
                 N_steps_period_array, N_periods_array, N_periods_avg_array, N_samples_noise_array,
                 delta_C_default, GammaL0_default, GammaR0_default, Gamma_eg0_default,
                 Gamma_phi0_default, sigma_eps_default, N_steps_period_default, N_periods_default, 
                 N_periods_avg_default, N_samples_noise_default,
                 dC_default_thresholds,
                 nu, m, B,
                 platform_type,
                 repo_path,
                 cmap_name,
                 render_mode):
        
        self.platform_type = platform_type
        self.repo_path = repo_path
        self.eps0_min = eps0_min
        self.eps0_max = eps0_max
        self.A_min = A_min
        self.A_max = A_max
        self.N_points_target = N_points_target
        self.m = m
        self.B = B
        
        N_steps_period_range = N_steps_period_array if isinstance(N_steps_period_array, tuple) else (int(N_steps_period_array[0]), int(N_steps_period_array[-1]))
        N_periods_range = N_periods_array if isinstance(N_periods_array, tuple) else (int(N_periods_array[0]), int(N_periods_array[-1]))
        N_periods_avg_range = N_periods_avg_array if isinstance(N_periods_avg_array, tuple) else (int(N_periods_avg_array[0]), int(N_periods_avg_array[-1]))
        N_samples_noise_range = N_samples_noise_array if isinstance(N_samples_noise_array, tuple) else (int(N_samples_noise_array[0]), int(N_samples_noise_array[-1]))

        self.sim_params = SimulationParameters(
            delta_C_range, GammaL0_range, GammaR0_range, Gamma_eg0_range, Gamma_phi0_range, sigma_eps_range,
            N_steps_period_range, N_periods_range, N_periods_avg_range, N_samples_noise_range,
            delta_C_default, GammaL0_default, GammaR0_default, Gamma_eg0_default,
            Gamma_phi0_default, sigma_eps_default,
            nu,
            N_steps_period_default, N_periods_default, 
            N_periods_avg_default, N_samples_noise_default
        )
        
        self.interferogram = InterferogramPlot(
            eps0_min, eps0_max, A_min, A_max, N_points_target,
            dC_default_thresholds, cmap_name, render_mode
        )
        
        t_max_default = N_periods_default/nu
        
        self.dynamics = DynamicsPlot(eps0_min, eps0_max, A_min, A_max, t_max_default)
        
        # Store current epsilon_L and epsilon_R values
        self.epsilon_L = None
        self.epsilon_R = None
        self._update_epsilon_bounds()  # Calculate initial values
                
        self.auto_update_enabled = False
        self.auto_update_dynamics_enabled = False
        self.auto_update_both_enabled = False
        self._is_generating = False
        self._is_generating_both = False
        self._is_processing_hover = False
        self._pending_hover_eps0 = None
        self._pending_hover_A = None
        self._debug_hover = True


        self._create_control_widgets()
        self._generate_interferogram_data()
    
    def _create_control_widgets(self):
        """Create main control widgets."""
        
        # Interferogram controls
        self.update_button = pn.widgets.Button(
            name='🔄 Regenerate Interferogram',
            button_type='success',
            width=200
        )
        
        self.auto_update_toggle = pn.widgets.Toggle(
            name='Auto-Update Interferogram',
            value=False,
            button_type='warning',
            width=200
        )
        
        # Dynamics controls
        self.dynamics_regenerate_button = pn.widgets.Button(
            name='🔄 Regenerate Dynamics',
            button_type='success',
            width=200
        )
        
        self.auto_update_dynamics_toggle = pn.widgets.Toggle(
            name='Auto-Update Dynamics',
            value=False,
            button_type='warning',
            width=200
        )
        
        # Both controls
        self.regenerate_both_button = pn.widgets.Button(
            name='🔄 Regenerate Both',
            button_type='primary',
            width=200
        )
        
        self.auto_update_both_toggle = pn.widgets.Toggle(
            name='Auto-Update Both',
            value=False,
            button_type='danger',
            width=200
        )
        
        self.status_text = pn.pane.Markdown(
            "**Status:** Ready",
            width=300#,
            #sizing_mode='fixed'
        )
        
        self.timing_text = pn.pane.Markdown(
            "**Last computation:** N/A",
            width=300#,
            #sizing_mode='fixed'
        )
        
        self.log_display = pn.pane.Markdown(
            "**Log:** Waiting for first computation...",
            width=700,
            height=300,
            styles={'resize': 'both', 'overflow': 'auto', 'background': '#f5f5f5', 
                   'padding': '10px', 'font-family': 'monospace', 'font-size': '11px', 
                   'border': '1px solid #ddd'},
            sizing_mode='fixed'
        )
        
        self.auto_update_toggle.param.watch(self._on_auto_update_interferogram_toggle, 'value')
        self.auto_update_dynamics_toggle.param.watch(self._on_auto_update_dynamics_toggle, 'value')
        self.auto_update_both_toggle.param.watch(self._on_auto_update_both_toggle, 'value')
        
        # Watch parameter changes for auto-update
        for param in ['delta_C', 'GammaL0', 'GammaR0', 'Gamma_eg0', 'Gamma_phi0', 'sigma_eps',
                     'N_steps_period', 'N_periods', 'N_periods_avg', 'N_samples_noise']:
            slider = getattr(self.sim_params, f'{param}_slider')
            slider.param.watch(self._on_parameter_change, 'value')
        
        self.sim_params.quasi_static_toggle.param.watch(self._on_parameter_change, 'value')
    
    def _on_auto_update_interferogram_toggle(self, event):
        """Handle interferogram auto-update toggle."""
        self.auto_update_enabled = event.new
        
        # Sync state
        if event.new and self.auto_update_both_enabled:
            self.auto_update_both_toggle.value = False
        
        if self.auto_update_enabled:
            self.update_button.name = '🔄 Regenerate Interferogram (Auto)'
            self.update_button.button_type = 'light'
        else:
            self.update_button.name = '🔄 Regenerate Interferogram'
            self.update_button.button_type = 'success'
    
    def _on_auto_update_dynamics_toggle(self, event):
        """Handle dynamics auto-update toggle."""
        self.auto_update_dynamics_enabled = event.new
        
        # Sync state
        if event.new and self.auto_update_both_enabled:
            self.auto_update_both_toggle.value = False
        
        if self.auto_update_dynamics_enabled:
            self.dynamics_regenerate_button.name = '🔄 Regenerate Dynamics (Auto)'
            self.dynamics_regenerate_button.button_type = 'light'
        else:
            self.dynamics_regenerate_button.name = '🔄 Regenerate Dynamics'
            self.dynamics_regenerate_button.button_type = 'success'
    
    def _on_auto_update_both_toggle(self, event):
        """Handle both auto-update toggle."""
        self.auto_update_both_enabled = event.new
        
        # Sync state - disable individual toggles when both is enabled
        if event.new:
            if self.auto_update_enabled:
                self.auto_update_toggle.value = False
            if self.auto_update_dynamics_enabled:
                self.auto_update_dynamics_toggle.value = False
            
            # Transfer dynamics auto-update behavior to "both"
            self.dynamics.auto_update = True
            self.dynamics.click_count = 0
            self.dynamics.hover_active = False
        else:
            self.dynamics.auto_update = False
            self.dynamics.click_count = 0
            self.dynamics.hover_active = False
        
        if self.auto_update_both_enabled:
            self.regenerate_both_button.name = '🔄 Regenerate Both (Auto)'
            self.regenerate_both_button.button_type = 'light'
        else:
            self.regenerate_both_button.name = '🔄 Regenerate Both'
            self.regenerate_both_button.button_type = 'primary'
    
    def _on_parameter_change(self, event):
        """Handle parameter slider changes."""
        # Update epsilon bounds when delta_C changes
        self._update_epsilon_bounds()
        
        # CRITICAL: Skip if ANY computation is in progress
        if self._is_generating or self._is_generating_both or self.dynamics.computing:
            return
        
        if self.auto_update_enabled:
            self._update_and_regenerate_interferogram()
        elif self.auto_update_dynamics_enabled:
            # Only regenerate dynamics if it has been computed before
            if self.dynamics.current_eps0 is not None and self.dynamics.current_A is not None:
                self._regenerate_dynamics_only()
        elif self.auto_update_both_enabled:
            self._regenerate_both()
    
    def _update_and_regenerate_interferogram(self, event=None):
        """Update parameters and regenerate interferogram only."""
        # CRITICAL: Skip if ANY computation is in progress
        if self._is_generating or self._is_generating_both or self.dynamics.computing:
            return
        self.sim_params.update_from_sliders()
        self._generate_interferogram_data()
    
    def _regenerate_dynamics_only(self, event=None):
        """Regenerate dynamics at current position with updated parameters."""
        if self.dynamics.current_eps0 is None or self.dynamics.current_A is None:
            self.dynamics.status_text.object = "**Dynamics:** ⚠️ No position set. Click on interferogram first."
            return
        
        # CRITICAL: Skip if ANY computation is in progress
        if self._is_generating or self._is_generating_both or self.dynamics.computing:
            return
        
        # FIXED: Pass update_params=True so it uses current slider values
        self._generate_dynamics(
            self.dynamics.current_eps0, 
            self.dynamics.current_A,
            update_params=True
        )
    
    def _regenerate_both(self, event=None):
        """Regenerate both interferogram and dynamics simultaneously using SimRunGridSingleMode."""
        # Check if dynamics position is set
        if self.dynamics.current_eps0 is None or self.dynamics.current_A is None:
            self.status_text.object = "**Status:** ⚠️ Set dynamics position first (click on interferogram)"
            return
        
        # CRITICAL: Skip if ANY computation is in progress
        if self._is_generating or self._is_generating_both or self.dynamics.computing:
            return
        
        self._is_generating_both = True
        self.sim_params.update_from_sliders()
        
        captured_stdout = StringIO()
        captured_stderr = StringIO()
        
        self.status_text.object = "**Status:** 🔄 Generating both..."
        
        try:
            with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                start_time = time.perf_counter()
                
                simr = SimRunGridSingleMode(
                    **self.sim_params.get_simrun_kwargs(
                        self.platform_type, self.repo_path,
                        eps0_min=self.eps0_min,
                        eps0_max=self.eps0_max,
                        A_min=self.A_min,
                        A_max=self.A_max,
                        N_points_target=self.N_points_target,
                        eps0_target_singlepoint=self.dynamics.current_eps0,
                        A_target_singlepoint=self.dynamics.current_A
                    )
                )
                
                (eps0_grid, A_grid, rho_avg_cdc_3d, 
                 time_dynamics, eps_dynamics, rho_dynamics, rho_avg, returncode) = run_simulation(simr)
                
                end_time = time.perf_counter()
                elapsed = end_time - start_time
            
            # Update interferogram data
            self.interferogram.update_data(eps0_grid, A_grid, rho_avg_cdc_3d)
            
            # Update dynamics data
            self.dynamics.time_dynamics = time_dynamics
            self.dynamics.eps_dynamics = eps_dynamics
            self.dynamics.rho_dynamics = rho_dynamics
            self.dynamics.computation_time = elapsed
            
            # Update t_max_plot for dynamics
            if time_dynamics is not None and len(time_dynamics) > 0:
                self.dynamics.t_max_plot = time_dynamics[-1]
            
            # Update dynamics version to trigger plot refresh
            self.dynamics.dynamics_version += 1
            if hasattr(self.dynamics, 'dynamics_version_widget'):
                self.dynamics.dynamics_version_widget.value = self.dynamics.dynamics_version
            
            # Update status
            self.status_text.object = "**Status:** ✅ Both ready"
            self.timing_text.object = f"**Last computation:** {elapsed:.2f} seconds"
            self.dynamics.status_text.object = (
                f"**Dynamics:** ✅ Ready ({elapsed:.2f}s) | "
                f"eps0={self.dynamics.current_eps0:.6f}, A={self.dynamics.current_A:.6f}"
            )
            
            log_text = f"**✅ Both computed in {elapsed:.2f}s**\n\n"
            
        except Exception as e:
            end_time = time.perf_counter()
            elapsed = end_time - start_time if 'start_time' in locals() else 0
            
            self.status_text.object = "**Status:** ❌ Error occurred"
            self.timing_text.object = f"**Failed after:** {elapsed:.2f} seconds"
            self.dynamics.status_text.object = "**Dynamics:** ❌ Error"
            
            log_text = f"**❌ ERROR after {elapsed:.2f}s**\n\n"
            log_text += f"**Exception Type:** {type(e).__name__}\n\n"
            log_text += f"**Error Message:**\n```\n{str(e)}\n```\n\n"
        
        # Log parameters and output
        log_text += "**Parameters:**\n"
        log_text += f"- delta_C = {self.sim_params.delta_C:.6e}\n"
        log_text += f"- GammaL0 = {self.sim_params.GammaL0}, GammaR0 = {self.sim_params.GammaR0}\n"
        log_text += f"- Gamma_eg0 = {self.sim_params.Gamma_eg0}"
        if self.sim_params.quasi_static:
            log_text += f"\n- sigma_eps = {self.sim_params.sigma_eps}\n"
            log_text += f"- N_samples_noise = {self.sim_params.N_samples_noise}\n"
        else:
            log_text += f", Gamma_phi0 = {self.sim_params.Gamma_phi0}\n"
        log_text += f"- Quasi-static mode: {self.sim_params.quasi_static}\n"
        log_text += f"- Dynamics at: eps0={self.dynamics.current_eps0:.8f}, A={self.dynamics.current_A:.8f}\n\n"
        
        stdout_content = captured_stdout.getvalue()
        stderr_content = captured_stderr.getvalue()
        
        if stdout_content:
            log_text += f"**CUDA Output:**\n```\n{stdout_content}\n```\n"
        if stderr_content:
            log_text += f"**CUDA Errors:**\n```\n{stderr_content}\n```\n"
        
        self.log_display.object = log_text
        
        self._is_generating_both = False
    
    def _generate_interferogram_data(self):
        """Generate interferogram data."""
    
        # CRITICAL: Skip if ANY computation is in progress
        if self._is_generating or self._is_generating_both or self.dynamics.computing:
            return
        
        self._is_generating = True
        
        captured_stdout = StringIO()
        captured_stderr = StringIO()
        
        if hasattr(self, 'status_text'):
            self.status_text.object = "**Status:** 🔄 Generating data..."
        
        try:
            with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                start_time = time.perf_counter()
                
                simr = SimRunGridMode(
                    **self.sim_params.get_simrun_kwargs(
                        self.platform_type, self.repo_path,
                        eps0_min=self.eps0_min,
                        eps0_max=self.eps0_max,
                        A_min=self.A_min,
                        A_max=self.A_max,
                        N_points_target=self.N_points_target
                    )
                )
                
                eps0_grid, A_grid, rho_avg_cdc_3d, _ = run_simulation(simr)
                
                end_time = time.perf_counter()
                elapsed = end_time - start_time
            
            # Success - update display
            self.interferogram.update_data(eps0_grid, A_grid, rho_avg_cdc_3d)
            
            if hasattr(self, 'status_text'):
                self.status_text.object = "**Status:** ✅ Data ready"
            if hasattr(self, 'timing_text'):
                self.timing_text.object = f"**Last computation:** {elapsed:.2f} seconds"
            
            log_text = f"**✅ Computation completed in {elapsed:.2f}s**\n\n"
            
        except Exception as e:
            # Error - capture and display
            end_time = time.perf_counter()
            elapsed = end_time - start_time if 'start_time' in locals() else 0
            
            if hasattr(self, 'status_text'):
                self.status_text.object = "**Status:** ❌ Error occurred"
            if hasattr(self, 'timing_text'):
                self.timing_text.object = f"**Failed after:** {elapsed:.2f} seconds"
            
            log_text = f"**❌ ERROR after {elapsed:.2f}s**\n\n"
            log_text += f"**Exception Type:** {type(e).__name__}\n\n"
            log_text += f"**Error Message:**\n```\n{str(e)}\n```\n\n"
        
        # Always show parameters and output
        log_text += "**Parameters:**\n"
        log_text += f"- delta_C = {self.sim_params.delta_C:.6e}\n"
        log_text += f"- GammaL0 = {self.sim_params.GammaL0}, GammaR0 = {self.sim_params.GammaR0}\n"
        log_text += f"- Gamma_eg0 = {self.sim_params.Gamma_eg0}"
        if self.sim_params.quasi_static:
            log_text += f"\n- sigma_eps = {self.sim_params.sigma_eps}\n"
            log_text += f"- N_samples_noise = {self.sim_params.N_samples_noise}\n"
        else:
            log_text += f", Gamma_phi0 = {self.sim_params.Gamma_phi0}\n"
        log_text += f"- Quasi-static mode: {self.sim_params.quasi_static}\n\n"
        
        stdout_content = captured_stdout.getvalue()
        stderr_content = captured_stderr.getvalue()
        
        if stdout_content:
            log_text += f"**CUDA Output:**\n```\n{stdout_content}\n```\n"
        if stderr_content:
            log_text += f"**CUDA Errors:**\n```\n{stderr_content}\n```\n"
    
        if hasattr(self, 'log_display'):
            self.log_display.object = log_text
        
        self._is_generating = False
    
    def _update_epsilon_bounds(self):
        """Calculate epsilon_L and epsilon_R based on current delta_C, m, and B."""
        delta_C = self.sim_params.delta_C
        m = self.m
        B = self.B
        
        # Calculate epsilon_L and epsilon_R
        # ε_{L,R} = [B ± sqrt(1 + m²(B² - 1)ΔC²)] / [m(B² - 1)]
        
        B2_minus_1 = B**2 - 1
        
        if abs(B2_minus_1) < 1e-10:  # Avoid division by zero
            self.epsilon_L = 0.0
            self.epsilon_R = 0.0
            return
        
        sqrt_term = np.sqrt(1 + m**2 * B2_minus_1 * delta_C**2)
        
        self.epsilon_R = (B + sqrt_term) / (m * B2_minus_1)
        self.epsilon_L = (B - sqrt_term) / (m * B2_minus_1)
        
        # Update widgets to trigger plot refresh
        if hasattr(self.dynamics, 'epsilon_L_widget'):
            self.dynamics.epsilon_L_widget.value = self.epsilon_L
        if hasattr(self.dynamics, 'epsilon_R_widget'):
            self.dynamics.epsilon_R_widget.value = self.epsilon_R
    
    def _on_interferogram_click(self, x, y):
        """Handle click on interferogram."""
        
        if not self.dynamics.enabled or x is None or y is None:
            return
        
        # CRITICAL: Skip if ANY computation is in progress
        if self._is_generating or self._is_generating_both or self.dynamics.computing:
            return
        
        eps0, A = x, y
        
        # Check if "Auto-Update Both" is active
        if self.auto_update_both_enabled:
            if self.dynamics.click_count % 2 == 0:
                self.dynamics.hover_active = True
                self.interferogram.marker_eps0 = None
                self.interferogram.marker_A = None
                self._generate_dynamics(eps0, A, update_params=False)
            else:
                self.dynamics.hover_active = False
                if (self.eps0_min <= eps0 <= self.eps0_max and 
                    self.A_min <= A <= self.A_max):
                    self.interferogram.marker_eps0 = eps0
                    self.interferogram.marker_A = A
                else:
                    self.interferogram.marker_eps0 = None
                    self.interferogram.marker_A = None
                self._generate_dynamics(eps0, A, update_params=False)
            
            self.dynamics.click_count += 1
        elif self.dynamics.auto_update:
            # Original dynamics auto-update behavior
            if self.dynamics.click_count % 2 == 0:
                self.dynamics.hover_active = True
                self.interferogram.marker_eps0 = None
                self.interferogram.marker_A = None
                self._generate_dynamics(eps0, A, update_params=False)
            else:
                self.dynamics.hover_active = False
                if (self.eps0_min <= eps0 <= self.eps0_max and 
                    self.A_min <= A <= self.A_max):
                    self.interferogram.marker_eps0 = eps0
                    self.interferogram.marker_A = A
                else:
                    self.interferogram.marker_eps0 = None
                    self.interferogram.marker_A = None
                self._generate_dynamics(eps0, A, update_params=False)
            
            self.dynamics.click_count += 1
        else:
            # Normal click behavior
            if (self.eps0_min <= eps0 <= self.eps0_max and 
                self.A_min <= A <= self.A_max):
                self.interferogram.marker_eps0 = eps0
                self.interferogram.marker_A = A
            else:
                self.interferogram.marker_eps0 = None
                self.interferogram.marker_A = None
            self._generate_dynamics(eps0, A, update_params=False)
        
        self.interferogram.marker_version += 1
        if hasattr(self.interferogram, 'marker_version_widget'):
            self.interferogram.marker_version_widget.value = self.interferogram.marker_version
            
    def _on_interferogram_hover(self, x, y):
        """Handle hover on interferogram - processes only the latest position."""
        
        if self._debug_hover:
            x_display = f"{x:.4f}" if x else "None"
            y_display = f"{y:.4f}" if y else "None"
            print(f"🔵 HOVER EVENT: x={x_display}, y={y_display}", flush=True)
        
        if not self.dynamics.enabled or x is None or y is None:
            if self._debug_hover:
                print(f"  ↳ SKIP: enabled={self.dynamics.enabled}", flush=True)
            return
        
        # Check if hover mode is active
        hover_mode_active = (self.auto_update_both_enabled and self.dynamics.hover_active) or \
                            (self.dynamics.auto_update and self.dynamics.hover_active)
        
        if self._debug_hover:
            print(f"  ↳ Hover mode active: {hover_mode_active}", flush=True)
        
        if not hover_mode_active:
            if self._debug_hover:
                print("  ↳ SKIP: Hover mode not active", flush=True)
            return
        
        # CRITICAL FIX: Always update pending coordinates
        self._pending_hover_eps0 = x
        self._pending_hover_A = y
        
        if self._debug_hover:
            print(f"  ↳ Pending coordinates updated: ({x:.6f}, {y:.6f})", flush=True)
        
        # If already processing, just return - the processing loop will pick up the new coords
        if self._is_processing_hover:
            if self._debug_hover:
                print("  ↳ Already processing - new coordinates will be picked up", flush=True)
            return
        
        # Start processing
        if self._debug_hover:
            print("  ↳ ✅ Starting processing loop", flush=True)
        
        self._is_processing_hover = True
        self._process_pending_hover()
    
    def _process_pending_hover(self):
        """Process only the LATEST pending hover computation, skipping intermediates."""
        try:
            # Only process if we still have pending coordinates
            if self._pending_hover_eps0 is None:
                return
                
            # Capture current coordinates
            eps0 = self._pending_hover_eps0
            A = self._pending_hover_A
            
            # CRITICAL: Clear BEFORE computing so new hovers during computation
            # will overwrite these values, and we'll know to skip this computation
            self._pending_hover_eps0 = None
            self._pending_hover_A = None
            
            if self._debug_hover:
                print(f"    🟢 Computing for ({eps0:.6f}, {A:.6f})", flush=True)
            
            # Generate dynamics - this is the slow part
            self._generate_dynamics(eps0, A, update_params=False)
            
            if self._debug_hover:
                print("    🟢 Computation finished", flush=True)
            
            # Check if NEW coordinates arrived during computation
            if self._pending_hover_eps0 is not None:
                # New hover arrived - process ONLY the latest one
                if self._debug_hover:
                    new_eps0 = self._pending_hover_eps0
                    new_A = self._pending_hover_A
                    print(f"    🔄 New hover detected ({new_eps0:.6f}, {new_A:.6f}), processing latest only", flush=True)
                
                # Recursively call to process the LATEST coordinates
                # (any intermediate ones that arrived are already overwritten and lost)
                self._process_pending_hover()
            else:
                if self._debug_hover:
                    print("    ✅ No new hovers, exiting", flush=True)
                    
        finally:
            # Always clear flag
            self._is_processing_hover = False
            if self._debug_hover:
                print("    🔓 Processing finished, flag cleared", flush=True)
    
    def _generate_dynamics(self, eps0, A, update_params=False):
        """Generate dynamics for given coordinates.
        
        Args:
            update_params: If True, update parameters from sliders before computing
        """
        
        if self._debug_hover:
            print(f"    🟢 _generate_dynamics CALLED: eps0={eps0:.6f}, A={A:.6f}, update_params={update_params}", flush=True)
        
        # CRITICAL: Final safety check - skip if ANY computation is in progress
        if self._is_generating or self._is_generating_both or self.dynamics.computing:
            if self._debug_hover:
                print("    ❌ _generate_dynamics BLOCKED: Another computation in progress", flush=True)
            return
        
        if self._debug_hover:
            print("    🟢 _generate_dynamics PROCEEDING: Calling dynamics.compute()", flush=True)
        
        def log_callback(text):
            self.log_display.object = text
            self.log_display.param.trigger('object')
        
        def marker_update_callback():
            self.interferogram.marker_version += 1
            if hasattr(self.interferogram, 'marker_version_widget'):
                self.interferogram.marker_version_widget.value = self.interferogram.marker_version
        
        self.dynamics.compute(eps0, A, self.sim_params, 
                            self.platform_type, self.repo_path, 
                            log_callback, marker_update_callback, 
                            update_params=update_params)
        
        if self._debug_hover:
            print("    🟢 _generate_dynamics FINISHED: dynamics.compute() returned", flush=True)
    
    def _on_manual_dynamics_generate(self, event=None):
        """Handle manual coordinate entry for dynamics."""
        eps0 = self.dynamics.eps0_input.value
        A = self.dynamics.A_input.value
        
        self.interferogram.set_marker(eps0, A)
        
        self.dynamics.auto_toggle.value = False
        self.dynamics.hover_active = False
        
        # Update parameters before generating
        self._generate_dynamics(eps0, A, update_params=True)
    
    def create_dashboard(self):
        """Create the complete Panel dashboard with direct Bokeh integration."""
        
        # Button callbacks
        self.update_button.on_click(self._update_and_regenerate_interferogram)
        self.dynamics_regenerate_button.on_click(self._regenerate_dynamics_only)
        self.regenerate_both_button.on_click(self._regenerate_both)
        self.dynamics.generate_button.on_click(self._on_manual_dynamics_generate)
        
        # Create a custom Panel component that wraps the interferogram with callbacks
        class InterferogramPane(pn.pane.HoloViews):
            """Custom pane that attaches Bokeh events."""
            
            def __init__(self, parent_app, **params):
                self.parent_app = parent_app
                super().__init__(**params)
                
            def _get_model(self, doc, root=None, parent=None, comm=None):
                """Override to attach events after model creation."""
                model = super()._get_model(doc, root, parent, comm)
                
                # Attach Bokeh events
                if model is not None:
                    from bokeh.events import Tap, MouseMove
                    
                    def on_tap_event(event):
                        if hasattr(event, 'x') and hasattr(event, 'y'):
                            if event.x is not None and event.y is not None:
                                self.parent_app._on_interferogram_click(event.x, event.y)
                                #print(f"✅ Tap at: ({event.x:.6f}, {event.y:.6f})", flush=True)
                    
                    def on_hover_event(event):
                        if hasattr(event, 'x') and hasattr(event, 'y'):
                            if event.x is not None and event.y is not None:
                                self.parent_app._on_interferogram_hover(event.x, event.y)
                    
                    try:
                        model.on_event(Tap, on_tap_event)
                        model.on_event(MouseMove, on_hover_event)
                        #print("✅ Events attached to Bokeh model", flush=True)
                    except Exception as e:
                        print(f"⚠️ Event attachment failed: {e}", flush=True)
                
                return model
        
        # Create interferogram with custom pane
        interferogram_dmap = self.interferogram.create_plot()
        
        interferogram_pane = InterferogramPane(
            self,
            object=interferogram_dmap,
            sizing_mode='fixed',
            width=800,
            height=600,
            backend='bokeh'
        )
        
        # Sidebar
        sidebar = pn.Column(
            "## 🎛️ Simulation Parameters",
            pn.layout.Divider(),
            self.sim_params.get_control_panel(),
            pn.layout.Divider(),
            "### Regeneration Controls",
            pn.Column(
                self.update_button,
                self.dynamics_regenerate_button,
                self.regenerate_both_button,
                width=220
            ),
            pn.layout.Divider(),
            "### Auto-Update Controls",
            pn.Column(
                self.auto_update_toggle,
                self.auto_update_dynamics_toggle,
                self.auto_update_both_toggle,
                width=220
            ),
            pn.layout.Divider(),
            self.status_text,
            self.timing_text,
            width=500,
            #sizing_mode='fixed',
            scroll=True
        )
        
        # Interferogram section
        interferogram_section = pn.Column(
            self.interferogram.get_control_panel(),
            interferogram_pane,
            self.interferogram.data_version_widget,
            self.interferogram.marker_version_widget#,
            #sizing_mode='fixed'
        )
        
        # Dynamics plot
        dynamics_dmap = self.dynamics.create_plot()
        
        dynamics_pane = pn.pane.HoloViews(
            dynamics_dmap,
            sizing_mode='fixed',
            width=800,
            height=600,
            backend='bokeh'
        )
        
        dynamics_plot_panel = pn.Column(
            dynamics_pane,
            self.dynamics.dynamics_version_widget,
            self.dynamics.epsilon_L_widget,
            self.dynamics.epsilon_R_widget,
            #sizing_mode='fixed',
            visible=False
        )
        
        dynamics_section = pn.Column(
            pn.layout.Divider(),
            "### Dynamics Plot",
            self.dynamics.get_control_panel(),
            self.dynamics.status_text,
            dynamics_plot_panel#,
            #sizing_mode='fixed'
        )
        
        def update_dynamics_visibility(event):
            dynamics_plot_panel.visible = event.new
        
        self.dynamics.show_toggle.param.watch(update_dynamics_visibility, 'value')
        
        plot_area = pn.Column(
            interferogram_section,
            dynamics_section,
            pn.layout.Divider(),
            "### Computation Log",
            self.log_display,
            #sizing_mode='fixed',
            scroll=True
        )
        
        dashboard = pn.Row(
            sidebar,
            plot_area#,
            #sizing_mode='fixed'
        )
        
        return dashboard


