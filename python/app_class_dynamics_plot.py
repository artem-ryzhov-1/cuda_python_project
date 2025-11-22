########################################
# python/app_class_dynamics_plot.py
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






class DynamicsPlot:
    """Manages the dynamics plot and computation."""
    
    def __init__(self, eps0_min, eps0_max, A_min, A_max, t_max_default):
        
        self.eps0_min = eps0_min
        self.eps0_max = eps0_max
        self.A_min = A_min
        self.A_max = A_max
        
        self.time_dynamics = None
        self.eps_dynamics = None
        self.rho_dynamics = None
        self.current_eps0 = None
        self.current_A = None
        self.computation_time = 0
        self.t_max_plot = t_max_default
        
        self.enabled = False
        self.auto_update = False
        self.hover_active = False
        self.computing = False
        self.click_count = 0
        
        self.dynamics_version = 0
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create dynamics-specific widgets."""
        
        self.show_toggle = pn.widgets.Toggle(
            name='Show Dynamics Plot',
            value=False,
            button_type='primary',
            width=140
        )
        
        self.auto_toggle = pn.widgets.Toggle(
            name='Auto-Update Dynamics',
            value=False,
            button_type='warning',
            width=140,
            disabled=True
        )
        
        self.eps0_input = pn.widgets.FloatInput(
            name='eps0',
            value=0.0,
            step=0.00001,
            format='0.9f',
            width=120
        )
        
        self.A_input = pn.widgets.FloatInput(
            name='A',
            value=0.0,
            step=0.00001,
            format='0.9f',
            width=120
        )
        
        self.generate_button = pn.widgets.Button(
            name='Generate',
            button_type='success',
            width=80
        )
        
        self.status_text = pn.pane.Markdown(
            "**Dynamics:** Not computed",
            width=500#,
            #sizing_mode='fixed'
        )
        
        # Version widget - styled invisible for panel serve compatibility
        self.dynamics_version_widget = pn.widgets.IntInput(
            value=0,
            visible=False
        )
        
        self.show_toggle.param.watch(self._on_show_toggle, 'value')
        self.auto_toggle.param.watch(self._on_auto_toggle, 'value')
        
        # Epsilon bound widgets - invisible
        self.epsilon_L_widget = pn.widgets.FloatInput(
            value=0.0,
            visible=False
        )
        self.epsilon_R_widget = pn.widgets.FloatInput(
            value=0.0,
            visible=False
        )

        # t_max_plot widget - styled invisible for panel serve compatibility
        self.t_max_plot_widget = pn.widgets.FloatInput(
            value=self.t_max_plot,
            visible=False
        )
    
    def _on_show_toggle(self, event):
        """Handle show toggle."""
        self.enabled = event.new
        self.auto_toggle.disabled = not event.new
        
        if not event.new:
            self.auto_update = False
            self.hover_active = False
            self.click_count = 0
    
    def _on_auto_toggle(self, event):
        """Handle auto-update toggle."""
        self.auto_update = event.new
        self.click_count = 0
        self.hover_active = False
    
    def compute(self, eps0, A, sim_params, platform_type, repo_path, log_callback=None, marker_update_callback=None, update_params=True):
        """Compute dynamics for given coordinates.
        
        Args:
            eps0: Target epsilon_0 value
            A: Target amplitude value
            sim_params: SimulationParameters instance
            platform_type: Platform type for simulation
            repo_path: Repository path for simulation
            log_callback: Optional callback for logging
            marker_update_callback: Optional callback for marker updates
            update_params: If True, update sim_params from sliders before computing
        """
        
        print(f"      🔷 DynamicsPlot.compute CALLED: computing={self.computing}", flush=True)
        
        if self.computing:
            print("      ❌ DynamicsPlot.compute BLOCKED: Already computing", flush=True)
            return
        
        self.computing = True
        print("      🔷 DynamicsPlot.compute STARTED: Flag set to True", flush=True)
        
        self.current_eps0 = eps0
        self.current_A = A
        
        self.eps0_input.value = eps0
        self.A_input.value = A
        
        # CRITICAL FIX: Update parameters if requested
        if update_params:
            sim_params.update_from_sliders()
        
        captured_stdout = StringIO()
        captured_stderr = StringIO()
        
        with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
            self.status_text.object = "**Dynamics:** 🔄 Computing..."
            start_time = time.perf_counter()
            
            simr = SimRunSingleMode(
                **sim_params.get_simrun_kwargs(
                    platform_type, repo_path,
                    eps0_target_singlepoint=eps0,
                    A_target_singlepoint=A
                )
            )
            
            try:
                self.time_dynamics, self.eps_dynamics, self.rho_dynamics, _, _ = run_simulation(simr)
                end_time = time.perf_counter()
                self.computation_time = end_time - start_time
                
                # Update plot time limit after successful computation
                if self.time_dynamics is not None and len(self.time_dynamics) > 0:
                    self.t_max_plot = self.time_dynamics[-1]
                    self.t_max_plot_widget.value = self.t_max_plot
                
                self.status_text.object = f"**Dynamics:** ✅ Ready ({self.computation_time:.2f}s) | eps0={eps0:.6f}, A={A:.6f}"
            except Exception as e:
                self.status_text.object = f"**Dynamics:** ❌ Error: {str(e)}"
                self.computing = False
                return
        
        if log_callback:
            log_text = f"**Computation completed in {self.computation_time:.2f}s**\n\n"
            log_text += f"**Parameters:** eps0={eps0:.8f}, A={A:.8f}\n\n"
            
            stdout_content = captured_stdout.getvalue()
            stderr_content = captured_stderr.getvalue()
            
            if stdout_content:
                log_text += f"**Output:**\n```\n{stdout_content}\n```\n"
            if stderr_content:
                log_text += f"**Warnings:**\n```\n{stderr_content}\n```\n"
            
            log_callback(log_text)
        
        if marker_update_callback:
            marker_update_callback()
        
        self.dynamics_version += 1
        if hasattr(self, 'dynamics_version_widget'):
            self.dynamics_version_widget.value = self.dynamics_version
        
        # ======================================================
        # FULL REBUILD OF THE DYNAMICS PLOT (CRITICAL FIX)
        # ======================================================
        if hasattr(self.parent_app, "dynamics_plot_pane"):
            new_plot = self.create_plot()
            self.parent_app.dynamics_plot_pane.object = new_plot
        # ======================================================

        print("      🔷 DynamicsPlot.compute FINISHING: About to clear flag", flush=True)
        self.computing = False
        print("      🔷 DynamicsPlot.compute FINISHED: Flag cleared", flush=True)
    
    def create_plot(self):
        """Create the dynamics plot with linked X axes and independent Y axes."""

        def make_plot(version, epsilon_L, epsilon_R, t_max_plot):

            print(f"🎨 make_plot CALLED: version={version}, t_max_plot={t_max_plot}", flush=True)
            
            time = self.time_dynamics if self.time_dynamics is not None else np.array([0, 1])
            
            if self.time_dynamics is not None and self.current_eps0 is not None and self.current_A is not None:
                eps_min = self.current_eps0 - self.current_A * 1.1
                eps_max = self.current_eps0 + self.current_A * 1.1
            else:
                eps_min = -0.01
                eps_max = 0.01
            
            print(f"   🔍 time_dynamics is None: {self.time_dynamics is None}", flush=True)
            print(f"   🔍 rho_dynamics is None: {self.rho_dynamics is None}", flush=True)
            
            if self.time_dynamics is None or self.rho_dynamics is None:
                print(f"   📊 Creating EMPTY plot with xlim=(0, {t_max_plot})", flush=True)
                # Empty population plot
                pop_data = np.zeros(len(time))
                curve_p00 = hv.Curve((time, pop_data), 'time', 'population', label='p00').opts(color=colors[0], line_width=1.5)
                curve_p01 = hv.Curve((time, pop_data), 'time', 'population', label='p01').opts(color=colors[1], line_width=1.5)
                curve_p10 = hv.Curve((time, pop_data), 'time', 'population', label='p10').opts(color=colors[2], line_width=1.5)
                curve_p11 = hv.Curve((time, pop_data), 'time', 'population', label='p11').opts(color=colors[3], line_width=1.5)

                pop_plot = (curve_p00 * curve_p01 * curve_p10 * curve_p11).opts(
                    width=800, height=350,
                    title='Population Dynamics (click on interferogram)',
                    xlabel='Time', ylabel='Population',
                    show_grid=True,
                    xlim=(0, t_max_plot),
                    ylim=(0, 1),
                    legend_position='right',
                    default_tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'save']
                )

                # Empty epsilon plot
                eps_curve = hv.Curve((time, np.zeros(len(time))), 'time', 'epsilon').opts(
                    color=colors[4], line_width=1.5
                )
                
                hline_eps_R_pos = hv.HLine(epsilon_R).opts(color='red', line_width=1, line_dash='dashed', alpha=0.7)
                hline_eps_R_neg = hv.HLine(-epsilon_R).opts(color='red', line_width=1, line_dash='dashed', alpha=0.7)
                hline_eps_L_pos = hv.HLine(epsilon_L).opts(color='green', line_width=1, line_dash='dashed', alpha=0.7)
                hline_eps_L_neg = hv.HLine(-epsilon_L).opts(color='green', line_width=1, line_dash='dashed', alpha=0.7)
                
                eps_overlay = eps_curve * hline_eps_R_pos * hline_eps_R_neg * hline_eps_L_pos * hline_eps_L_neg
                eps_plot = eps_overlay.opts(
                    width=800, height=200,
                    title='Epsilon Dynamics',
                    xlabel='Time', ylabel='ε(t)',
                    show_grid=True,
                    xlim=(0, t_max_plot),
                    ylim=(eps_min, eps_max),
                    default_tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'save']
                )
            else:
                print(f"   📊 Creating REAL plot with xlim=(0, {t_max_plot})", flush=True)
                print(f"   📊 time_dynamics length: {len(self.time_dynamics)}", flush=True)
                print(f"   📊 time_dynamics[-1]: {self.time_dynamics[-1]}", flush=True)
                
                # Real population plot
                time = self.time_dynamics
                p00 = self.rho_dynamics[:, 0]
                p01 = self.rho_dynamics[:, 1]
                p10 = self.rho_dynamics[:, 2]
                p11 = self.rho_dynamics[:, 3]

                pop_max = np.max([p00.max(), p01.max(), p10.max(), p11.max()])
                pop_ylim = (0, pop_max * 1.1)

                curve_p00 = hv.Curve((time, p00), 'time', 'population', label='p00').opts(color=colors[0], line_width=1.5)
                curve_p01 = hv.Curve((time, p01), 'time', 'population', label='p01').opts(color=colors[1], line_width=1.5)
                curve_p10 = hv.Curve((time, p10), 'time', 'population', label='p10').opts(color=colors[2], line_width=1.5)
                curve_p11 = hv.Curve((time, p11), 'time', 'population', label='p11').opts(color=colors[3], line_width=1.5)
                
                pop_overlay = (curve_p00 * curve_p01 * curve_p10 * curve_p11)
                pop_plot = pop_overlay.opts(
                    width=800, height=350,
                    title=f'Population Dynamics (eps0={self.current_eps0:.6f}, A={self.current_A:.6f}), t_max={t_max_plot:.1f}',
                    xlabel='Time', ylabel='Population',
                    show_grid=True,
                    xlim=(0, t_max_plot),
                    ylim=pop_ylim,
                    legend_position='right',
                    default_tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'],
                    framewise=True  # Force independent frame calculation
                )

                # Real epsilon plot
                eps_curve = hv.Curve((time, self.eps_dynamics), 'time', 'epsilon').opts(
                    color=colors[4], line_width=1.5
                )
                
                hline_eps_R_pos = hv.HLine(epsilon_R).opts(color='red', line_width=1.5, line_dash='dashed', alpha=0.8)
                hline_eps_R_neg = hv.HLine(-epsilon_R).opts(color='red', line_width=1.5, line_dash='dashed', alpha=0.8)
                hline_eps_L_pos = hv.HLine(epsilon_L).opts(color='green', line_width=1.5, line_dash='dashed', alpha=0.8)
                hline_eps_L_neg = hv.HLine(-epsilon_L).opts(color='green', line_width=1.5, line_dash='dashed', alpha=0.8)

                eps_overlay = eps_curve * hline_eps_R_pos * hline_eps_R_neg * hline_eps_L_pos * hline_eps_L_neg
                eps_plot = eps_overlay.opts(
                    width=800, height=200,
                    title=f'Epsilon Dynamics (range: [{eps_min:.6f}, {eps_max:.6f}]), t_max={t_max_plot:.1f}',
                    xlabel='Time', ylabel='ε(t)',
                    show_grid=True,
                    xlim=(0, t_max_plot),
                    ylim=(eps_min, eps_max),
                    default_tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'save']
                )
            
            # CRITICAL: Use .opts(axiswise=True) on the layout to force axis recalculation
            layout = (pop_plot + eps_plot).cols(1).opts(axiswise=True)
            return layout

        # Create DynamicMap
        return hv.DynamicMap(pn.bind(make_plot, 
                            self.dynamics_version_widget,
                            self.epsilon_L_widget,
                            self.epsilon_R_widget,
                            self.t_max_plot_widget))

    
    def get_control_panel(self):
        """Return control panel for dynamics."""
        return pn.Row(
            self.show_toggle,
            self.auto_toggle,
            pn.Spacer(width=20),
            self.eps0_input,
            self.A_input,
            self.generate_button#,
            #sizing_mode='fixed'
        )


