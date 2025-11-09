########################################
# python/app_interferogram_dynamics_class_raw_best.py
########################################

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import rasterize, datashade
import datashader as ds
from datashader.colors import viridis, inferno
import colorcet as cc
import time
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import copy

# Enable Panel extension for Jupyter
pn.extension()
hv.extension('bokeh')

# Import your simulation modules
from simulation import run_simulation
from config import SimRunGridMode, SimRunSingleMode


class SimulationParameters:
    """Manages simulation parameters and their widgets."""
    
    def __init__(self, delta_C_range, GammaL0_range, GammaR0_range, Gamma_eg0_range, 
                 Gamma_phi0_range, N_steps_period_range, N_periods_range, N_periods_avg_range,
                 delta_C_default, GammaL0_default, GammaR0_default, Gamma_eg0_default,
                 Gamma_phi0_default, N_steps_period_default, N_periods_default, N_periods_avg_default):
        
        # Current values
        self.delta_C = delta_C_default
        self.GammaL0 = GammaL0_default
        self.GammaR0 = GammaR0_default
        self.Gamma_eg0 = Gamma_eg0_default
        self.Gamma_phi0 = Gamma_phi0_default
        self.N_steps_period = N_steps_period_default
        self.N_periods = N_periods_default
        self.N_periods_avg = N_periods_avg_default
        
        # Store ranges
        self.delta_C_range = delta_C_range
        self.GammaL0_range = GammaL0_range
        self.GammaR0_range = GammaR0_range
        self.Gamma_eg0_range = Gamma_eg0_range
        self.Gamma_phi0_range = Gamma_phi0_range
        self.N_steps_period_range = N_steps_period_range
        self.N_periods_range = N_periods_range
        self.N_periods_avg_range = N_periods_avg_range
        
        # Create widgets
        self._create_widgets()
    
    def _create_widgets(self):
        """Create parameter widgets."""
        
        # Continuous parameters
        self.delta_C_slider = pn.widgets.FloatSlider(
            name='delta_C', start=self.delta_C_range[0], end=self.delta_C_range[1],
            value=self.delta_C, step=(self.delta_C_range[1] - self.delta_C_range[0]) / 1000,
            format='0.5e'
        )
        self.delta_C_input = pn.widgets.FloatInput(value=self.delta_C, format='0.5e', width=100)
        
        self.GammaL0_slider = pn.widgets.FloatSlider(
            name='GammaL0', start=self.GammaL0_range[0], end=self.GammaL0_range[1],
            value=self.GammaL0, step=0.5
        )
        self.GammaL0_input = pn.widgets.FloatInput(value=self.GammaL0, width=100)
        
        self.GammaR0_slider = pn.widgets.FloatSlider(
            name='GammaR0', start=self.GammaR0_range[0], end=self.GammaR0_range[1],
            value=self.GammaR0, step=0.1
        )
        self.GammaR0_input = pn.widgets.FloatInput(value=self.GammaR0, width=100)
        
        self.Gamma_eg0_slider = pn.widgets.FloatSlider(
            name='Gamma_eg0', start=self.Gamma_eg0_range[0], end=self.Gamma_eg0_range[1],
            value=self.Gamma_eg0, step=0.1
        )
        self.Gamma_eg0_input = pn.widgets.FloatInput(value=self.Gamma_eg0, width=100)
        
        self.Gamma_phi0_slider = pn.widgets.FloatSlider(
            name='Gamma_phi0', start=self.Gamma_phi0_range[0], end=self.Gamma_phi0_range[1],
            value=self.Gamma_phi0, step=0.1
        )
        self.Gamma_phi0_input = pn.widgets.FloatInput(value=self.Gamma_phi0, width=100)
        
        # Discrete parameters
        self.N_steps_period_slider = pn.widgets.IntSlider(
            name='N_steps_period', start=self.N_steps_period_range[0], 
            end=self.N_steps_period_range[1], value=self.N_steps_period, step=1
        )
        self.N_steps_period_input = pn.widgets.IntInput(value=self.N_steps_period, width=100)
        
        self.N_periods_slider = pn.widgets.IntSlider(
            name='N_periods', start=self.N_periods_range[0], end=self.N_periods_range[1],
            value=self.N_periods, step=1
        )
        self.N_periods_input = pn.widgets.IntInput(value=self.N_periods, width=100)
        
        self.N_periods_avg_slider = pn.widgets.IntSlider(
            name='N_periods_avg', start=self.N_periods_avg_range[0], 
            end=self.N_periods_avg_range[1], value=self.N_periods_avg, step=1
        )
        self.N_periods_avg_input = pn.widgets.IntInput(value=self.N_periods_avg, width=100)
        
        # Link sliders and inputs
        self._link_all_widgets()
    
    def _link_all_widgets(self):
        """Link all slider-input pairs."""
        for param in ['delta_C', 'GammaL0', 'GammaR0', 'Gamma_eg0', 'Gamma_phi0',
                     'N_steps_period', 'N_periods', 'N_periods_avg']:
            self._link_slider_input(param)
    
    def _link_slider_input(self, param_name):
        """Link a slider and input box bidirectionally."""
        slider = getattr(self, f'{param_name}_slider')
        input_box = getattr(self, f'{param_name}_input')
        
        def slider_to_input(event):
            input_box.value = event.new
        
        def input_to_slider(event):
            if event.new is not None:
                val = max(slider.start, min(slider.end, event.new))
                slider.value = val
                if val != event.new:
                    input_box.value = val
        
        slider.param.watch(slider_to_input, 'value')
        input_box.param.watch(input_to_slider, 'value')
    
    def update_from_sliders(self):
        """Update internal values from slider values."""
        self.delta_C = self.delta_C_slider.value
        self.GammaL0 = self.GammaL0_slider.value
        self.GammaR0 = self.GammaR0_slider.value
        self.Gamma_eg0 = self.Gamma_eg0_slider.value
        self.Gamma_phi0 = self.Gamma_phi0_slider.value
        self.N_steps_period = self.N_steps_period_slider.value
        self.N_periods = self.N_periods_slider.value
        self.N_periods_avg = self.N_periods_avg_slider.value
    
    def get_simrun_kwargs(self, platform_type, repo_path, **extra_kwargs):
        """Get kwargs for SimRunGridMode or SimRunSingleMode."""
        base_kwargs = {
            'delta_C': self.delta_C,
            'GammaL0': self.GammaL0,
            'GammaR0': self.GammaR0,
            'Gamma_eg0': self.Gamma_eg0,
            'Gamma_phi0': self.Gamma_phi0,
            'N_steps_period': self.N_steps_period,
            'N_periods': self.N_periods,
            'N_periods_avg': self.N_periods_avg,
            'platform_type': platform_type,
            'repo_path': repo_path
        }
        base_kwargs.update(extra_kwargs)
        return base_kwargs
    
    def get_control_panel(self):
        """Return Panel layout with all parameter controls."""
        return pn.Column(
            "### Physical Parameters",
            pn.Row(self.delta_C_slider, self.delta_C_input),
            pn.Row(self.GammaL0_slider, self.GammaL0_input),
            pn.Row(self.GammaR0_slider, self.GammaR0_input),
            pn.Row(self.Gamma_eg0_slider, self.Gamma_eg0_input),
            pn.Row(self.Gamma_phi0_slider, self.Gamma_phi0_input),
            pn.layout.Divider(),
            "### Time Parameters",
            pn.Row(self.N_steps_period_slider, self.N_steps_period_input),
            pn.Row(self.N_periods_slider, self.N_periods_input),
            pn.Row(self.N_periods_avg_slider, self.N_periods_avg_input),
        )


class InterferogramPlot:
    """Manages the interferogram plot and its data."""
    
    def __init__(self, eps0_min, eps0_max, A_min, A_max, N_points_target,
                 dC_default_thresholds, cmap_name='fire'):
        
        self.eps0_min = eps0_min
        self.eps0_max = eps0_max
        self.A_min = A_min
        self.A_max = A_max
        self.N_points_target = N_points_target
        self.cmap_name = cmap_name
        self.dC_default_thresholds = dC_default_thresholds
        
        # Level labels
        self.level_labels = ["00", "01", "10", "11", "C", "dC"]
        
        # Data storage
        self.avg_grids = {label: None for label in self.level_labels}
        self.eps0_grid = None
        self.A_grid = None
        self.data_version = 0
        
        # Marker state
        self.marker_eps0 = None
        self.marker_A = None
        self.marker_version = 0
        
        # Color limit storage per level
        self._clim_values_per_level = {}
        self._current_level = None
        
        # Create widgets
        self._create_widgets()
    
    def _create_widgets(self):
        """Create interferogram-specific widgets."""
        
        self.level_selector = pn.widgets.RadioButtonGroup(
            name='Display Level',
            options=self.level_labels,
            value='dC',
            button_type='primary'
        )
        
        self.clim_low_slider = pn.widgets.FloatSlider(
            name='Color Min',
            start=-abs(2*self.dC_default_thresholds[0]),
            end=abs(2*self.dC_default_thresholds[0]),
            value=self.dC_default_thresholds[0],
            step=0.01
        )
        self.clim_low_input = pn.widgets.FloatInput(
            value=self.dC_default_thresholds[0], width=100
        )
        
        self.clim_high_slider = pn.widgets.FloatSlider(
            name='Color Max',
            start=-abs(2*self.dC_default_thresholds[1]),
            end=abs(2*self.dC_default_thresholds[1]),
            value=self.dC_default_thresholds[1],
            step=0.01
        )
        self.clim_high_input = pn.widgets.FloatInput(
            value=self.dC_default_thresholds[1], width=100
        )
        
        # Link color sliders
        self._link_slider_input('clim_low')
        self._link_slider_input('clim_high')
        
        # Version widgets (hidden)
        self.data_version_widget = pn.widgets.IntInput(value=0, visible=False)
        self.marker_version_widget = pn.widgets.IntInput(value=0, visible=False)
        
        # Watch level changes
        self.level_selector.param.watch(self._on_level_change, 'value')
    
    def _link_slider_input(self, param_name):
        """Link slider and input."""
        slider = getattr(self, f'{param_name}_slider')
        input_box = getattr(self, f'{param_name}_input')
        
        def slider_to_input(event):
            input_box.value = event.new
        
        def input_to_slider(event):
            if event.new is not None:
                val = max(slider.start, min(slider.end, event.new))
                slider.value = val
                if val != event.new:
                    input_box.value = val
        
        slider.param.watch(slider_to_input, 'value')
        input_box.param.watch(input_to_slider, 'value')
    
    def _on_level_change(self, event):
        """Handle level change and update color limits."""
        prev_level = self._current_level
        new_level = event.new
        
        # Save current slider values
        if prev_level is not None:
            self._clim_values_per_level[prev_level] = (
                self.clim_low_slider.value,
                self.clim_high_slider.value
            )
        
        # Set range based on level
        if new_level in ["00", "01", "10", "11"]:
            slider_min, slider_max = 0, 1
        elif new_level == "C":
            slider_min, slider_max = -18, 18
        elif new_level == "dC":
            slider_min = -abs(2 * self.dC_default_thresholds[0])
            slider_max = abs(2 * self.dC_default_thresholds[1])
        else:
            slider_min, slider_max = 0, 1
        
        # Update ranges
        self.clim_low_slider.start = slider_min
        self.clim_low_slider.end = slider_max
        self.clim_high_slider.start = slider_min
        self.clim_high_slider.end = slider_max
        
        # Restore or set default values
        if new_level in self._clim_values_per_level:
            low_val, high_val = self._clim_values_per_level[new_level]
        else:
            low_val, high_val = slider_min, slider_max
        
        self.clim_low_slider.value = low_val
        self.clim_high_slider.value = high_val
        self.clim_low_input.value = low_val
        self.clim_high_input.value = high_val
        
        self._current_level = new_level
    
    def update_data(self, eps0_grid, A_grid, rho_avg_cdc_3d):
        """Update interferogram data."""
        self.eps0_grid = eps0_grid
        self.A_grid = A_grid
        
        for i, label in enumerate(self.level_labels):
            self.avg_grids[label] = rho_avg_cdc_3d[i]
        
        # Update version AFTER all data is set
        self.data_version += 1
        if hasattr(self, 'data_version_widget'):
            self.data_version_widget.value = self.data_version
    
    def set_marker(self, eps0, A):
        """Set marker position."""
        # Only set marker if within range
        if (self.eps0_min <= eps0 <= self.eps0_max and 
            self.A_min <= A <= self.A_max):
            self.marker_eps0 = eps0
            self.marker_A = A
        else:
            self.marker_eps0 = None
            self.marker_A = None
        
        # DON'T update version here - it will be updated after dynamics computation
    
    def clear_marker(self):
        """Clear marker."""
        self.marker_eps0 = None
        self.marker_A = None
        
        # DON'T update version here - it will be updated after dynamics computation
    
    def get_level_data(self, level):
        """Get data for a specific level."""
        if level in self.avg_grids:
            return self.avg_grids[level]
        return self.avg_grids["00"]
    
    def create_plot(self):
        """Create the interactive interferogram plot."""
        
        def make_plot(level, clim_low, clim_high, data_version, marker_version):
            data = self.get_level_data(level)
            
            # Check if data AND grids are available
            if data is None or self.eps0_grid is None or self.A_grid is None:
                # Return placeholder with proper dimensions and range
                return hv.Image(
                    (np.array([self.eps0_min, self.eps0_max]), 
                     np.array([self.A_min, self.A_max]), 
                     np.zeros((2, 2))),
                    kdims=['eps0', 'A'],
                    vdims=['value']
                ).opts(
                    cmap=self.cmap_name,
                    colorbar=True,
                    width=800, height=600,
                    title='Interactive Interferogram (loading...)',
                    xlabel='eps0', ylabel='A',
                    xlim=(self.eps0_min, self.eps0_max),
                    ylim=(self.A_min, self.A_max)
                )
            
            # IMPORTANT: Copy marker values at the start to avoid race conditions
            marker_eps0_local = self.marker_eps0
            marker_A_local = self.marker_A
            
            img = hv.Image(
                (self.eps0_grid, self.A_grid, data),
                kdims=['eps0', 'A'],
                vdims=['value']
            ).opts(
                cmap=self.cmap_name,
                colorbar=True,
                width=800,
                height=600,
                clim=(clim_low, clim_high),
                title=f'Interactive Interferogram - Level: {level}',
                xlabel='eps0',
                ylabel='A',
                tools=['hover', 'tap'],
                default_tools=['pan', 'wheel_zoom', 'box_zoom', 'reset']
            )
            
            # Add marker if present - use LOCAL copies
            if marker_eps0_local is not None and marker_A_local is not None:
                vline = hv.VLine(marker_eps0_local).opts(color='blue', line_width=2, line_dash='solid')
                hline = hv.HLine(marker_A_local).opts(color='blue', line_width=2, line_dash='solid')
                img = img * vline * hline
            
            return img
        
        return hv.DynamicMap(
            pn.bind(make_plot,
                    self.level_selector,
                    self.clim_low_slider,
                    self.clim_high_slider,
                    self.data_version_widget,
                    self.marker_version_widget)
        )
    
    def get_control_panel(self):
        """Return control panel for interferogram."""
        return pn.Row(
            pn.Column("**Level:**", self.level_selector),
            pn.Column(
                pn.Row(self.clim_low_slider, self.clim_low_input),
                pn.Row(self.clim_high_slider, self.clim_high_input)
            )
        )


class DynamicsPlot:
    """Manages the dynamics plot and computation."""
    
    def __init__(self, eps0_min, eps0_max, A_min, A_max):
        
        self.eps0_min = eps0_min
        self.eps0_max = eps0_max
        self.A_min = A_min
        self.A_max = A_max
        
        # Dynamics data
        self.time_dynamics = None
        self.eps_dynamics = None
        self.rho_dynamics = None
        self.current_eps0 = None
        self.current_A = None
        self.computation_time = 0
        
        # State
        self.enabled = False
        self.auto_update = False
        self.hover_active = False
        self.computing = False
        self.click_count = 0
        
        # Version tracking
        self.dynamics_version = 0
        
        # Create widgets
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
            width=300,
            sizing_mode='fixed'
        )
        
        self.dynamics_version_widget = pn.widgets.IntInput(value=0, visible=False)
        
        # Watch toggles
        self.show_toggle.param.watch(self._on_show_toggle, 'value')
        self.auto_toggle.param.watch(self._on_auto_toggle, 'value')
    
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
    
    def compute(self, eps0, A, sim_params, platform_type, repo_path, log_callback=None, marker_update_callback=None):
        """Compute dynamics for given coordinates."""
        
        if self.computing:
            return
        
        self.computing = True
        self.current_eps0 = eps0
        self.current_A = A
        
        # Update inputs
        self.eps0_input.value = eps0
        self.A_input.value = A
        
        # Capture output
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
                
                self.status_text.object = f"**Dynamics:** ✅ Ready ({self.computation_time:.2f}s) | eps0={eps0:.6f}, A={A:.6f}"
            except Exception as e:
                self.status_text.object = f"**Dynamics:** ❌ Error: {str(e)}"
                self.computing = False
                return
        
        # Log callback
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
        
        # Trigger marker update callback FIRST
        if marker_update_callback:
            marker_update_callback()
        
        # Update version AFTER everything else (including marker callback)
        self.dynamics_version += 1
        if hasattr(self, 'dynamics_version_widget'):
            self.dynamics_version_widget.value = self.dynamics_version
        
        self.computing = False
    
    def create_plot(self):
        """Create the dynamics plot."""
        
        def make_plot(version):
            if self.time_dynamics is None or self.rho_dynamics is None:
                empty_pop = hv.Curve([(0, 0)], kdims=['time'], vdims=['population']).opts(
                    width=800, height=350,
                    title='Population Dynamics (click on interferogram)',
                    xlabel='Time', ylabel='Population',
                    show_grid=True,
                    xlim=(0, 1), ylim=(0, 1)
                )
                empty_eps = hv.Curve([(0, 0)], kdims=['time'], vdims=['epsilon']).opts(
                    width=800, height=200,
                    title='Epsilon Dynamics',
                    xlabel='Time', ylabel='ε(t)',
                    show_grid=True,
                    xlim=(0, 1), ylim=(0, 1)
                )
                # Return empty plots using Layout
                return hv.Layout([empty_pop, empty_eps]).cols(1)
            
            # Population plot
            time = self.time_dynamics
            p00, p01, p10, p11 = [self.rho_dynamics[:, i] for i in range(4)]
            
            pop_max = np.max([p00.max(), p01.max(), p10.max(), p11.max()])
            pop_ylim = (0, pop_max * 1.1)
            
            curves = [
                hv.Curve((time, p00), label='p00').opts(color='red', line_width=1.5),
                hv.Curve((time, p01), label='p01').opts(color='blue', line_width=1.5),
                hv.Curve((time, p10), label='p10').opts(color='green', line_width=1.5),
                hv.Curve((time, p11), label='p11').opts(color='orange', line_width=1.5)
            ]
            
            pop_overlay = curves[0] * curves[1] * curves[2] * curves[3]
            pop_plot = pop_overlay.opts(
                width=800, height=350,
                title=f'Population Dynamics (eps0={self.current_eps0:.6f}, A={self.current_A:.6f})',
                xlabel='Time', ylabel='Population',
                show_grid=True, legend_position='right',
                ylim=pop_ylim
            )
            
            # Epsilon plot
            eps_min = self.current_eps0 - self.current_A * 1.1
            eps_max = self.current_eps0 + self.current_A * 1.1
            
            eps_plot = hv.Curve((time, self.eps_dynamics)).opts(
                color='purple', line_width=1.5,
                width=800, height=200,
                title='Epsilon Dynamics',
                xlabel='Time', ylabel='ε(t)',
                show_grid=True,
                ylim=(eps_min, eps_max)
            )
            
            # Return actual plots using Layout to ensure correct ordering
            return hv.Layout([pop_plot, eps_plot]).cols(1)
        
        return hv.DynamicMap(pn.bind(make_plot, self.dynamics_version_widget))
    
    def get_control_panel(self):
        """Return control panel for dynamics."""
        return pn.Row(
            self.show_toggle,
            self.auto_toggle,
            pn.Spacer(width=20),
            self.eps0_input,
            self.A_input,
            self.generate_button,
            sizing_mode='fixed'
        )


class InteractiveInterferogramDynamics:
    """Main coordinator class for the interactive dashboard."""
    
    def __init__(self, eps0_min, eps0_max, A_min, A_max, N_points_target,
                 delta_C_range, GammaL0_range, GammaR0_range, Gamma_eg0_range, Gamma_phi0_range,
                 N_steps_period_array, N_periods_array, N_periods_avg_array,
                 delta_C_default, GammaL0_default, GammaR0_default, Gamma_eg0_default,
                 Gamma_phi0_default, N_steps_period_default, N_periods_default, N_periods_avg_default,
                 dC_default_thresholds,
                 platform_type,
                 repo_path,
                 cmap_name='fire'):
        
        self.platform_type = platform_type
        self.repo_path = repo_path
        self.eps0_min = eps0_min
        self.eps0_max = eps0_max
        self.A_min = A_min
        self.A_max = A_max
        self.N_points_target = N_points_target
        
        # Convert arrays to tuples if needed
        N_steps_period_range = N_steps_period_array if isinstance(N_steps_period_array, tuple) else (int(N_steps_period_array[0]), int(N_steps_period_array[-1]))
        N_periods_range = N_periods_array if isinstance(N_periods_array, tuple) else (int(N_periods_array[0]), int(N_periods_array[-1]))
        N_periods_avg_range = N_periods_avg_array if isinstance(N_periods_avg_array, tuple) else (int(N_periods_avg_array[0]), int(N_periods_avg_array[-1]))
        
        # Create component objects
        self.sim_params = SimulationParameters(
            delta_C_range, GammaL0_range, GammaR0_range, Gamma_eg0_range, Gamma_phi0_range,
            N_steps_period_range, N_periods_range, N_periods_avg_range,
            delta_C_default, GammaL0_default, GammaR0_default, Gamma_eg0_default,
            Gamma_phi0_default, N_steps_period_default, N_periods_default, N_periods_avg_default
        )
        
        self.interferogram = InterferogramPlot(
            eps0_min, eps0_max, A_min, A_max, N_points_target,
            dC_default_thresholds, cmap_name
        )
        
        self.dynamics = DynamicsPlot(eps0_min, eps0_max, A_min, A_max)
        
        # Control widgets
        self.auto_update_enabled = False
        self._is_generating = False
        
        # Create control widgets FIRST (they check hasattr)
        self._create_control_widgets()
        
        # THEN generate initial data
        self._generate_interferogram_data()
    
    def _create_control_widgets(self):
        """Create main control widgets."""
        
        self.update_button = pn.widgets.Button(
            name='🔄 Regenerate Data',
            button_type='success',
            width=180
        )
        
        self.auto_update_toggle = pn.widgets.Toggle(
            name='Auto Update',
            value=False,
            button_type='warning',
            width=120
        )
        
        self.status_text = pn.pane.Markdown(
            "**Status:** Ready",
            width=300,
            sizing_mode='fixed'
        )
        
        self.timing_text = pn.pane.Markdown(
            "**Last computation:** N/A",
            width=300,
            sizing_mode='fixed'
        )
        
        self.log_display = pn.pane.Markdown(
            "**Log:** Waiting for first computation...",
            width=650,
            height=150,
            styles={'resize': 'both', 'overflow': 'auto', 'background': '#f5f5f5', 
                   'padding': '10px', 'font-family': 'monospace', 'font-size': '11px', 
                   'border': '1px solid #ddd'},
            sizing_mode='fixed'
        )
        
        # Watch auto-update toggle
        self.auto_update_toggle.param.watch(self._on_auto_update_toggle, 'value')
        
        # Watch parameter sliders for auto-update
        for param in ['delta_C', 'GammaL0', 'GammaR0', 'Gamma_eg0', 'Gamma_phi0',
                     'N_steps_period', 'N_periods', 'N_periods_avg']:
            slider = getattr(self.sim_params, f'{param}_slider')
            slider.param.watch(self._on_parameter_change, 'value')
    
    def _on_auto_update_toggle(self, event):
        """Handle auto-update toggle."""
        self.auto_update_enabled = event.new
        if self.auto_update_enabled:
            self.update_button.name = '🔄 Update (Auto On)'
            self.update_button.button_type = 'light'
        else:
            self.update_button.name = '🔄 Regenerate Data'
            self.update_button.button_type = 'success'
    
    def _on_parameter_change(self, event):
        """Handle parameter slider changes."""
        if self.auto_update_enabled:
            self._update_and_regenerate()
    
    def _update_and_regenerate(self, event=None):
        """Update parameters and regenerate interferogram."""
        self.sim_params.update_from_sliders()
        self._generate_interferogram_data()
    
    def _generate_interferogram_data(self):
        """Generate interferogram data."""
        
        if self._is_generating:
            return
        
        self._is_generating = True
        
        captured_stdout = StringIO()
        captured_stderr = StringIO()
        
        with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
            if hasattr(self, 'status_text'):
                self.status_text.object = "**Status:** 🔄 Generating data..."
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
            
            self.interferogram.update_data(eps0_grid, A_grid, rho_avg_cdc_3d)
            
            end_time = time.perf_counter()
            elapsed = end_time - start_time
        
        if hasattr(self, 'status_text'):
            self.status_text.object = "**Status:** ✅ Data ready"
        if hasattr(self, 'timing_text'):
            self.timing_text.object = f"**Last computation:** {elapsed:.2f} seconds"
        
        # Build log
        log_text = f"**Computation completed in {elapsed:.2f}s**\n\n"
        log_text += "**Parameters:**\n"
        log_text += f"- delta_C = {self.sim_params.delta_C:.6e}\n"
        log_text += f"- GammaL0 = {self.sim_params.GammaL0}, GammaR0 = {self.sim_params.GammaR0}\n"
        log_text += f"- Gamma_eg0 = {self.sim_params.Gamma_eg0}, Gamma_phi0 = {self.sim_params.Gamma_phi0}\n"
        log_text += f"- N_steps = {self.sim_params.N_steps_period}, N_periods = {self.sim_params.N_periods}, N_avg = {self.sim_params.N_periods_avg}\n\n"
        
        stdout_content = captured_stdout.getvalue()
        stderr_content = captured_stderr.getvalue()
        
        if stdout_content:
            log_text += f"**Output:**\n```\n{stdout_content}\n```\n"
        if stderr_content:
            log_text += f"**Warnings:**\n```\n{stderr_content}\n```\n"

        if hasattr(self, 'log_display'):
            self.log_display.object = log_text
        
        self._is_generating = False
    
    def _on_interferogram_click(self, x, y):
        """Handle click on interferogram."""
        
        if not self.dynamics.enabled or x is None or y is None:
            return
        
        eps0, A = x, y
        
        if self.dynamics.auto_update:
            # Cycle through click states
            if self.dynamics.click_count % 2 == 0:
                # First click: Start hover mode, clear marker
                self.dynamics.hover_active = True
                self.interferogram.marker_eps0 = None
                self.interferogram.marker_A = None
                self._generate_dynamics(eps0, A)
            else:
                # Second click: Stop hover mode, set marker
                self.dynamics.hover_active = False
                if (self.eps0_min <= eps0 <= self.eps0_max and 
                    self.A_min <= A <= self.A_max):
                    self.interferogram.marker_eps0 = eps0
                    self.interferogram.marker_A = A
                else:
                    self.interferogram.marker_eps0 = None
                    self.interferogram.marker_A = None
                self._generate_dynamics(eps0, A)
            
            self.dynamics.click_count += 1
        else:
            # Manual mode: always set marker and generate
            if (self.eps0_min <= eps0 <= self.eps0_max and 
                self.A_min <= A <= self.A_max):
                self.interferogram.marker_eps0 = eps0
                self.interferogram.marker_A = A
            else:
                self.interferogram.marker_eps0 = None
                self.interferogram.marker_A = None
            self._generate_dynamics(eps0, A)
        
        # DON'T update marker_version here - remove these lines:
        # self.interferogram.marker_version += 1
        # if hasattr(self.interferogram, 'marker_version_widget'):
        #     self.interferogram.marker_version_widget.value = self.interferogram.marker_version
    
    def _on_interferogram_hover(self, x, y):
        """Handle hover on interferogram."""
        
        if not self.dynamics.enabled or x is None or y is None:
            return
        
        if not self.dynamics.auto_update or not self.dynamics.hover_active:
            return
        
        self._generate_dynamics(x, y)
    
    def _generate_dynamics(self, eps0, A):
        """Generate dynamics for given coordinates."""
        
        def log_callback(text):
            self.log_display.object = text
            self.log_display.param.trigger('object')
        
        def marker_update_callback():
            # Always update marker version after dynamics completes
            self.interferogram.marker_version += 1
            if hasattr(self.interferogram, 'marker_version_widget'):
                self.interferogram.marker_version_widget.value = self.interferogram.marker_version
        
        self.dynamics.compute(eps0, A, self.sim_params, 
                            self.platform_type, self.repo_path, 
                            log_callback, marker_update_callback)
    
    def _on_manual_dynamics_generate(self, event):
        """Handle manual coordinate entry for dynamics."""
        
        eps0 = self.dynamics.eps0_input.value
        A = self.dynamics.A_input.value
        
        # Set marker (only if in range)
        self.interferogram.set_marker(eps0, A)
        
        # Turn off auto-update mode
        self.dynamics.auto_toggle.value = False
        self.dynamics.hover_active = False
        
        # Generate dynamics
        self._generate_dynamics(eps0, A)
    
    def create_dashboard(self):
        """Create the complete Panel dashboard."""
        
        # Connect buttons
        self.update_button.on_click(self._update_and_regenerate)
        self.dynamics.generate_button.on_click(self._on_manual_dynamics_generate)
        
        # Create interferogram plot with interaction
        interferogram_dmap = self.interferogram.create_plot()
        
        # Create tap and hover streams and subscribe them to the plot
        self.tap_stream = hv.streams.Tap(source=interferogram_dmap, x=None, y=None)
        self.hover_stream = hv.streams.PointerXY(source=interferogram_dmap, x=None, y=None)
        
        # Watch for stream events - watch only one parameter to avoid double triggers
        def handle_tap(event):
            if self.tap_stream.x is not None and self.tap_stream.y is not None:
                self._on_interferogram_click(self.tap_stream.x, self.tap_stream.y)
        
        def handle_hover(event):
            if self.hover_stream.x is not None and self.hover_stream.y is not None:
                self._on_interferogram_hover(self.hover_stream.x, self.hover_stream.y)
        
        # Only watch 'x' to avoid double-triggering (y changes at the same time)
        self.tap_stream.param.watch(handle_tap, 'x')
        self.hover_stream.param.watch(handle_hover, 'x')
        
        # Sidebar
        sidebar = pn.Column(
            "## 🎛️ Simulation Parameters",
            pn.layout.Divider(),
            self.sim_params.get_control_panel(),
            pn.layout.Divider(),
            pn.Row(self.update_button, self.auto_update_toggle),
            self.status_text,
            self.timing_text,
            width=500,
            sizing_mode='fixed'
        )
        
        # Interferogram section
        interferogram_section = pn.Column(
            self.interferogram.get_control_panel(),
            interferogram_dmap,
            sizing_mode='fixed'
        )
        
        # Dynamics section (conditionally shown)
        def dynamics_section_visible(show_dynamics):
            if show_dynamics:
                return pn.Column(
                    pn.layout.Divider(),
                    "### Dynamics Plot",
                    self.dynamics.get_control_panel(),
                    self.dynamics.status_text,
                    self.dynamics.create_plot(),
                    sizing_mode='fixed'
                )
            else:
                return pn.Column(
                    pn.layout.Divider(),
                    "### Dynamics Plot",
                    self.dynamics.get_control_panel(),
                    sizing_mode='fixed'
                )
        
        dynamics_section_dynamic = pn.bind(dynamics_section_visible, self.dynamics.show_toggle)
        
        # Main plot area
        plot_area = pn.Column(
            interferogram_section,
            dynamics_section_dynamic,
            pn.layout.Divider(),
            "### Computation Log",
            self.log_display,
            sizing_mode='fixed'
        )
        
        # Complete dashboard
        dashboard = pn.Row(
            sidebar,
            plot_area,
            sizing_mode='fixed'
        )
        
        return dashboard
    
