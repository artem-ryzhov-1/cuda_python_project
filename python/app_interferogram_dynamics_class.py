########################################
# python/app_interferogram_dynamics_class.py
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

# Import simulation modules
from simulation import run_simulation
from config import SimRunGridMode, SimRunSingleMode, SimRunGridSingleMode


import matplotlib.pyplot as plt

# Get default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class SimulationParameters:
    """Manages simulation parameters and their widgets."""
    
    def __init__(self, delta_C_range, GammaL0_range, GammaR0_range, Gamma_eg0_range, 
                 Gamma_phi0_range, sigma_eps_range, N_steps_period_range, N_periods_range, 
                 N_periods_avg_range, N_samples_noise_range,
                 delta_C_default, GammaL0_default, GammaR0_default, Gamma_eg0_default,
                 Gamma_phi0_default, sigma_eps_default, nu, N_steps_period_default, N_periods_default, 
                 N_periods_avg_default, N_samples_noise_default):
        
        # Current values
        self.delta_C = delta_C_default
        self.GammaL0 = GammaL0_default
        self.GammaR0 = GammaR0_default
        self.Gamma_eg0 = Gamma_eg0_default
        self.Gamma_phi0 = Gamma_phi0_default
        self.sigma_eps = sigma_eps_default
        self.N_steps_period = N_steps_period_default
        self.N_periods = N_periods_default
        self.N_periods_avg = N_periods_avg_default
        self.N_samples_noise = N_samples_noise_default
        self.quasi_static = False
        
        # Store ranges
        self.delta_C_range = delta_C_range
        self.GammaL0_range = GammaL0_range
        self.GammaR0_range = GammaR0_range
        self.Gamma_eg0_range = Gamma_eg0_range
        self.Gamma_phi0_range = Gamma_phi0_range
        self.sigma_eps_range = sigma_eps_range
        self.N_steps_period_range = N_steps_period_range
        self.N_periods_range = N_periods_range
        self.N_periods_avg_range = N_periods_avg_range
        self.N_samples_noise_range = N_samples_noise_range
        
        # Constant values
        self.nu = nu
        
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
        
        self.sigma_eps_slider = pn.widgets.FloatSlider(
            name='sigma_eps', start=self.sigma_eps_range[0], end=self.sigma_eps_range[1],
            value=self.sigma_eps, step=0.00001, disabled=True
        )
        self.sigma_eps_input = pn.widgets.FloatInput(value=self.sigma_eps, width=100, disabled=True)
        
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
        
        self.N_samples_noise_slider = pn.widgets.IntSlider(
            name='N_samples_noise', start=self.N_samples_noise_range[0], 
            end=self.N_samples_noise_range[1], value=self.N_samples_noise, step=1, disabled=True
        )
        self.N_samples_noise_input = pn.widgets.IntInput(value=self.N_samples_noise, width=100, disabled=True)
        
        # Quasi-static toggle
        self.quasi_static_toggle = pn.widgets.Toggle(
            name='Quasi-static',
            value=False,
            button_type='primary',
            width=120
        )
        
        # Link sliders and inputs
        self._link_all_widgets()
        
        # Watch quasi-static toggle
        self.quasi_static_toggle.param.watch(self._on_quasi_static_toggle, 'value')
    
    def _link_all_widgets(self):
        """Link all slider-input pairs."""
        for param in ['delta_C', 'GammaL0', 'GammaR0', 'Gamma_eg0', 'Gamma_phi0', 'sigma_eps',
                     'N_steps_period', 'N_periods', 'N_periods_avg', 'N_samples_noise']:
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
    
    def _on_quasi_static_toggle(self, event):
        """Handle quasi-static toggle changes."""
        self.quasi_static = event.new
        
        # Update widget states
        self.Gamma_phi0_slider.disabled = event.new
        self.Gamma_phi0_input.disabled = event.new
        self.sigma_eps_slider.disabled = not event.new
        self.sigma_eps_input.disabled = not event.new
        self.N_samples_noise_slider.disabled = not event.new
        self.N_samples_noise_input.disabled = not event.new
    
    def update_from_sliders(self):
        """Update internal values from slider values."""
        self.delta_C = self.delta_C_slider.value
        self.GammaL0 = self.GammaL0_slider.value
        self.GammaR0 = self.GammaR0_slider.value
        self.Gamma_eg0 = self.Gamma_eg0_slider.value
        self.Gamma_phi0 = self.Gamma_phi0_slider.value
        self.sigma_eps = self.sigma_eps_slider.value
        self.N_steps_period = self.N_steps_period_slider.value
        self.N_periods = self.N_periods_slider.value
        self.N_periods_avg = self.N_periods_avg_slider.value
        self.N_samples_noise = self.N_samples_noise_slider.value
        self.quasi_static = self.quasi_static_toggle.value
    
    def get_simrun_kwargs(self, platform_type, repo_path, **extra_kwargs):
        """Get kwargs for SimRunGridMode or SimRunSingleMode."""
        base_kwargs = {
            'delta_C': self.delta_C,
            'GammaL0': self.GammaL0,
            'GammaR0': self.GammaR0,
            'Gamma_eg0': self.Gamma_eg0,
            'Gamma_phi0': None if self.quasi_static else self.Gamma_phi0,
            'nu': self.nu,
            'N_steps_period': self.N_steps_period,
            'N_periods': self.N_periods,
            'N_periods_avg': self.N_periods_avg,
            'quasi_static_ensemble_dephasing_flag': self.quasi_static,
            'sigma_eps': self.sigma_eps if self.quasi_static else None,
            'N_samples_noise': self.N_samples_noise if self.quasi_static else None,
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
            pn.Row(self.sigma_eps_slider, self.sigma_eps_input),
            pn.layout.Divider(),
            "### Time Parameters",
            pn.Row(self.N_steps_period_slider, self.N_steps_period_input),
            pn.Row(self.N_periods_slider, self.N_periods_input),
            pn.Row(self.N_periods_avg_slider, self.N_periods_avg_input),
            pn.Row(self.N_samples_noise_slider, self.N_samples_noise_input),
            pn.layout.Divider(),
            self.quasi_static_toggle
        )




class InterferogramPlot:
    """Manages the interferogram plot and its data."""
    
    def __init__(self, eps0_min, eps0_max, A_min, A_max, N_points_target,
                 dC_default_thresholds, cmap_name, render_mode):
        
        self.eps0_min = eps0_min
        self.eps0_max = eps0_max
        self.A_min = A_min
        self.A_max = A_max
        self.N_points_target = N_points_target
        self.cmap_name = cmap_name
        self.dC_default_thresholds = dC_default_thresholds
        
        # Render mode: 'vector', 'raster_static', 'raster_static_gpu', 'raster_dynamic', or 'raster_dynamic_gpu'
        valid_modes = ['vector', 'raster_static', 'raster_static_gpu', 'raster_dynamic', 'raster_dynamic_gpu']
        if render_mode not in valid_modes:
            raise ValueError(f"render_mode must be one of {valid_modes}, got {render_mode}")
        
        self.render_mode = render_mode
        self.gpu_enabled = (render_mode == 'raster_dynamic_gpu')
                            
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
        
        # Version widgets - styled invisible for panel serve compatibility
        self.data_version_widget = pn.widgets.IntInput(
            value=0,
            visible=False  # Better than CSS tricks
        )
        self.marker_version_widget = pn.widgets.IntInput(
            value=0,
            visible=False
        )
        
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
        
        self.data_version += 1
        if hasattr(self, 'data_version_widget'):
            self.data_version_widget.value = self.data_version
        
        self.marker_version += 1
        if hasattr(self, 'marker_version_widget'):
            self.marker_version_widget.value = self.marker_version
    
    def set_marker(self, eps0, A):
        """Set marker position."""
        if (self.eps0_min <= eps0 <= self.eps0_max and 
            self.A_min <= A <= self.A_max):
            self.marker_eps0 = eps0
            self.marker_A = A
        else:
            self.marker_eps0 = None
            self.marker_A = None
        
        self.marker_version += 1
        if hasattr(self, 'marker_version_widget'):
            self.marker_version_widget.value = self.marker_version
    
    def clear_marker(self):
        """Clear marker."""
        self.marker_eps0 = None
        self.marker_A = None
        self.marker_version += 1
        if hasattr(self, 'marker_version_widget'):
            self.marker_version_widget.value = self.marker_version
            
    def get_level_data(self, level):
        """Get data for a specific level."""
        if level in self.avg_grids:
            return self.avg_grids[level]
        return self.avg_grids["00"]
    
    def create_plot(self):
        """Create the interactive interferogram plot using selected render mode."""
        
        if self.render_mode == 'vector':
            return self._create_plot_vector()
        elif self.render_mode in ['raster_static', 'raster_static_gpu']:
            return self._create_plot_raster_static()
        elif self.render_mode in ['raster_dynamic', 'raster_dynamic_gpu']:
            return self._create_plot_raster_dynamic()
    
    def _create_plot_vector(self):
        """Vector approach - all points sent to browser."""
        
        def make_plot(level, clim_low, clim_high, data_version, marker_version):
            data = self.get_level_data(level)
            
            if data is None or self.eps0_grid is None or self.A_grid is None:
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
                title=f'Interactive Interferogram - Level: {level} [VECTOR]',
                xlabel='eps0',
                ylabel='A'
            )
            
            if marker_eps0_local is not None and marker_A_local is not None:
                vline = hv.VLine(marker_eps0_local).opts(color='blue', line_width=2, line_dash='solid')
                hline = hv.HLine(marker_A_local).opts(color='blue', line_width=2, line_dash='solid')
            else:
                vline = hv.VLine(self.eps0_min - 1).opts(color='blue', line_width=0, alpha=0)
                hline = hv.HLine(self.A_min - 1).opts(color='blue', line_width=0, alpha=0)
            
            # Create invisible rectangle for capturing interactions
            invisible_rect = hv.Rectangles([(self.eps0_min, self.A_min, self.eps0_max, self.A_max)]).opts(
                alpha=0,
                line_alpha=0,
                tools=['tap', 'hover'],
                default_tools=['pan', 'wheel_zoom', 'box_zoom', 'reset']
            )
            
            return img * invisible_rect * vline * hline
        
        return hv.DynamicMap(
            pn.bind(make_plot,
                    self.level_selector,
                    self.clim_low_slider,
                    self.clim_high_slider,
                    self.data_version_widget,
                    self.marker_version_widget)
        )
    
    def _create_plot_raster_static(self):
        """Raster approach with dynamic=False - pre-rendered, no zoom quality.
        Uses GPU only if gpu_enabled=True."""
        from holoviews.operation.datashader import rasterize
        
        def make_plot(level, clim_low, clim_high, data_version, marker_version):
            data = self.get_level_data(level)
            
            if data is None or self.eps0_grid is None or self.A_grid is None:
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
            
            marker_eps0_local = self.marker_eps0
            marker_A_local = self.marker_A
            
            # Title shows GPU status
            title_suffix = "[RASTER-STATIC-GPU]" if self.gpu_enabled else "[RASTER-STATIC-CPU]"
            
            # Convert to GPU arrays if GPU enabled
            if self.gpu_enabled and GPU_AVAILABLE:
                eps0_array = cp.asarray(self.eps0_grid)
                A_array = cp.asarray(self.A_grid)
                data_array = cp.asarray(data)
            else:
                eps0_array = self.eps0_grid
                A_array = self.A_grid
                data_array = data
            
            img = hv.Image(
                (eps0_array, A_array, data_array),
                kdims=['eps0', 'A']
            )
            
            # Apply rasterization with explicit GPU control
            if self.gpu_enabled:
                img_rasterized = rasterize(
                    img, 
                    aggregator=ds.mean('z'),
                    dynamic=False, 
                    precompute=True
                )
            else:
                img_rasterized = rasterize(
                    img, 
                    aggregator='mean',
                    dynamic=False, 
                    precompute=True
                )
            
            img_rasterized = img_rasterized.opts(
                cmap=self.cmap_name,
                colorbar=True,
                width=800,
                height=600,
                clim=(clim_low, clim_high),
                title=f'Interactive Interferogram - Level: {level} {title_suffix}',
                xlabel='eps0',
                ylabel='A'
            )
            
            if marker_eps0_local is not None and marker_A_local is not None:
                vline = hv.VLine(marker_eps0_local).opts(color='blue', line_width=2, line_dash='solid')
                hline = hv.HLine(marker_A_local).opts(color='blue', line_width=2, line_dash='solid')
            else:
                vline = hv.VLine(self.eps0_min - 1).opts(color='blue', line_width=0, alpha=0)
                hline = hv.HLine(self.A_min - 1).opts(color='blue', line_width=0, alpha=0)
            
            invisible_rect = hv.Rectangles([(self.eps0_min, self.A_min, self.eps0_max, self.A_max)]).opts(
                alpha=0,
                line_alpha=0,
                tools=['tap', 'hover'],
                default_tools=['pan', 'wheel_zoom', 'box_zoom', 'reset']
            )
            
            return img_rasterized * invisible_rect * vline * hline
        
        return hv.DynamicMap(
            pn.bind(make_plot,
                    self.level_selector,
                    self.clim_low_slider,
                    self.clim_high_slider,
                    self.data_version_widget,
                    self.marker_version_widget)
        )
    
    def _create_plot_raster_dynamic(self):
        """Raster approach with dynamic=True - re-aggregates on zoom for quality."""
        
        def make_image(level, clim_low, clim_high, data_version):
            """Just the image, no markers."""
            data = self.get_level_data(level)
            
            if data is None or self.eps0_grid is None or self.A_grid is None:
                return hv.Image(
                    (np.array([self.eps0_min, self.eps0_max]), 
                     np.array([self.A_min, self.A_max]), 
                     np.zeros((2, 2))),
                    kdims=['eps0', 'A']
                ).opts(
                    cmap=self.cmap_name,
                    colorbar=True,
                    width=800, height=600,
                    title='Interactive Interferogram (loading...)',
                    xlabel='eps0', ylabel='A',
                    xlim=(self.eps0_min, self.eps0_max),
                    ylim=(self.A_min, self.A_max)
                )
            
            title_suffix = "[RASTER-DYNAMIC-GPU]" if self.gpu_enabled else "[RASTER-DYNAMIC-CPU]"
            
            img = hv.Image(
                (self.eps0_grid, self.A_grid, data),
                kdims=['eps0', 'A']
            ).opts(
                cmap=self.cmap_name,
                colorbar=True,
                width=800,
                height=600,
                clim=(clim_low, clim_high),
                title=f'Interactive Interferogram - Level: {level} {title_suffix}',
                xlabel='eps0',
                ylabel='A'
            )
            
            return img
        
        def make_markers_and_overlay(marker_version):
            """Markers plus invisible overlay for interactions."""
            invisible_rect = hv.Rectangles([(self.eps0_min, self.A_min, self.eps0_max, self.A_max)]).opts(
                alpha=0,
                line_alpha=0,
                tools=['tap', 'hover'],
                default_tools=['pan', 'wheel_zoom', 'box_zoom', 'reset']
            )
            
            if self.marker_eps0 is not None and self.marker_A is not None:
                vline = hv.VLine(self.marker_eps0).opts(color='blue', line_width=2, line_dash='solid')
                hline = hv.HLine(self.marker_A).opts(color='blue', line_width=2, line_dash='solid')
            else:
                vline = hv.VLine(self.eps0_min - 1).opts(color='blue', line_width=0, alpha=0)
                hline = hv.HLine(self.A_min - 1).opts(color='blue', line_width=0, alpha=0)
            
            return invisible_rect * vline * hline
        
        image_dmap = hv.DynamicMap(
            pn.bind(make_image,
                    self.level_selector,
                    self.clim_low_slider,
                    self.clim_high_slider,
                    self.data_version_widget)
        )
        
        markers_overlay_dmap = hv.DynamicMap(
            pn.bind(make_markers_and_overlay, self.marker_version_widget)
        )
        
        image_rasterized = rasterize(
            image_dmap, 
            aggregator='mean',
            precompute=True
        )
        
        return image_rasterized * markers_overlay_dmap
    
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
            width=500,
            sizing_mode='fixed'
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
        
        if self.computing:
            return
        
        self.computing = True
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
        
        self.computing = False
    
    def create_plot(self):
        """Create the dynamics plot with linked X axes and independent Y axes."""
    
        def make_plot(version, epsilon_L, epsilon_R):
            # Always use the same key dimension for automatic x-axis linking
            time = self.time_dynamics if self.time_dynamics is not None else np.array([0, 1])
            
            # FIXED: Use self.current_eps0 and self.current_A instead of widget values
            if self.time_dynamics is not None and self.current_eps0 is not None and self.current_A is not None:
                # Use actual simulation parameters
                eps_min = self.current_eps0 - self.current_A * 1.1
                eps_max = self.current_eps0 + self.current_A * 1.1
            else:
                # No simulation → small default range
                eps_min = -0.01
                eps_max = 0.01
            
            if self.time_dynamics is None or self.rho_dynamics is None:
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
                    ylim=(0, 1),
                    xlim=(0, self.t_max_plot),
                    legend_position='right',
                    framewise=True
                )
    
                # Empty epsilon plot
                eps_curve = hv.Curve((time, np.zeros(len(time))), 'time', 'epsilon').opts(
                    color=colors[4], line_width=1.5
                )
                
                # Add horizontal lines for epsilon bounds
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
                    ylim=(eps_min, eps_max),
                    xlim=(0, self.t_max_plot),
                    framewise=True
                )
            else:
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
                    title=f'Population Dynamics (eps0={self.current_eps0:.6f}, A={self.current_A:.6f})',
                    xlabel='Time', ylabel='Population',
                    show_grid=True,
                    legend_position='right',
                    ylim=pop_ylim,
                    xlim=(0, self.t_max_plot),
                    framewise=True
                )
    
                # Real epsilon plot
                eps_curve = hv.Curve((time, self.eps_dynamics), 'time', 'epsilon').opts(
                    color=colors[4], line_width=1.5
                )
                
                # Add horizontal lines for epsilon bounds
                hline_eps_R_pos = hv.HLine(epsilon_R).opts(color='red', line_width=1.5, line_dash='dashed', alpha=0.8)
                hline_eps_R_neg = hv.HLine(-epsilon_R).opts(color='red', line_width=1.5, line_dash='dashed', alpha=0.8)
                hline_eps_L_pos = hv.HLine(epsilon_L).opts(color='green', line_width=1.5, line_dash='dashed', alpha=0.8)
                hline_eps_L_neg = hv.HLine(-epsilon_L).opts(color='green', line_width=1.5, line_dash='dashed', alpha=0.8)
    
                eps_overlay = eps_curve * hline_eps_R_pos * hline_eps_R_neg * hline_eps_L_pos * hline_eps_L_neg
                eps_plot = eps_overlay.opts(
                    width=800, height=200,
                    title=f'Epsilon Dynamics (range: [{eps_min:.6f}, {eps_max:.6f}])',
                    xlabel='Time', ylabel='ε(t)',
                    show_grid=True,
                    ylim=(eps_min, eps_max),
                    xlim=(0, self.t_max_plot),
                    framewise=True
                )
            
            # Stack vertically - X axes link automatically because they share 'time' dimension
            layout = (pop_plot + eps_plot).cols(1)
            return layout
    
        # Create DynamicMap - removed eps0_input and A_input from pn.bind
        return hv.DynamicMap(pn.bind(make_plot, 
                              self.dynamics_version_widget,
                              self.epsilon_L_widget,
                              self.epsilon_R_widget))
    
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
        
        if self.auto_update_enabled and not self._is_generating:
            self._update_and_regenerate_interferogram()
        elif self.auto_update_dynamics_enabled and not self.dynamics.computing:
            # Only regenerate dynamics if it has been computed before
            if self.dynamics.current_eps0 is not None and self.dynamics.current_A is not None:
                self._regenerate_dynamics_only()
        elif self.auto_update_both_enabled and not self._is_generating_both:
            self._regenerate_both()
    
    def _update_and_regenerate_interferogram(self, event=None):
        """Update parameters and regenerate interferogram only."""
        if self._is_generating:
            return
        self.sim_params.update_from_sliders()
        self._generate_interferogram_data()
    
    def _regenerate_dynamics_only(self, event=None):
        """Regenerate dynamics at current position with updated parameters."""
        if self.dynamics.current_eps0 is None or self.dynamics.current_A is None:
            self.dynamics.status_text.object = "**Dynamics:** ⚠️ No position set. Click on interferogram first."
            return
        
        if self.dynamics.computing:
            return
        
        # FIXED: Pass update_params=True so it uses current slider values
        self._generate_dynamics(
            self.dynamics.current_eps0, 
            self.dynamics.current_A,
            update_params=True
        )
    
    def _regenerate_both(self, event=None):
        """Regenerate both interferogram and dynamics simultaneously using SimRunGridSingleMode."""
        if self._is_generating_both:
            return
        
        # Check if dynamics position is set
        if self.dynamics.current_eps0 is None or self.dynamics.current_A is None:
            self.status_text.object = "**Status:** ⚠️ Set dynamics position first (click on interferogram)"
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
        
        if self._is_generating:
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
        """Handle hover on interferogram."""
        
        if not self.dynamics.enabled or x is None or y is None:
            return
        
        # Check for both auto-update modes
        if self.auto_update_both_enabled:
            if self.dynamics.hover_active:
                self._generate_dynamics(x, y, update_params=False)
        elif self.dynamics.auto_update:
            if self.dynamics.hover_active:
                self._generate_dynamics(x, y, update_params=False)
    
    def _generate_dynamics(self, eps0, A, update_params=False):
        """Generate dynamics for given coordinates.
        
        Args:
            update_params: If True, update parameters from sliders before computing
        """
        
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
                                print(f"✅ Tap at: ({event.x:.6f}, {event.y:.6f})")
                    
                    def on_hover_event(event):
                        if hasattr(event, 'x') and hasattr(event, 'y'):
                            if event.x is not None and event.y is not None:
                                self.parent_app._on_interferogram_hover(event.x, event.y)
                    
                    try:
                        model.on_event(Tap, on_tap_event)
                        model.on_event(MouseMove, on_hover_event)
                        print("✅ Events attached to Bokeh model")
                    except Exception as e:
                        print(f"⚠️ Event attachment failed: {e}")
                
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
            sizing_mode='fixed',
            scroll=True
        )
        
        # Interferogram section
        interferogram_section = pn.Column(
            self.interferogram.get_control_panel(),
            interferogram_pane,
            self.interferogram.data_version_widget,
            self.interferogram.marker_version_widget,
            sizing_mode='fixed'
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
            sizing_mode='fixed',
            visible=False
        )
        
        dynamics_section = pn.Column(
            pn.layout.Divider(),
            "### Dynamics Plot",
            self.dynamics.get_control_panel(),
            self.dynamics.status_text,
            dynamics_plot_panel,
            sizing_mode='fixed'
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
            sizing_mode='fixed',
            scroll=True
        )
        
        dashboard = pn.Row(
            sidebar,
            plot_area,
            sizing_mode='fixed'
        )
        
        return dashboard



