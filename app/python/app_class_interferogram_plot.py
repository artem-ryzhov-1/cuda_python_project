########################################
# app/python/app_class_interferogram_plot.py
########################################

import numpy as np
import panel as pn
import holoviews as hv
import datashader as ds
from holoviews.operation.datashader import rasterize

# Enable Panel extension for Jupyter
pn.extension()
hv.extension('bokeh')


import matplotlib.pyplot as plt

# Get default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



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
        self.level_labels = ["00", "01", "10", "11", "C", "Δφ"]
        
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
            value='Δφ',
            button_type='primary'
        )
        
        self.clim_low_slider = pn.widgets.FloatSlider(
            name='Color Min ([1])',
            start=self.dC_default_thresholds[0],
            end=self.dC_default_thresholds[1],
            value=self.dC_default_thresholds[0],
            step=(abs(self.dC_default_thresholds[1]-self.dC_default_thresholds[0]))/1000
        )
        self.clim_low_input = pn.widgets.FloatInput(
            value=self.dC_default_thresholds[0], width=100,
            step=(abs(self.dC_default_thresholds[1]-self.dC_default_thresholds[0]))/1000
        )
        
        self.clim_high_slider = pn.widgets.FloatSlider(
            name='Color Max ([1])',
            start=self.dC_default_thresholds[0],
            end=self.dC_default_thresholds[1],
            value=self.dC_default_thresholds[1],
            step=(abs(self.dC_default_thresholds[1]-self.dC_default_thresholds[0]))/1000
        )
        self.clim_high_input = pn.widgets.FloatInput(
            value=self.dC_default_thresholds[1], width=100,
            step=(abs(self.dC_default_thresholds[1]-self.dC_default_thresholds[0]))/1000
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
            slider_min, slider_max = -18.2, 18.2
        elif new_level == "Δφ":
            slider_min = self.dC_default_thresholds[0]
            slider_max = self.dC_default_thresholds[1]
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
                    xlabel='ε₀/Ec', ylabel='A/Ec',
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
                xlabel='ε₀/Ec',
                ylabel='A/Ec'
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
                    xlabel='ε₀/Ec', ylabel='A/Ec',
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
                xlabel='ε₀/Ec',
                ylabel='A/Ec'
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
                    xlabel='ε₀/Ec', ylabel='A/Ec',
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
                xlabel='ε₀/Ec',
                ylabel='A/Ec'
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



