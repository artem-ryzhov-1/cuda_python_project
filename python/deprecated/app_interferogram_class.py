########################################
# python/app_interferogram_class.py
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
from config import SimRunGridMode


class InteractiveInterferogram:
    """
    Interactive interferogram visualization using Datashader + Panel.
    Optimized for Jupyter Lab and Google Colab.
    """

    def __init__(
        self,
        eps0_min,
        eps0_max,
        A_min,
        A_max,
        N_points_target,
        delta_C_range,
        GammaL0_range,
        GammaR0_range,
        Gamma_eg0_range,
        Gamma_phi0_range,
        N_steps_period_array,
        N_periods_array,
        N_periods_avg_array,
        delta_C_default,
        GammaL0_default,
        GammaR0_default,
        Gamma_eg0_default,
        Gamma_phi0_default,
        N_steps_period_default,
        N_periods_default,
        N_periods_avg_default,
        dC_default_thresholds,
        platform_type=platform_type,
        repo_path=repo_path,
        cmap_name='fire'
    ):
        """Initialize the interactive interferogram viewer."""

        # Store platform and path information
        self.platform_type = platform_type
        self.repo_path = repo_path

        # Store grid inputs
        self.eps0_min = eps0_min
        self.eps0_max = eps0_max
        self.A_min = A_min
        self.A_max = A_max
        self.N_points_target = N_points_target

        # Store parameter ranges
        self.delta_C_range = delta_C_range
        self.GammaL0_range = GammaL0_range
        self.GammaR0_range = GammaR0_range
        self.Gamma_eg0_range = Gamma_eg0_range
        self.Gamma_phi0_range = Gamma_phi0_range

        # Store discrete parameter ranges (can be tuples)
        if isinstance(N_steps_period_array, tuple):
            self.N_steps_period_range = N_steps_period_array
        else:
            self.N_steps_period_range = (int(N_steps_period_array[0]), int(N_steps_period_array[-1]))

        if isinstance(N_periods_array, tuple):
            self.N_periods_range = N_periods_array
        else:
            self.N_periods_range = (int(N_periods_array[0]), int(N_periods_array[-1]))

        if isinstance(N_periods_avg_array, tuple):
            self.N_periods_avg_range = N_periods_avg_array
        else:
            self.N_periods_avg_range = (int(N_periods_avg_array[0]), int(N_periods_avg_array[-1]))

        # Colormap and thresholds
        self.cmap_name = cmap_name
        self.dC_default_thresholds = dC_default_thresholds

        # Level labels
        self.level_labels = ["00", "01", "10", "11", "C", "dC"]

        # Initialize data storage
        self.avg_grids = {label: None for label in self.level_labels}
        self.current_data = None

        # Initialize current parameters
        self.delta_C = delta_C_default
        self.GammaL0 = GammaL0_default
        self.GammaR0 = GammaR0_default
        self.Gamma_eg0 = Gamma_eg0_default
        self.Gamma_phi0 = Gamma_phi0_default
        self.N_steps_period = N_steps_period_default
        self.N_periods = N_periods_default
        self.N_periods_avg = N_periods_avg_default

        # Auto-update state
        self.auto_update_enabled = False

        # Add a counter to force plot updates
        self.data_version = 0

        # Flag to prevent double execution
        self._is_generating = False

        self._clim_values_per_level = {}

        self._current_level = None

        # Create widgets
        self._create_widgets()

        # Generate initial data
        self._generate_data()

    def _create_widgets(self):
        """Create all Panel widgets."""

        # === Continuous parameter sliders with text inputs ===
        self.delta_C_slider = pn.widgets.FloatSlider(
            name='delta_C',
            start=self.delta_C_range[0],
            end=self.delta_C_range[1],
            value=self.delta_C,
            step=(self.delta_C_range[1] - self.delta_C_range[0]) / 1000,
            format='0.5e'
        )
        self.delta_C_input = pn.widgets.FloatInput(
            value=self.delta_C,
            format='0.5e',
            width=100
        )

        self.GammaL0_slider = pn.widgets.FloatSlider(
            name='GammaL0',
            start=self.GammaL0_range[0],
            end=self.GammaL0_range[1],
            value=self.GammaL0,
            step=0.5
        )
        self.GammaL0_input = pn.widgets.FloatInput(
            value=self.GammaL0,
            width=100
        )

        self.GammaR0_slider = pn.widgets.FloatSlider(
            name='GammaR0',
            start=self.GammaR0_range[0],
            end=self.GammaR0_range[1],
            value=self.GammaR0,
            step=0.1
        )
        self.GammaR0_input = pn.widgets.FloatInput(
            value=self.GammaR0,
            width=100
        )

        self.Gamma_eg0_slider = pn.widgets.FloatSlider(
            name='Gamma_eg0',
            start=self.Gamma_eg0_range[0],
            end=self.Gamma_eg0_range[1],
            value=self.Gamma_eg0,
            step=0.1
        )
        self.Gamma_eg0_input = pn.widgets.FloatInput(
            value=self.Gamma_eg0,
            width=100
        )

        self.Gamma_phi0_slider = pn.widgets.FloatSlider(
            name='Gamma_phi0',
            start=self.Gamma_phi0_range[0],
            end=self.Gamma_phi0_range[1],
            value=self.Gamma_phi0,
            step=0.1
        )
        self.Gamma_phi0_input = pn.widgets.FloatInput(
            value=self.Gamma_phi0,
            width=100
        )

        # === Discrete parameter sliders ===
        self.N_steps_period_slider = pn.widgets.IntSlider(
            name='N_steps_period',
            start=self.N_steps_period_range[0],
            end=self.N_steps_period_range[1],
            value=self.N_steps_period,
            step=1
        )
        self.N_steps_period_input = pn.widgets.IntInput(
            value=self.N_steps_period,
            width=100
        )

        self.N_periods_slider = pn.widgets.IntSlider(
            name='N_periods',
            start=self.N_periods_range[0],
            end=self.N_periods_range[1],
            value=self.N_periods,
            step=1
        )
        self.N_periods_input = pn.widgets.IntInput(
            value=self.N_periods,
            width=100
        )

        self.N_periods_avg_slider = pn.widgets.IntSlider(
            name='N_periods_avg',
            start=self.N_periods_avg_range[0],
            end=self.N_periods_avg_range[1],
            value=self.N_periods_avg,
            step=1
        )
        self.N_periods_avg_input = pn.widgets.IntInput(
            value=self.N_periods_avg,
            width=100
        )

        # Link sliders and inputs bidirectionally
        self._link_slider_input('delta_C')
        self._link_slider_input('GammaL0')
        self._link_slider_input('GammaR0')
        self._link_slider_input('Gamma_eg0')
        self._link_slider_input('Gamma_phi0')
        self._link_slider_input('N_steps_period')
        self._link_slider_input('N_periods')
        self._link_slider_input('N_periods_avg')

        # === Level selector ===
        self.level_selector = pn.widgets.RadioButtonGroup(
            name='Display Level',
            options=['00', '01', '10', '11', 'C', 'dC'],
            value='dC',
            button_type='primary'
        )

        # === Color range sliders ===
        self.clim_low_slider = pn.widgets.FloatSlider(
            name='Color Min',
            start=-abs(2*self.dC_default_thresholds[0]),
            end=abs(2*self.dC_default_thresholds[0]),
            value=self.dC_default_thresholds[0],
            step=0.01
        )
        self.clim_low_input = pn.widgets.FloatInput(
            value=self.dC_default_thresholds[0],
            width=100
        )

        self.clim_high_slider = pn.widgets.FloatSlider(
            name='Color Max',
            start=-abs(2*self.dC_default_thresholds[1]),
            end=abs(2*self.dC_default_thresholds[1]),
            value=self.dC_default_thresholds[1],
            step=0.01
        )
        self.clim_high_input = pn.widgets.FloatInput(
            value=self.dC_default_thresholds[1],
            width=100
        )

        # Link color sliders and inputs
        self._link_slider_input('clim_low')
        self._link_slider_input('clim_high')

        # === Update button ===
        self.update_button = pn.widgets.Button(
            name='🔄 Regenerate Data',
            button_type='success',
            width=180
        )

        # === Auto-update toggle ===
        self.auto_update_toggle = pn.widgets.Toggle(
            name='Auto Update',
            value=False,
            button_type='warning',
            width=120
        )

        # === Status indicator ===
        self.status_text = pn.pane.Markdown(
            "**Status:** Ready",
            width=300,
            sizing_mode='fixed'
        )

        # === Computation time display ===
        self.timing_text = pn.pane.Markdown(
            "**Last computation:** N/A",
            width=300,
            sizing_mode='fixed'
        )

        # === Log display area (scrollable) ===
        self.log_display = pn.pane.Markdown(
            "**Log:** Waiting for first computation...",
            width=650,
            height=150,
            styles={'resize': 'both', 'overflow': 'auto', 'background': '#f5f5f5', 'padding': '10px',
                    'font-family': 'monospace', 'font-size': '11px', 'border': '1px solid #ddd'},
            sizing_mode='fixed'
        )

        # Watch level selector to update color limits
        self.level_selector.param.watch(self._on_level_change, 'value')

        # Watch auto-update toggle
        self.auto_update_toggle.param.watch(self._on_auto_update_toggle, 'value')

        # Watch simulation parameter sliders for auto-update (NOT color sliders)
        for slider_name in ['delta_C', 'GammaL0', 'GammaR0', 'Gamma_eg0', 'Gamma_phi0',
                           'N_steps_period', 'N_periods', 'N_periods_avg']:
            slider = getattr(self, f'{slider_name}_slider')
            slider.param.watch(self._on_slider_change, 'value')

    def _link_slider_input(self, param_name):
        """Link a slider and input box bidirectionally."""
        slider = getattr(self, f'{param_name}_slider')
        input_box = getattr(self, f'{param_name}_input')

        def slider_to_input(event):
            input_box.value = event.new

        def input_to_slider(event):
            if event.new is not None:
                # Clamp to slider range
                val = max(slider.start, min(slider.end, event.new))
                slider.value = val
                if val != event.new:
                    input_box.value = val

        slider.param.watch(slider_to_input, 'value')
        input_box.param.watch(input_to_slider, 'value')

    def _on_level_change(self, event):
        prev_level = self._current_level
        new_level = event.new

        # Save current slider values for previous level
        if prev_level is not None:
            self._clim_values_per_level[prev_level] = (
                self.clim_low_slider.value,
                self.clim_high_slider.value
            )

        # Set fixed range depending on level
        if new_level in ["00", "01", "10", "11"]:
            slider_min, slider_max = 0, 1
        elif new_level == "C":
            slider_min, slider_max = -18.2, 18.2
        elif new_level == "dC":
            slider_min = -abs(2 * self.dC_default_thresholds[0])
            slider_max = abs(2 * self.dC_default_thresholds[1])
        else:
            slider_min, slider_max = 0, 1

        # Set slider ranges
        self.clim_low_slider.start = slider_min
        self.clim_low_slider.end = slider_max
        self.clim_high_slider.start = slider_min
        self.clim_high_slider.end = slider_max

        # Restore last-used values if available, else use full range
        if new_level in self._clim_values_per_level:
            low_val, high_val = self._clim_values_per_level[new_level]
        else:
            low_val, high_val = slider_min, slider_max

        self.clim_low_slider.value = low_val
        self.clim_high_slider.value = high_val
        self.clim_low_input.value = low_val
        self.clim_high_input.value = high_val

        # Track current level
        self._current_level = new_level

    def _on_auto_update_toggle(self, event):
        """Handle auto-update toggle."""
        self.auto_update_enabled = event.new
        if self.auto_update_enabled:
            self.update_button.name = '🔄 Update (Auto On)'
            self.update_button.button_type = 'light'
        else:
            self.update_button.name = '🔄 Regenerate Data'
            self.update_button.button_type = 'success'

    def _on_slider_change(self, event):
        """Handle slider changes for auto-update."""
        if self.auto_update_enabled:
            self._update_params_and_regenerate()

    def _update_params_and_regenerate(self, event=None):
        """Update current parameters from sliders and regenerate."""

        # Update current parameters from sliders
        self.delta_C = self.delta_C_slider.value
        self.GammaL0 = self.GammaL0_slider.value
        self.GammaR0 = self.GammaR0_slider.value
        self.Gamma_eg0 = self.Gamma_eg0_slider.value
        self.Gamma_phi0 = self.Gamma_phi0_slider.value
        self.N_steps_period = self.N_steps_period_slider.value
        self.N_periods = self.N_periods_slider.value
        self.N_periods_avg = self.N_periods_avg_slider.value

        # Regenerate data
        self._generate_data()

    def _generate_data(self):
        """Generate interferogram data based on current parameters."""

        # Prevent double execution
        if self._is_generating:
            return

        self._is_generating = True

        # Capture stdout/stderr
        captured_stdout = StringIO()
        captured_stderr = StringIO()

        with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
            self.status_text.object = "**Status:** 🔄 Generating data..."
            start_time = time.perf_counter()

            simr = SimRunGridMode(
                delta_C=self.delta_C,
                GammaL0=self.GammaL0,
                GammaR0=self.GammaR0,
                Gamma_eg0=self.Gamma_eg0,
                Gamma_phi0=self.Gamma_phi0,
                eps0_min=self.eps0_min,
                eps0_max=self.eps0_max,
                A_min=self.A_min,
                A_max=self.A_max,
                N_points_target=self.N_points_target,
                N_steps_period=self.N_steps_period,
                N_periods=self.N_periods,
                N_periods_avg=self.N_periods_avg,
                platform_type=self.platform_type,
                repo_path=self.repo_path
            )

            self.eps0_grid, self.A_grid, rho_avg_cdc_3d, _ = run_simulation(simr)

            # Store the averaged grids
            for i, label in enumerate(self.level_labels):
                self.avg_grids[label] = rho_avg_cdc_3d[i]

            end_time = time.perf_counter()
            elapsed = end_time - start_time

        self.status_text.object = "**Status:** ✅ Data ready"
        self.timing_text.object = f"**Last computation:** {elapsed:.2f} seconds"

        # Build log text
        log_text = f"**Computation completed in {elapsed:.2f}s**\n\n"
        log_text += f"**Parameters:**\n"
        log_text += f"- delta_C = {self.delta_C:.6e}\n"
        log_text += f"- GammaL0 = {self.GammaL0}, GammaR0 = {self.GammaR0}\n"
        log_text += f"- Gamma_eg0 = {self.Gamma_eg0}, Gamma_phi0 = {self.Gamma_phi0}\n"
        log_text += f"- N_steps = {self.N_steps_period}, N_periods = {self.N_periods}, N_avg = {self.N_periods_avg}\n\n"

        stdout_content = captured_stdout.getvalue()
        stderr_content = captured_stderr.getvalue()

        if stdout_content:
            log_text += f"**Output:**\n```\n{stdout_content}\n```\n"
        if stderr_content:
            log_text += f"**Warnings:**\n```\n{stderr_content}\n```\n"

        self.log_display.object = log_text

        # Increment data version to trigger plot update
        self.data_version += 1
        if hasattr(self, 'data_version_widget'):
            self.data_version_widget.value = self.data_version

        self._is_generating = False

    def _compute_level_data(self, level):
        """Return data for the selected level."""

        if level in ["00", "01", "10", "11", "C", "dC"]:
            return self.avg_grids[level]
        else:
            return self.avg_grids["00"]

    '''
    def view_plot(self, level=None, clim_low=None, clim_high=None, data_version=None):
        """Create or update the HoloViews plot with datashader."""

        if level is None:
            level = self.level_selector.value
        if clim_low is None:
            clim_low = self.clim_low_slider.value
        if clim_high is None:
            clim_high = self.clim_high_slider.value

        # Unique cache keys per level
        cache_key = f"_cached_img_{level}"
        version_key = f"_cached_data_version_{level}"

        # Update cache only if data changed
        if not hasattr(self, cache_key) or getattr(self, version_key, -1) != self.data_version:
            data = self._compute_level_data(level)
            base_img = hv.Image(
                (self.eps0_grid, self.A_grid, data),
                kdims=['eps0', 'A'],
                vdims=['value']
            )
            setattr(self, cache_key, base_img)
            setattr(self, version_key, self.data_version)
        else:
            base_img = getattr(self, cache_key)

        # === 💡 Force re-render by deepcopying the image ===
        img = copy.deepcopy(base_img).opts(
            cmap=self.cmap_name,
            colorbar=True,
            width=800,
            height=600,
            clim=(clim_low, clim_high),
            title=f'Interactive Interferogram - Level: {level}',
            xlabel='eps0',
            ylabel='A',
            tools=['hover', 'box_zoom', 'wheel_zoom', 'pan', 'reset'],
            active_tools=['wheel_zoom']
        )

        return img
    '''

    def _dynamic_plot(self):
        """Returns a DynamicMap that updates color limits and refreshes when data changes."""

        # Widget to track data version and trigger redraw
        if not hasattr(self, 'data_version_widget'):
            self.data_version_widget = pn.widgets.IntInput(value=self.data_version, visible=False)

        def make_image(level, clim_low, clim_high, data_version):
            data = self._compute_level_data(level)
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
                tools=['hover', 'box_zoom', 'wheel_zoom', 'pan', 'reset'],
                active_tools=['wheel_zoom']
            )
            return img

        return hv.DynamicMap(
            pn.bind(make_image,
                    self.level_selector,
                    self.clim_low_slider,
                    self.clim_high_slider,
                    self.data_version_widget)
        )


    def create_dashboard(self):
        """Create the complete Panel dashboard."""

        # Connect update button
        self.update_button.on_click(self._update_params_and_regenerate)

        # Hidden widget to trigger plot updates (keep if you need it for other updates)
        self.data_version_widget = pn.widgets.IntInput(value=0, visible=False)

        # Sidebar with all controls stays the same
        sidebar = pn.Column(
            "## 🎛️ Simulation Parameters",
            pn.layout.Divider(),
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
            pn.layout.Divider(),
            pn.Row(self.update_button, self.auto_update_toggle),
            self.status_text,
            self.timing_text,
            width=500,
            sizing_mode='fixed'
        )

        # The plot + sliders layout now uses the DynamicMap directly
        plot_and_controls = pn.Column(
            pn.Row(
                pn.Column("**Level:**", self.level_selector),
                pn.Column(
                    pn.Row(self.clim_low_slider, self.clim_low_input),
                    pn.Row(self.clim_high_slider, self.clim_high_input)
                )
            ),
            self._dynamic_plot(),  # <-- insert DynamicMap here directly
            pn.layout.Divider(),
            "### Computation Log",
            self.log_display,
            sizing_mode='fixed'
        )

        dashboard = pn.Row(
            sidebar,
            plot_and_controls,
            sizing_mode='fixed'
        )

        return dashboard

'''
app_interferogram = InteractiveInterferogram(
    eps0_min=-0.006,
    eps0_max=0.006,
    A_min=0.0,
    A_max=0.01,
    N_points_target=500_000,
    delta_C_range=(0, 0.0006),
    GammaL0_range=(0, 100),
    GammaR0_range=(0, 24),
    Gamma_eg0_range=(0, 16),
    Gamma_phi0_range=(0, 72),
    N_steps_period_array=(100, 2000),
    N_periods_array=(1, 20),
    N_periods_avg_array=(1, 10),
    delta_C_default=0.00011608757555650906,
    GammaL0_default=50.0,
    GammaR0_default=12.0,
    Gamma_eg0_default=0.8,
    Gamma_phi0_default=3.6,
    N_steps_period_default=1000,
    N_periods_default=10,
    N_periods_avg_default=1,
    dC_default_thresholds=(-3000, 1000),
    cmap_name='fire'
)
'''

#dashboard = app_interferogram.create_dashboard()

#dashboard
