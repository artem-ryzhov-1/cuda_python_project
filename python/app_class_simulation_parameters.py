########################################
# python/app_class_simulation_parameters.py
########################################

import panel as pn
pn.extension('mathjax')
import holoviews as hv

import matplotlib.pyplot as plt
# Get default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Enable Panel extension for Jupyter
pn.extension()
hv.extension('bokeh')

# Import simulation modules
from simulation import run_simulation
from config import SimRunGridMode, SimRunSingleMode, SimRunGridSingleMode





class SimulationParameters:
    """Manages simulation parameters and their widgets."""
    
    def __init__(self, delta_C_range, GammaL0_range, GammaR0_range, Gamma_eg0_range, 
                 Gamma_phi0_range, sigma_eps_range, nu_range, E_C_range,
                 
                 N_steps_period_range, N_periods_range, 
                 N_periods_avg_range, N_samples_noise_range,

                 delta_C_default, GammaL0_default, GammaR0_default, Gamma_eg0_default,
                 Gamma_phi0_default, sigma_eps_default, nu_default, E_C_default,
                 
                 N_steps_period_default, N_periods_default, 
                 N_periods_avg_default, N_samples_noise_default):
        
        # Current values
        self.delta_C = delta_C_default
        self.GammaL0 = GammaL0_default
        self.GammaR0 = GammaR0_default
        self.Gamma_eg0 = Gamma_eg0_default
        self.Gamma_phi0 = Gamma_phi0_default
        self.sigma_eps = sigma_eps_default
        self.nu = nu_default
        self.E_C = E_C_default
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
        self.nu_range = nu_range
        self.E_C_range = E_C_range
        self.N_steps_period_range = N_steps_period_range
        self.N_periods_range = N_periods_range
        self.N_periods_avg_range = N_periods_avg_range
        self.N_samples_noise_range = N_samples_noise_range
        
        # Create widgets
        self._create_widgets()
    
    def _create_widgets(self):
        """Create parameter widgets."""
        self.delta_C_label     = pn.pane.LaTeX(r"$\Delta_{C},\ \mathrm{[E_C]}$")
        self.GammaL0_label     = pn.pane.LaTeX(r"$\Gamma_{L0},\ \mathrm{[GHz]}$")
        self.GammaR0_label     = pn.pane.LaTeX(r"$\Gamma_{R0},\ \mathrm{[GHz]}$")
        self.Gamma_eg0_label   = pn.pane.LaTeX(r"$\Gamma_{eg0},\ \mathrm{[GHz]}$")
        self.Gamma_phi0_label  = pn.pane.LaTeX(r"$\Gamma_{\phi0},\ \mathrm{[GHz]}$")
        self.sigma_eps_label   = pn.pane.LaTeX(r"$\sigma_{\varepsilon},\ \mathrm{[E_C]}$")
        self.nu_label          = pn.pane.LaTeX(r"$\nu,\ \mathrm{[GHz]}$")
        self.E_C_label         = pn.pane.LaTeX(r"$E_C,\ \mathrm{[eV]}$")

        # Continuous parameters
        self.delta_C_slider = pn.widgets.FloatSlider(
            name='', start=self.delta_C_range[0], end=self.delta_C_range[1],
            value=self.delta_C, step=(self.delta_C_range[1] - self.delta_C_range[0]) / 1000,
            show_value=False
        )
        self.delta_C_input = pn.widgets.FloatInput(
            value=self.delta_C, width=100,
            step=(self.delta_C_range[1] - self.delta_C_range[0]) / 1000
        )
        
        self.GammaL0_slider = pn.widgets.FloatSlider(
            name='', start=self.GammaL0_range[0], end=self.GammaL0_range[1],
            value=self.GammaL0, step=(self.GammaL0_range[1] - self.GammaL0_range[0]) / 1000,
            show_value=False
        )
        self.GammaL0_input = pn.widgets.FloatInput(value=self.GammaL0, width=100)
        
        self.GammaR0_slider = pn.widgets.FloatSlider(
            name='', start=self.GammaR0_range[0], end=self.GammaR0_range[1],
            value=self.GammaR0, step=(self.GammaR0_range[1] - self.GammaR0_range[0]) / 1000,
            show_value=False
        )
        self.GammaR0_input = pn.widgets.FloatInput(value=self.GammaR0, width=100)
        
        self.Gamma_eg0_slider = pn.widgets.FloatSlider(
            name='', start=self.Gamma_eg0_range[0], end=self.Gamma_eg0_range[1],
            value=self.Gamma_eg0, step=(self.Gamma_eg0_range[1] - self.Gamma_eg0_range[0]) / 1000,
            show_value=False
        )
        self.Gamma_eg0_input = pn.widgets.FloatInput(value=self.Gamma_eg0, width=100)
        
        self.Gamma_phi0_slider = pn.widgets.FloatSlider(
            name='', start=self.Gamma_phi0_range[0], end=self.Gamma_phi0_range[1],
            value=self.Gamma_phi0, step=(self.Gamma_phi0_range[1] - self.Gamma_phi0_range[0]) / 1000,
            show_value=False
        )
        self.Gamma_phi0_input = pn.widgets.FloatInput(value=self.Gamma_phi0, width=100)
        
        self.sigma_eps_slider = pn.widgets.FloatSlider(
            name='', start=self.sigma_eps_range[0], end=self.sigma_eps_range[1],
            value=self.sigma_eps, step=(self.sigma_eps_range[1] - self.sigma_eps_range[0]) / 1000,
            disabled=True, show_value=False
        )
        self.sigma_eps_input = pn.widgets.FloatInput(value=self.sigma_eps, width=100, disabled=True)

        self.nu_slider = pn.widgets.FloatSlider(
            name='', start=self.nu_range[0], end=self.nu_range[1],
            value=self.nu, step=(self.nu_range[1] - self.nu_range[0]) / 1000,
            show_value=False
        )
        self.nu_input = pn.widgets.FloatInput(
            value=self.nu, width=100
        )
        
        self.E_C_slider = pn.widgets.FloatSlider(
            name='', start=self.E_C_range[0], end=self.E_C_range[1],
            value=self.E_C, step=(self.E_C_range[1] - self.E_C_range[0]) / 1000,
            show_value=False
        )
        self.E_C_input = pn.widgets.FloatInput(
            value=self.E_C, width=100
        )
        
        # Discrete parameters
        self.N_steps_period_slider = pn.widgets.IntSlider(
            name='Number of steps per period', start=self.N_steps_period_range[0], 
            end=self.N_steps_period_range[1], value=self.N_steps_period, step=1,
            show_value=False
        )
        self.N_steps_period_input = pn.widgets.IntInput(value=self.N_steps_period, width=100)
        
        self.N_periods_slider = pn.widgets.IntSlider(
            name='Number of periods', start=self.N_periods_range[0], end=self.N_periods_range[1],
            value=self.N_periods, step=1,
            show_value=False
        )
        self.N_periods_input = pn.widgets.IntInput(value=self.N_periods, width=100)
        
        self.N_periods_avg_slider = pn.widgets.IntSlider(
            name='Number of periods of averaging', start=self.N_periods_avg_range[0], 
            end=self.N_periods_avg_range[1], value=self.N_periods_avg, step=1,
            show_value=False
        )
        self.N_periods_avg_input = pn.widgets.IntInput(value=self.N_periods_avg, width=100)
        
        self.N_samples_noise_slider = pn.widgets.IntSlider(
            name='Number of samples in the quasi-static approach', start=self.N_samples_noise_range[0], 
            end=self.N_samples_noise_range[1], value=self.N_samples_noise, step=1, disabled=True,
            show_value=False
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
                      'nu', 'E_C',
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
        self.nu = self.nu_slider.value
        self.E_C = self.E_C_slider.value
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
            'E_C': self.E_C,
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
        top_margin_label = -10
        bottom_margin_label = -10
        top_margin_slider = 0
        bottom_margin_slider = -10

        return pn.Column(
            "### Physical Parameters",
            pn.Row(self.delta_C_label, margin=(top_margin_label, 0, bottom_margin_label, 0)),
            pn.Row(self.delta_C_slider, self.delta_C_input, margin=(top_margin_slider, 0, bottom_margin_slider, 0)),
            pn.Row(self.GammaL0_label, margin=(top_margin_label, 0, bottom_margin_label, 0)),
            pn.Row(self.GammaL0_slider, self.GammaL0_input, margin=(top_margin_slider, 0, bottom_margin_slider, 0)),
            pn.Row(self.GammaR0_label, margin=(top_margin_label, 0, bottom_margin_label, 0)),
            pn.Row(self.GammaR0_slider, self.GammaR0_input, margin=(top_margin_slider, 0, bottom_margin_slider, 0)),
            pn.Row(self.Gamma_eg0_label, margin=(top_margin_label, 0, bottom_margin_label, 0)),
            pn.Row(self.Gamma_eg0_slider, self.Gamma_eg0_input, margin=(top_margin_slider, 0, bottom_margin_slider, 0)),
            pn.Row(self.Gamma_phi0_label, margin=(top_margin_label, 0, bottom_margin_label, 0)),
            pn.Row(self.Gamma_phi0_slider, self.Gamma_phi0_input, margin=(top_margin_slider, 0, bottom_margin_slider, 0)),
            pn.Row(self.sigma_eps_label, margin=(top_margin_label, 0, bottom_margin_label, 0)),
            pn.Row(self.sigma_eps_slider, self.sigma_eps_input, margin=(top_margin_slider, 0, bottom_margin_slider, 0)),
            pn.Row(self.nu_label, margin=(top_margin_label, 0, bottom_margin_label, 0)),
            pn.Row(self.nu_slider, self.nu_input, margin=(top_margin_slider, 0, bottom_margin_slider, 0)),
            pn.Row(self.E_C_label, margin=(top_margin_label, 0, bottom_margin_label, 0)),
            pn.Row(self.E_C_slider, self.E_C_input, margin=(top_margin_slider, 0, bottom_margin_slider, 0)),
            pn.layout.Divider(),
            "### Time Parameters",
            pn.Row(self.N_steps_period_slider, self.N_steps_period_input),
            pn.Row(self.N_periods_slider, self.N_periods_input),
            pn.Row(self.N_periods_avg_slider, self.N_periods_avg_input),
            pn.Row(self.N_samples_noise_slider, self.N_samples_noise_input),
            pn.layout.Divider(),
            self.quasi_static_toggle
        )

