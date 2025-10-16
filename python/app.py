
import panel as pn
from app_interferogram_dynamics_class import InteractiveInterferogramDynamics

# Make sure you have Panel and any other required libraries installed
pn.extension()



# Define your app parameters (the same as before)
app_interferogram_dynamics = InteractiveInterferogramDynamics(
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

# Create the dashboard
dashboard = app_interferogram_dynamics.create_dashboard()

# Make the dashboard a Panel app (you can wrap it in pn.panel if needed)
panel_app = pn.Column(dashboard)

# Serve the app using Panel's server
panel_app.servable()  # This makes it ready to serve
