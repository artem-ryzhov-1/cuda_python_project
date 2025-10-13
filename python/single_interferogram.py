

import time

time1 = time.time()

from config import SimRun
from simulation import run_simulation

#from main_v4_1_win_function import run_simulation

time2 = time.time()

simr = SimRun(
    delta_C=0.001,
    GammaL0=50,
    GammaR0=12,
    Gamma_eg0=0.8,
    Gamma_phi0=3.6,
    eps0_min=-0.006,
    eps0_max=0.006,
    A_min=0,
    A_max=0.01,
    N_points_target=1000000,
    N_steps_period=1000,
    N_periods=10,
    N_periods_avg=1
)

time3 = time.time()

run_simulation(simr)

time4 = time.time()

print(f"time2 - time1: {(time2 - time1):.3f} s")
print(f"time3 - time2: {(time3 - time2):.3f} s")
print(f"time4 - time3: {(time4 - time3):.3f} s")
print(f"time4 - time1: {(time4 - time1):.3f} s")








