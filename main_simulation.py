#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from full_simulation import run_full_simulation
import os

# Windows
os.system('cls') # Clear console
# ================= USER PARAMETERS =================
obs_file = 'inp/obs.txt'  # path to observed displacement data
nx_grid = 60                        # number of grid points in x
ny_grid = 60                        # number of grid points in y

sigma_frac = 0.1                    # noise standard deviation fraction
Lc_frac = 0.1                        # correlation length fraction to MAX distance
cov_type = 'gaussian'               # covariance type ('gaussian' or 'exponential')

num_inst = 10                        # number of instantaneous deformation realizations
num_times = 12                        # number of temporal deformation steps
T0 = 0.35                            # decay time for temporal deformation
deformation_type = 'linear'             # 'linear', 'log', or 'exp'

noise_total = 100                     # total number of noise realizations
num_noise_plot = 10                   # how many noise samples to plot

# ================= LOAD OBSERVATIONS =================
obs_disp = np.loadtxt(obs_file)[:, 2]  # assume third column is displacement

# ================= RUN FULL SIMULATION =================
run_full_simulation(
    obs_file=obs_file,
    obs_disp=obs_disp,
    nx_grid=nx_grid,
    ny_grid=ny_grid,
    sigma_frac=sigma_frac,
    Lc_frac=Lc_frac,
    cov_type=cov_type,
    num_inst=num_inst,
    num_times=num_times,
    T0=T0,
    deformation_type=deformation_type,
    noise_total=noise_total,
    num_noise_plot=num_noise_plot,
    out_dir='out'
)
