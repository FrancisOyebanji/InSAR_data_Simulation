#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simulate_temporal_deformation.py
Generates temporal deformation fields from OBS displacement.
"""

import numpy as np
import scipy.io as sio

def simulate_temporal_deformation(obs, nx_grid, ny_grid, num_times, T0, deformation_type='log', save_name='temporal_deformation.mat'):
    """
    Generate temporal deformation.

    Parameters
    ----------
    obs : 1D array of OBS displacement
        Instantaneous displacement (third column of OBS)
    nx_grid, ny_grid : int
        Grid size
    num_times : int
        Number of time steps
    T0 : float
        Decay or scaling time
    deformation_type : str
        'linear', 'log', or 'exp'
    save_name : str
        Output filename
    """
    U0 = obs.reshape(ny_grid, nx_grid, order='F')
    ts_cube = np.zeros((ny_grid, nx_grid, num_times))
    t_vec = np.linspace(0.1, num_times*0.1, num_times)

    for i, t in enumerate(t_vec):
        if deformation_type == 'linear':
            ts_cube[:,:,i] = U0 * (t/T0)
        elif deformation_type == 'log':
            ts_cube[:,:,i] = U0 * np.log(1 + t/T0)
        elif deformation_type == 'exp':
            ts_cube[:,:,i] = U0 * (1 - np.exp(-t/T0))
        else:
            raise ValueError("Unknown deformation_type")

    sio.savemat(save_name, {'ts_cube': ts_cube})
    print(f"✅ Temporal deformation saved as '{save_name}'")
