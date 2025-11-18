#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simulate_instantaneous_deformation.py
Combine pre-generated spatial noise with instantaneous deformation (OBS).
"""

import numpy as np
import scipy.io as sio

def simulate_instantaneous_deformation(obs, noise_file, nx_grid, ny_grid, save_name='instantaneous_deformation.mat'):
    """
    Combine instantaneous displacement with spatial noise.

    Parameters
    ----------
    obs : 1D array of OBS displacement
    noise_file : str
        Path to noise .mat file
    nx_grid, ny_grid : int
        Grid dimensions
    save_name : str
        Output filename
    """
    # Load noise
    noise_mat = sio.loadmat(noise_file)
    noise_grid = noise_mat['data_all'][:,0,0].reshape(ny_grid, nx_grid, order='F')

    # Combine with OBS displacement
    inst_def_grid = obs.reshape(ny_grid, nx_grid, order='F') + noise_grid

    # Save
    sio.savemat(save_name, {'inst_def_grid': inst_def_grid})
    print(f"✅ Instantaneous deformation saved as '{save_name}'")
