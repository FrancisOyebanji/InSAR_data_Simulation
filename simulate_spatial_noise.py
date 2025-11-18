#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simulate_spatial_noise.py
Generate spatially correlated noise.
All parameters are passed from the main program.
No plotting is performed here.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.io as sio

def simulate_spatial_noise(obs_file,
                           num_groups,
                           sigma_frac,
                           Lc_frac,
                           cov_type,
                           save_name):
    """
    Generate spatial noise based on OBS coordinates.

    Parameters
    ----------
    obs_file : str
        OBS file with columns [X,Y,displacement]
    num_groups : int
        Number of noise realizations
    sigma_frac : float
        Noise standard deviation relative to max OBS displacement
    Lc_frac : float
        Correlation length as fraction of max spatial extent
    cov_type : str
        'gaussian' or 'exponential'
    save_name : str
        Output .mat filename

    Returns
    -------
    data_all : ndarray
        N x 1 x num_groups array of noise realizations
    """
    obs = np.loadtxt(obs_file)
    coords = obs[:, :2]
    N = coords.shape[0]

    # Correlation length in physical units
    Lc = Lc_frac * np.max(np.ptp(coords, axis=0))
    sigma = sigma_frac * np.max(np.abs(obs[:,2]))

    # Covariance matrix
    D = squareform(pdist(coords))
    if cov_type.lower() in ['gauss', 'gaussian']:
        C = np.exp(-(D/Lc)**2)
    elif cov_type.lower() in ['exp', 'exponential']:
        C = np.exp(-D/Lc)
    else:
        raise ValueError("Unknown covariance type. Choose 'gaussian' or 'exponential'.")

    C = (C + C.T)/2 + 1e-8*np.eye(N)
    Lchol = np.linalg.cholesky(C)

    # Generate noise realizations
    data_all = np.zeros((N,1,num_groups))
    for k in range(num_groups):
        z = np.random.randn(N)
        noise = Lchol @ z
        noise = noise / np.max(np.abs(noise)) * sigma
        data_all[:,0,k] = noise

    sio.savemat(save_name, {'data_all': data_all, 'num_groups': num_groups})
    print(f"✅ Spatial noise saved successfully to '{save_name}'")
    return data_all
