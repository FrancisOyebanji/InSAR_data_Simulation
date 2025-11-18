#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
from simulate_spatial_noise import simulate_spatial_noise
from simulate_temporal_deformation import simulate_temporal_deformation
from simulate_instantaneous_deformation import simulate_instantaneous_deformation
from plot_fields import plot_3D_fields
import matplotlib.pyplot as plt
import os

def run_full_simulation(obs_file,
                        obs_disp,
                        nx_grid, ny_grid,
                        sigma_frac, Lc_frac, cov_type,
                        num_inst, num_times, T0,
                        deformation_type,
                        noise_total,
                        num_noise_plot=10,
                        out_dir='out'):
    """
    Full simulation of spatial and temporal deformation with noise.
    Saves all results to 'out' folder.
    """

    ny, nx = ny_grid, nx_grid
    N = nx * ny

    # make sure out_dir exists
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- STEP 0: Covariance matrix ----------------
    # Load observation data
    obs_data = np.loadtxt(obs_file)
    
    # Take the first two columns as coordinates (x, y)
    coords = obs_data[:, :2]
    
    # Compute pairwise Euclidean distance between all points
    D = squareform(pdist(coords))
    D_max = np.max(D)  # maximum distance
    
    # Convert Lc_frac (proportion) to actual correlation length
    Lc = Lc_frac * D_max
    
    # Build covariance matrix based on selected type
    if cov_type.lower() in ['gaussian', 'gauss']:
        C = np.exp(-(D / Lc) ** 2)  # Gaussian covariance
    else:
        C = np.exp(-D / Lc)         # Exponential covariance
        
    # Optional: visualize the distance matrix
    plt.figure(figsize=(5, 4))
    plt.imshow(D, origin='upper', cmap='viridis')
    plt.colorbar()
    plt.title("Distance Matrix")
    plt.show()

    # Plot the covariance matrix with automatic contrast
    vmin = np.percentile(C, 5)    # 5% percentile
    vmax = np.percentile(C, 95)   # 95% percentile
    plt.figure(figsize=(5, 4))
    plt.imshow(C, origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title('Covariance Matrix')
    plt.colorbar()
    plt.xlabel('Grid index')
    plt.ylabel('Grid index')
    plt.show()
    # ---------------- STEP 1: Generate spatial noise ----------------
    print("Generating spatial noise...")
    noise_file = os.path.join(out_dir, 'simulated_noise.mat')
    simulate_spatial_noise(obs_file=obs_file,
                           num_groups=noise_total,
                           sigma_frac=sigma_frac,
                           Lc_frac=Lc_frac,
                           cov_type=cov_type,
                           save_name=noise_file)

    noise_mat = sio.loadmat(noise_file)
    noise_all = noise_mat['data_all']  # shape: N x 1 x noise_total

    # reshape into grid
    noise_all_grid = np.zeros((ny, nx, noise_total))
    for k in range(noise_total):
        noise_all_grid[:,:,k] = noise_all[:,0,k].reshape(ny,nx,order='F')

    # ---------------- STEP 2: Instantaneous deformation + noise ----------------
    print("Generating instantaneous deformation...")
    inst_all_grid = np.zeros((ny, nx, num_inst))
    noise_idx_inst = np.random.choice(noise_total, size=num_inst, replace=False)
    for k in range(num_inst):
        inst_all_grid[:,:,k] = obs_disp.reshape(ny,nx,order='F') + noise_all_grid[:,:,noise_idx_inst[k]]

    # save instantaneous deformation
    inst_file = os.path.join(out_dir, 'instantaneous_deformation.mat')
    sio.savemat(inst_file, {'inst_all_grid': inst_all_grid})
    print(f"✅ Instantaneous deformation + noise saved to {inst_file}")

    # ---------------- STEP 3: Temporal deformation + noise ----------------
    print("Generating temporal deformation...")
    ts_all_cube = np.zeros((ny, nx, num_times))
    noise_idx_ts = np.random.choice(noise_total, size=num_times, replace=False)

    temp_file = os.path.join(out_dir, 'temp_def.mat')
    simulate_temporal_deformation(obs_disp, nx, ny, num_times, T0, deformation_type, save_name=temp_file)
    ts_mat = sio.loadmat(temp_file)
    ts_cube = ts_mat['ts_cube']  # ny x nx x num_times

    for t in range(num_times):
        # each temporal step add one randomly selected noise group
        ts_all_cube[:,:,t] = ts_cube[:,:,t] + noise_all_grid[:,:,noise_idx_ts[t]]

    # save temporal deformation
    ts_file = os.path.join(out_dir, 'temporal_deformation.mat')
    sio.savemat(ts_file, {'ts_all_cube': ts_all_cube})
    print(f"✅ Temporal deformation + noise saved to {ts_file}")

    # ---------------- STEP 4: Plot noise ----------------
    plot_groups_noise = min(num_noise_plot, noise_total)
    noise_idx_plot = np.random.choice(noise_total, size=plot_groups_noise, replace=False)
    plot_3D_fields([noise_all_grid[:,:,i] for i in noise_idx_plot],
                   colormap='coolwarm',
                   title_prefix='Noise')

    # ---------------- STEP 5: Plot instantaneous deformation + noise ----------------
    plot_3D_fields([inst_all_grid[:,:,i] for i in range(num_inst)],
                   colormap='coolwarm',
                   title_prefix='Instantaneous + Noise')

    # ---------------- STEP 6: Plot temporal deformation + noise ----------------
    plot_3D_fields([ts_all_cube[:,:,i] for i in range(num_times)],
                   colormap='coolwarm',
                   title_prefix='Temporal + Noise')

    print("✅ Full simulation completed.")
