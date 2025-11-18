# -*- coding: utf-8 -*-
"""
Generate InSAR deformation with multiple localized fault segments that taper off.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import random

# -------------------- USER SETTINGS --------------------
nx, ny = 58, 58
nt = 9
deformation_type = "fault"
output_dir = "insar_localized_refined"
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)
random.seed(42)

# -------------------- NOISE COMPONENTS --------------------
dem_error = gaussian_filter(np.random.randn(nx, ny), sigma=10) * 0.4

x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
xx, yy = np.meshgrid(x, y)
orbital_base = 0.2 * xx + 0.15 * yy

# -------------------- LOCALIZED FAULT MODEL --------------------
def localized_fault(x, y, xc, yc, fault_angle, fault_length, slip, width, taper_distance):
    """
    Create a localized fault segment with tapering deformation.
    
    Parameters:
    - xc, yc: center of fault segment
    - fault_angle: angle of fault in degrees
    - fault_length: length of the fault segment
    - slip: magnitude of displacement
    - width: transition width across the fault
    - taper_distance: distance over which deformation tapers to zero (in pixels)
    """
    angle_rad = np.radians(fault_angle)
    
    # Rotate coordinate system
    x_centered = x - xc
    y_centered = y - yc
    x_rot = x_centered * np.cos(angle_rad) + y_centered * np.sin(angle_rad)
    y_rot = -x_centered * np.sin(angle_rad) + y_centered * np.cos(angle_rad)
    
    # Distance perpendicular to fault (for slip pattern)
    distance_perpendicular = x_rot
    
    # Distance along fault (for length constraint)
    distance_along = y_rot
    
    # Create slip pattern using tanh
    deformation = slip * np.tanh(distance_perpendicular / width)
    
    # Apply length constraint (fault only exists within fault_length)
    length_mask = np.exp(-(distance_along ** 2) / (2 * (fault_length / 2) ** 2))
    deformation *= length_mask
    
    # Apply tapering based on distance from fault line
    # Convert taper_distance from pixels to coordinate space
    pixel_size = 2.0 / nx  # coordinate range is -1 to 1, so 2.0 total
    taper_coord = taper_distance * pixel_size
    
    # Distance from fault line
    distance_from_fault = np.abs(distance_perpendicular)
    
    # Stronger exponential taper for tighter falloff
    taper = np.exp(-2.0 * distance_from_fault / taper_coord)
    deformation *= taper
    
    return deformation

def mogi_source(x, y, xc=0, yc=0, depth=0.5, dV=1.0, nu=0.25):
    """Normalized Mogi deformation."""
    r2 = (x - xc) ** 2 + (y - yc) ** 2
    denom = (r2 + depth ** 2) ** 1.5
    uz = (1 - nu) * dV * depth / denom
    uz = uz / np.max(np.abs(uz)) if np.max(np.abs(uz)) != 0 else uz
    return uz

# -------------------- TIME LOOP --------------------
frames = []
ground_truth_amp = np.zeros((nx, ny))
ground_truth_mask = np.zeros((nx, ny))

# Generate multiple fault parameters (4-6 faults to match the pattern)
n_faults = random.randint(1, 3)
fault_params = []
# All faults share the same orientation
common_angle = np.random.uniform(0, 180)
for i in range(n_faults):
    params = {
        'xc': np.random.uniform(-0.7, 0.7),
        'yc': np.random.uniform(-0.7, 0.7),
        'angle': common_angle,  # Same angle for all faults
        'length': np.random.uniform(0.25, 0.45),  # Shorter fault segments
        'slip': np.random.uniform(0.7, 1.3),
        'width': np.random.uniform(0.06, 0.12),  # Sharper transition
        'taper_distance': 5  # pixels - tighter tapering
    }
    fault_params.append(params)

for t in range(nt):
    # Smooth correlated atmospheric noise
    atm = gaussian_filter(np.random.randn(nx, ny), sigma=8) * 0.8
    orbital = orbital_base * (1 + 0.1 * np.random.randn())
    decor = np.random.randn(nx, ny) * 0.05
    total = atm + dem_error + orbital + decor

    deformation = np.zeros_like(xx)

    if 1 <= t <= 8:
        if deformation_type == "fault":
            # Add multiple localized faults
            for params in fault_params:
                fault_def = localized_fault(
                    xx, yy,
                    xc=params['xc'],
                    yc=params['yc'],
                    fault_angle=params['angle'],
                    fault_length=params['length'],
                    slip=params['slip'] * (0.8 + 0.4 * np.random.rand()),
                    width=params['width'],
                    taper_distance=params['taper_distance']
                )
                
                # Slight smoothing for realism (less smoothing for sharper features)
                fault_def = gaussian_filter(fault_def, sigma=0.4)
                deformation += fault_def
            
        else:  # mogi
            n_sources = random.randint(2, 4)
            for _ in range(n_sources):
                xc = np.random.uniform(-0.4, 0.4)
                yc = np.random.uniform(-0.4, 0.4)
                dV = np.random.uniform(0.5, 2.0)
                depth = np.random.uniform(0.3, 0.6)
                local_def = mogi_source(xx, yy, xc, yc, depth, dV)
                local_def = gaussian_filter(local_def, sigma=1.2)
                local_def *= np.random.uniform(0.25, 0.5)
                deformation += local_def

        # Update ground truth
        ground_truth_amp += deformation
        
        total += deformation

    frames.append(total)

    # --- Save noisy frame ---
    plt.figure(figsize=(4, 4))
    plt.imshow(total, cmap="RdBu", origin="lower", vmin=-2, vmax=2)
    plt.colorbar(label="Phase (radians)")
    plt.title(f"Frame {t+1} ({'with deformation' if 1 <= t <= 7 else 'noise only'})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"frame_{t+1:02d}.png"))
    plt.close()

# Create binary mask based on significant deformation (stricter threshold)
threshold = 0.15 * np.max(np.abs(ground_truth_amp))
ground_truth_mask = (np.abs(ground_truth_amp) > threshold).astype(float)

# -------------------- SAVE RESULTS --------------------
frames = np.stack(frames, axis=0)
np.save(os.path.join(output_dir, "insar_with_refined_localized_deformation.npy"), frames)

# Save amplitude ground truth
plt.figure(figsize=(4, 4))
plt.imshow(ground_truth_amp, cmap="RdBu", origin="lower")
plt.colorbar(label="Phase (radians)")
plt.title("Ground Truth Deformation (cumulative)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ground_truth_amplitude.png"))
plt.close()
np.save(os.path.join(output_dir, "ground_truth_amplitude.npy"), ground_truth_amp)

# Save binary mask
plt.figure(figsize=(4, 4))
plt.imshow(ground_truth_mask, cmap="gray", origin="lower")
plt.colorbar(label="Mask (1 = deformation region)")
plt.title("Ground Truth Binary Mask")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ground_truth_mask.png"))
plt.close()
np.save(os.path.join(output_dir, "ground_truth_mask.npy"), ground_truth_mask)

print(f"âœ… Generated {nt} frames with {n_faults} fault segments.")
print(f"Saved results in '{output_dir}/'")
print(f"Fault parameters:")
for i, params in enumerate(fault_params):
   print(f"  Fault {i+1}: center=({params['xc']:.2f}, {params['yc']:.2f}), "
         f"angle={params['angle']:.1f}Â°, length={params['length']:.2f}")