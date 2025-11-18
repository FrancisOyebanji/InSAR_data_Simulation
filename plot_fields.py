#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_fields.py
Imagesc-style plotting with coordinate axes
"""

import matplotlib.pyplot as plt
import math
import numpy as np

def plot_3D_fields(field_list, X_list=None, Y_list=None, colormap='coolwarm', title_prefix='Field'):
    """
    Plot multiple 2D fields in a grid with coordinate axes.

    Parameters
    ----------
    field_list : list of 2D arrays
        List of 2D field arrays (ny x nx).
    X_list, Y_list : list of 2D arrays, optional
        Corresponding coordinate matrices for each field.
        If None, indices are used.
    colormap : str
        Colormap for plotting.
    title_prefix : str
        Prefix for subplot titles.
    """
    num_fields = len(field_list)
    cols = math.ceil(math.sqrt(num_fields))
    rows = math.ceil(num_fields/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), constrained_layout=True)
    axes = axes.flatten()

    # Determine global min/max for color normalization
    cmin = min(f.min() for f in field_list)
    cmax = max(f.max() for f in field_list)

    for i, f in enumerate(field_list):
        if X_list is not None and Y_list is not None:
            Xmat = X_list[i]
            Ymat = Y_list[i]
            axes[i].imshow(f, origin='lower', cmap=colormap, vmin=cmin, vmax=cmax,
                           extent=[Xmat.min(), Xmat.max(), Ymat.min(), Ymat.max()],
                           aspect='auto')
        else:
            axes[i].imshow(f, origin='lower', cmap=colormap, vmin=cmin, vmax=cmax,
                           aspect='auto')

        axes[i].set_title(f'{title_prefix} {i+1}', fontsize=10)
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].grid(True, linestyle='--', alpha=0.3)

    for j in range(num_fields, len(axes)):
        axes[j].axis('off')

    # Add a single colorbar for all subplots
    plt.colorbar(plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=cmin, vmax=cmax)),
                 ax=axes[:num_fields].tolist(), fraction=0.02, pad=0.02)
    plt.show()
