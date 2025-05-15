import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap



def plot_velocity(output, target, path, vmin=None, vmax=None):
    # diff = np.abs(target - output) / target
    diff = np.abs(target - output) / (-1*target)
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))  # Increased figure height

    if vmin is None or vmax is None:
        vmax, vmin = np.max(target), np.min(target)

    # Plot for Prediction
    im1 = ax[0].matshow(output, cmap='jet', vmin=vmin, vmax=vmax)
    ax[0].set_title('Prediction', fontsize=12)
    fig.colorbar(im1, ax=ax[0], orientation='horizontal', fraction=0.046, pad=0.06, label='P-wave (km/s)')

    # Plot for Ground Truth
    im2 = ax[1].matshow(target, cmap='jet', vmin=vmin, vmax=vmax)
    ax[1].set_title('Ground Truth', fontsize=12)
    fig.colorbar(im2, ax=ax[1], orientation='horizontal', fraction=0.046, pad=0.06, label='P-wave (km/s)')

    # Plot for Difference
    im3 = ax[2].matshow(diff, cmap='jet', vmin=0.0, vmax=0.1)

    ax[2].set_title('Difference', fontsize=12)
    fig.colorbar(im3, ax=ax[2], orientation='horizontal', fraction=0.046, pad=0.06, label='Normalized Difference')

    # Set common labels
    for axis in ax:
        axis.set_ylabel('z (km)', fontsize=10)
        axis.set_xlabel('x (km)', fontsize=10)

    plt.tight_layout()
    plt.savefig(path)
    plt.close('all')
