"""
Visualization:
- Distributed information plane
- Distinguishability matrices from feature values
"""

import os
import matplotlib.pyplot as plt
import numpy as np 

default_mpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def save_distinguishability_matrices(distinguishability_matrix, outdir, out_fname):
  plt.figure(figsize=(8, 8))
  saveto = os.path.join(outdir, out_fname)
  plt.savefig(saveto)
  plt.clf()
  return

def save_distributed_info_plane(kl_series, loss_series, outdir, entropy_y=None):
  number_features = kl_series.shape[1]
  desired_data_length = min(1000, kl_series.shape[0])
  sieve_factor = kl_series.shape[0]//desired_data_length
  info_in_plot_lims = [0, 15]

  plotting_start_ind = desired_data_length // 2


  approx_info_in_parts = kl_series[::sieve_factor]
  approx_info_in_full = np.sum(approx_info_in_parts, axis=-1)
  performance_out_train = loss_series[::sieve_factor]

  plt.figure(figsize=(8, 4))
  ax = plt.gca()
  ax.plot(approx_info_in_full[plotting_start_ind:], performance_out_train[plotting_start_ind:], lw=4, color='k')
  if entropy_y is not None:
    ax.plot(info_in_plot_lims, [entropy_y]*2, 'k:')
  ax.set_xlim(info_in_plot_lims)

  ax2 = ax.twinx()
  for feature_ind in range(number_features):
    ax2.plot(approx_info_in_full[plotting_start_ind:], 
      approx_info_in_parts[plotting_start_ind:, feature_ind], 
      color=default_mpl_colors[feature_ind%len(default_mpl_colors)], lw=4)
  ax.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
  ax.patch.set_visible(False) # hide the 'canvas'
  
  saveto = os.path.join(outdir, 'distributed_info_plane.png')
  plt.savefig(saveto, dpi=300)
  plt.clf()
  return

