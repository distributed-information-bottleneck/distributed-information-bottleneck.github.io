"""
Visualization:
- Distributed information plane
- Compression scheme confusion matrices from feature values
"""

import os
import matplotlib.pyplot as plt
import numpy as np 
import utils

default_mpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def save_compression_matrices(feature_encoder, inp_features, out_fname, inp_features_raw=None, feature_label=None, max_number_to_display=128):
  if inp_features_raw is None:
    inp_features_raw = inp_features
  unique_raw_vals, unique_raw_indices = np.unique(inp_features_raw, return_index=True)
  if len(unique_raw_vals) < 10:  # display the histogram as a barchart instead, and just eval those unique vals
    display_histogram = True
    sorted_features_raw = np.sort(unique_raw_vals)
    feature_val_counts = [np.average(inp_features_raw == unique_raw_val) for unique_raw_val in unique_raw_vals]
    feature_inp_inds = unique_raw_indices[np.argsort(unique_raw_vals)]
  else:
    display_histogram = False
    # Grab randomly from across the set
    random_selection_inds = np.random.choice(inp_features_raw.shape[0], max_number_to_display)
    feature_inp_inds = np.argsort(inp_features_raw[random_selection_inds])
    sorted_features_raw = np.sort(inp_features_raw[random_selection_inds])

  feature_inps = tf.gather(inp_features, feature_inp_inds, axis=0)
  emb_mus, emb_logvars = tf.split(feature_encoder(feature_inps), 2, axis=-1)
  
  bhat_distance_matrix = utils.bhattacharyya_dist_mat(emb_mus, emb_logvars, emb_mus, emb_logvars)
  compression_matrix = np.exp(-bhat_distance_matrix)

  fig = plt.figure(figsize=(6, 6))
  gs = fig.add_gridspec(2, 2,  width_ratios=(1, 2), height_ratios=(1, 2),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)
  ax = fig.add_subplot(gs[1, 1])
  ax.imshow(compression_matrix, vmin=0, vmax=1, cmap='Blues_r')
  
  plt.axis('off')
  ax2 = fig.add_subplot(gs[1, 0])
  if display_histogram:
    ax2.barh(sorted_features_raw, feature_val_counts, height=0.8)
    ax2.set_xticks([])
    ax2.set_xlim(0, 1)
    ax2.spines['bottom'].set_visible(False)
  else:
    ax2.plot(sorted_features_raw, np.arange(n), 'k', lw=3)
    ax2.set_ylim(n, 0)
    ax2.set_yticks([])
  ax2.spines['right'].set_visible(False)
  ax2.spines['top'].set_visible(False)
  ax2.spines['left'].set_visible(False)

  ax3 = fig.add_subplot(gs[0, 1])
  if display_histogram:
    ax3.bar(sorted_features_raw,feature_val_counts, width=0.8)
    ax3.set_yticks([])
    ax3.set_ylim(0, 1)
    ax3.spines['left'].set_visible(False)

  else:
    ax3.plot(np.arange(n), sorted_features_raw, 'k', lw=3)
    ax3.set_xlim(0, n)
    ax3.set_xticks([])
  ax3.spines['bottom'].set_visible(False)
  ax3.spines['right'].set_visible(False)
  ax3.spines['top'].set_visible(False)

  ax0 = fig.add_subplot(gs[0, 0])
  ax0.text(0, 0, feature_label)
  ax0.set_xlim(-0.5, 0.5)
  ax0.set_ylim(-0.5, 0.5)
  plt.axis('off')

  plt.savefig(out_fname)
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
  if number_features > 1:
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

