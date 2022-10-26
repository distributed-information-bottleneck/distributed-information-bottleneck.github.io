"""
Visualization:
- Distributed information plane
- Distinguishability matrices from feature values

Coming soon!
"""

import os
import matplotlib.pyplot as plt


def save_distinguishability_matrices(distinguishability_matrix, outdir, out_fname):
  plt.figure(figsize=(8, 8))
  saveto = os.path.join(outdir, out_fname)
  plt.savefig(saveto)

def save_distributed_info_plane(beta_series, kl_series, loss_series, outdir):
  plt.figure(figsize=(12, 8))
  
  saveto = os.path.join(outdir, 'distributed_info_plane.png')
  plt.savefig(saveto)

