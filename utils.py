"""
General utils: 
- Functions for calculating similarities for InfoNCE
- The Bhattacharyya calculation for quantifying the distinguishability of representations
"""
import tensorflow as tf
import numpy as np

@tf.function
def pairwise_l2_distance(pts1, pts2):
  """Computes squared L2 distances between each element of each set of points.
  
  Args:
    pts1: [N, d] tensor of points.
    pts2: [M, d] tensor of points.
  Returns:
    distance_matrix: [N, M] tensor of distances.
  """
  norm1 = tf.reduce_sum(tf.square(pts1), axis=-1, keepdims=True)
  norm2 = tf.reduce_sum(tf.square(pts2), axis=-1)
  norm2 = tf.expand_dims(norm2, -2)
  distance_matrix = tf.maximum(
      norm1 + norm2 - 2.0 * tf.matmul(pts1, pts2, transpose_b=True), 0.0)
  return distance_matrix


@tf.function
def pairwise_l1_distance(pts1, pts2):
  """Computes L1 distances between each element of each set of points.
  
  Args:
    pts1: [N, d] tensor of points.
    pts2: [M, d] tensor of points.
  Returns:
    distance_matrix: [N, M] tensor of distances.
  """
  stack_size2 = pts2.shape[0]
  pts1_tiled = tf.tile(tf.expand_dims(pts1, 1), [1, stack_size2, 1])
  distance_matrix = tf.reduce_sum(tf.abs(pts1_tiled-pts2), -1)
  return distance_matrix


@tf.function
def pairwise_linf_distance(pts1, pts2):
  """Computes Chebyshev distances between each element of each set of points.
  
  The Chebyshev/chessboard distance is the L_infinity distance between two
  points, the maximum difference between any of their dimensions.
  Args:
    pts1: [N, d] tensor of points.
    pts2: [M, d] tensor of points.
  Returns:
    distance_matrix: [N, M] tensor of distances.
  """
  stack_size2 = pts2.shape[0]
  pts1_tiled = tf.tile(tf.expand_dims(pts1, 1), [1, stack_size2, 1])
  distance_matrix = tf.reduce_max(tf.abs(pts1_tiled-pts2), -1)
  return distance_matrix


def get_scaled_similarity(embeddings1,
                          embeddings2,
                          similarity_type,
                          temperature):
  """Returns matrix of similarities between two sets of embeddings.
  
  Similarity is a scalar relating two embeddings, such that a more similar pair
  of embeddings has a higher value of similarity than a less similar pair.  This
  is intentionally vague to emphasize the freedom in defining measures of
  similarity. For the similarities defined, the distance-related ones range from
  -inf to 0 and cosine similarity ranges from -1 to 1.
  Args:
    embeddings1: [N, d] float tensor of embeddings.
    embeddings2: [M, d] float tensor of embeddings.
    similarity_type: String with the method of computing similarity between
      embeddings. Implemented:
        l2sq -- Squared L2 (Euclidean) distance
        l2 -- L2 (Euclidean) distance
        l1 -- L1 (Manhattan) distance
        linf -- L_inf (Chebyshev) distance
        cosine -- Cosine similarity, the inner product of the normalized vectors
    temperature: Float value which divides all similarity values, setting a
      scale for the similarity values.  Should be positive.
  Returns:  
    distance_matrix: [N, M] tensor of similarities.
  Raises:
    ValueError: If the similarity type is not recognized.
  """
  eps = 1e-9
  if similarity_type == 'l2sq':
    similarity = -1.0 * pairwise_l2_distance(embeddings1, embeddings2)
  elif similarity_type == 'l2':
    # Add a small value eps in the square root so that the gradient is always
    # with respect to a nonzero value.
    similarity = -1.0 * tf.sqrt(
        pairwise_l2_distance(embeddings1, embeddings2) + eps)
  elif similarity_type == 'l1':
    similarity = -1.0 * pairwise_l1_distance(embeddings1, embeddings2)
  elif similarity_type == 'linf':
    similarity = -1.0 * pairwise_linf_distance(embeddings1, embeddings2)
  elif similarity_type == 'cosine':
    embeddings1, _ = tf.linalg.normalize(embeddings1, ord=2, axis=-1)
    embeddings2, _ = tf.linalg.normalize(embeddings2, ord=2, axis=-1)
    similarity = tf.matmul(embeddings1, embeddings2, transpose_b=True)
  else:
    raise ValueError('Similarity type not implemented: ', similarity_type)

  similarity /= temperature
  return similarity



def bhattacharyya_dist_mat_multivariate(mus1, logvars1, mus2, logvars2):
  """Computes Bhattacharyya distances between multivariate Gaussians.

  Args:
    mus1: [N, d] float array of the means of the Gaussians.
    logvars1: [N, d] float array of the log variances of the Gaussians (so we're assuming diagonal 
    covariance matrices; these are the logs of the diagonal).
    mus2: [M, d] float array of the means of the Gaussians.
    logvars2: [M, d] float array of the log variances of the Gaussians.
  Returns:
    [N, M] array of distances.
  Raises:
    ValueError: If the similarity type is not recognized.
  """
  N = mus1.shape[0]
  M = mus2.shape[0]
  embedding_dimension = mus1.shape[1]
  assert (mus2.shape[1] == embedding_dimension)

  ## Manually broadcast in case either M or N is 1
  mus1 = np.tile(mus1[:, np.newaxis], [1, M, 1])
  logvars1 = np.tile(logvars1[:, np.newaxis], [1, M, 1])
  mus2 = np.tile(mus2[np.newaxis], [N, 1, 1])
  logvars2 = np.tile(logvars2[np.newaxis], [N, 1, 1])
  difference_mus = mus1 - mus2  # [N, M, embedding_dimension]; we want [N, M, embedding_dimension, 1]
  difference_mus = difference_mus[..., np.newaxis]
  difference_mus_T = np.transpose(difference_mus, [0, 1, 3, 2])

  sigma_diag = 0.5 * (np.exp(logvars1) + np.exp(logvars2))  ## [N, M, embedding_dimension], but we want a diag mat [N, M, embedding_dimension, embedding_dimension]
  sigma_mat = np.apply_along_axis(np.diag, -1, sigma_diag)
  sigma_mat_inv = np.apply_along_axis(np.diag, -1, 1./sigma_diag)

  determinant_sigma = np.prod(sigma_diag, axis=-1)
  determinant_sigma1 = np.exp(np.sum(logvars1, axis=-1))
  determinant_sigma2 = np.exp(np.sum(logvars2, axis=-1))
  term1 = 0.125 * np.squeeze( (difference_mus_T @ sigma_mat_inv @ difference_mus) ).reshape([N, M])
  term2 = 0.5 * np.log(determinant_sigma / np.sqrt(determinant_sigma1 * determinant_sigma2))
  return term1+term2
