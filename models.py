"""
Models:
- The positional encoding layer for puffing out the inputs to help with training
- Distributed IB implementation
- Keras callback to anneal the value of \beta (the bottleneck strength)
"""

import tensorflow as tf
import numpy as np
import utils

class PositionalEncoding(tf.keras.layers.Layer):
  """Simple positional encoding layer, that appends to an input sinusoids of multiple frequencies.
  """
  def __init__(self, frequencies):
    super(PositionalEncoding, self).__init__()
    self.frequencies = frequencies

  def build(self, input_shape):
    return

  def call(self, inputs):  # Defines the computation from inputs to outputs
    return tf.concat([inputs] + [tf.math.sin(frequency*inputs) for frequency in self.frequencies], -1)


class DistributedIBNet(tf.keras.Model):
  """Distributed IB implementation where each feature is passed through an MLP

  Args:
    feature_dimensionalities: List of ints specifying the dimension of each feature.
      E.g. if the first feature is just a scalar and the second is a one-hot with 4 values, 
      the feature_dimensionalities should be [1, 4].
    feature_encoder_architecture: List of ints specifying the number of units in each layer of the 
      feature encoders. E.g. [64, 128] specifies that each feature encoder has 64 units in 
      the first layer and 128 in the second.
    integration_network_architecture: List of ints specifying the architecture of the MLP that
      integrates all of the feature embeddings into a prediction.
    output_dimensionality: Int specifying the dimensionality of the output. If the task is 
      scalar regression, output_dimensionality=1; if it's classification, 
      output_dimensionality=number_classes; etc.
    use_positional_encoding: Boolean specifying whether to preprocess each feature with a 
      positional encoding layer.  Helps with training when using low-dimensional features.
      Default: True.
    number_positional_encoding_frequencies: Number of frequencies to use in the
      positional encoding layer; a range of powers of 2.
      Default: 5, which we found to work well across the board.
    activation_fn: The activation function to use in the feature encoders and integration
      network, with the exception of the output layer.
      Default: relu.
    feature_embedding_dimension: Int specifying the embedding space dimensionality for each
      feature. 
      Default: 32.
    output_activation_fn: The activation function for the output. 
      Default: None.
  """
  def __init__(self, 
    feature_dimensionalities,
    feature_encoder_architecture,
    integration_network_architecture,
    output_dimensionality,
    use_positional_encoding=True,
    number_positional_encoding_frequencies=5,
    activation_fn='relu',
    feature_embedding_dimension=32,
    output_activation_fn=None,
    ):
      super(DistributedIBNet, self).__init__()
      self.feature_dimensionalities = feature_dimensionalities
      self.number_features = len(feature_dimensionalities)
      positional_encoding_frequencies = 2**np.arange(1, number_positional_encoding_frequencies)
      feature_encoders = []
      for feature_dimensionality in feature_dimensionalities:
        feature_encoder_layers = [tf.keras.layers.Input((feature_dimensionality,))]
        if use_positional_encoding:
          feature_encoder_layers += [PositionalEncoding(positional_encoding_frequencies)]
        feature_encoder_layers += [tf.keras.layers.Dense(number_units, activation_fn) for number_units in feature_encoder_architecture]
        feature_encoder_layers += [tf.keras.layers.Dense(2*feature_embedding_dimension)]
        feature_encoders.append(tf.keras.Sequential(feature_encoder_layers))
      self.feature_encoders = feature_encoders 

      integration_network_layers = [tf.keras.layers.Input((len(feature_dimensionalities)*feature_embedding_dimension,))]
      integration_network_layers += [tf.keras.layers.Dense(number_units, activation_fn) for number_units in integration_network_architecture]
      integration_network_layers += [tf.keras.layers.Dense(output_dimensionality, output_activation_fn)]
      self.integration_network = tf.keras.Sequential(integration_network_layers)

      self.beta = tf.Variable(1., dtype=tf.float32, trainable=False)

  def build(self, input_shape):
    assert input_shape[-1] == np.sum(self.feature_dimensionalities)
    for feature_ind in range(self.number_features):
      self.feature_encoders[feature_ind].build(input_shape[:-1]+[self.feature_dimensionalities[feature_ind]])

    self.integration_network.build()
    return

  def call(self, inputs):
    # Encode each feature into a Gaussian in embedding space
    # Evaluate the KL divergence of the Gaussian from the unit normal Gaussian (the prior)
    # Sample from the Gaussian for the reparameterization trick
    # Gather the samples from all the features, and pass through the integration network for the final prediction
    features_split = tf.split(inputs, self.feature_dimensionalities, axis=-1)

    feature_embeddings, kl_divergence_channels = [[], []]

    for feature_ind in range(self.number_features):
      emb_mus, emb_logvars = tf.split(self.feature_encoders[feature_ind](features_split[feature_ind]), 2, axis=-1)

      emb_channeled = tf.random.normal(emb_mus.shape, mean=emb_mus, stddev=tf.exp(emb_logvars/2.))

      feature_embeddings.append(emb_channeled)
      kl_divergence_channel = tf.reduce_mean(
        tf.reduce_sum(0.5 * (tf.square(emb_mus) + tf.exp(emb_logvars) - emb_logvars - 1.), axis=-1))
      kl_divergence_channels.append(kl_divergence_channel)
      # Add a metric to track the KL divergence per feature over the course of training, since it's not automatic
      self.add_metric(kl_divergence_channel, name=f'KL{feature_ind}')

    # This is the bottleneck: a loss contribution based on the total KL across channels (one per feature)
    self.add_loss(self.beta*tf.reduce_sum(kl_divergence_channels))

    # Add another metric to store the value of the bottleneck strength \beta over training
    self.add_metric(self.beta, name='beta')
    prediction = self.integration_network(tf.concat(feature_embeddings, -1))
    return prediction

class InfoBottleneckAnnealingCallback(tf.keras.callbacks.Callback):
  """Callback to logarithmically increase beta during training.

  Args:
    beta_start: The value of beta at the start of annealing.
    beta_end: The value of beta at the end of annealing.
    number_pretraining_epochs: The number of epochs to hold beta=beta_start
      at the beginning of training.
    number_annealing_epochs: The number of epochs to logarithmically ramp beta from
      beta_start to beta_end.
  """

  def __init__(self, 
               beta_start,
               beta_end,
               number_pretraining_epochs,
               number_annealing_epochs):
    super(InfoBottleneckAnnealingCallback, self).__init__()
    self.beta_start = beta_start
    self.beta_end = beta_end 
    self.number_pretraining_epochs = number_pretraining_epochs
    self.number_annealing_epochs = number_annealing_epochs
  def on_epoch_begin(self, epoch, logs=None):
    self.model.beta.assign(tf.exp(tf.math.log(self.beta_start)+
      tf.cast(max(epoch-self.number_pretraining_epochs, 0), tf.float32)/self.number_annealing_epochs*(tf.math.log(self.beta_end)-tf.math.log(self.beta_start))))


class SaveCompressionMatricesCallback(tf.keras.callbacks.Callback):
  """Callback to save compression scheme matrices during training.

  Assume one dimensional feature data, and compute the distinguishability
  matrices for a sample of the data.
  Args:
    save_frequency: The number of epochs between each save, since it takes a little while.
    x_processed: The input values to pass through the actual 
    x_raw: The raw data values, to use when displaying the range of values.
    outdir: The directory in which to save the images.
  """

  def __init__(self, 
               save_frequency,
               x_processed,
               x_raw,
               outdir):
    super(SaveCompressionMatricesCallback, self).__init__()
    self.save_frequency = save_frequency
    self.x_processed = x_processed 
    self.x_raw = x_raw 
    self.outdir = outdir

  def on_epoch_end(self, epoch, logs=None):
    if (epoch % self.save_frequency) == 0:
      beta_value = self.model.beta.value()
      features_split = tf.split(self.x_processed, self.model.feature_dimensionalities, axis=-1)
      features_raw = tf.split(self.x_raw, self.model.feature_dimensionalities, axis=-1)
      for feature_ind in range(self.model.number_features):
        out_fname = os.path.join(self.outdir, f'feature_{feature_ind}_log10beta_{np.log10(beta_value):.3f}.png')
        visualization.save_compression_matrices(
          self.model.feature_encoders[feature_ind], 
          features_split[feature_ind], 
          out_fname, 
          inp_features_raw=features_split_raw[feature_ind])

class InfoPerFeatureCallback(tf.keras.callbacks.Callback):
  """Callback to compute the information contained in each compression channel (encoder) during training.

  
  Args:
    save_frequency: The number of epochs between each save, since it takes a little while.
    tf_dataset_validation: The validation dataset, complete with all featuers and 
      the relevance variable Y (which will be stripped away as it is not needed here).
    info_bound_batch_size: The number of data points to use for each batch when
      estimating the upper and lower bounds.  Increasing this parameter yields 
      tighter bounds on the mutual information.
    info_bound_number_batches: The number of batches over which to average the 
      upper and lower bounds.  Increasing this parameter reduces the uncertainty
      of the bounds.
  """
  def __init__(self, 
               save_frequency,
               tf_dataset_validation,
               info_bound_batch_size=1024,
               info_bound_number_batches=8):
    super(InfoPerFeatureCallback, self).__init__()
    self.save_frequency = save_frequency
    self.tf_dataset_validation = tf_dataset_validation.map(lambda x, y: x)
    self.bounds = []
    self.info_bound_batch_size = info_bound_batch_size
    self.info_bound_number_batches = info_bound_number_batches

  def on_epoch_end(self, epoch, logs=None):
    if (epoch % self.save_frequency) == 0:
      for feature_ind in range(self.model.number_features):
        infonce_lower, loo_upper = utils.estimate_mi_sandwich_bounds(
          self.model.feature_encoders[feature_ind], 
          self.tf_dataset_validation.map(lambda x: tf.split(x, self.model.feature_dimensionalities, axis=-1)[feature_ind]), 
          eval_batch_size=self.info_bound_batch_size, 
          num_eval_batches=self.info_bound_number_batches)
        self.bounds.append([infonce_lower, loo_upper])
