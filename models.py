"""
Models:
- The positional encoding layer for puffing out the inputs to help with training
- An implementation of the distributed IB
"""

import tensorflow as tf
import numpy as np

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


class DistributedMLP(tf.keras.Model):
  """Distributed IB implementation where each feature is passed through an MLP
  """
  def __init__(self, 
    feature_dimensionalities,
    encoder_architecture,
    integration_network_architecture,
    output_dimensionality,
    use_positional_encoding=True,
    positional_encoding_frequencies=2**np.arange(5),
    activation_fn='relu',
    feature_embedding_dimension=32,
    output_activation_fn=None,
    ):
      super(DistributedMLP, self).__init__()
      self.feature_dimensionalities = feature_dimensionalities
      feature_encoders = []
      for feature_dimensionality in feature_dimensionalities:
        feature_encoder_layers = [tf.keras.layers.Input((feature_dimensionality,))]
        if use_positional_encoding:
          feature_encoder_layers += [PositionalEncoding(positional_encoding_frequencies)]
        feature_encoder_layers += [tf.keras.layers.Dense(number_units, activation_fn) for number_units in encoder_architecture]
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
    for feature_ind in range(len(self.feature_dimensionalities)):
      self.feature_encoders[feature_ind].build(input_shape[:-1]+[self.feature_dimensionalities[feature_ind]])

    self.integration_network.build()
    return

  def call(self, inputs, training=None):  # Defines the computation from inputs to outputs
    features_split = tf.split(inputs, self.feature_dimensionalities, axis=-1)

    feature_embeddings, kl_divergence_channels = [[], []]

    for feature_ind in range(len(self.feature_dimensionalities)):
      emb_mus, emb_logvars = tf.split(self.feature_encoders[feature_ind](features_split[feature_ind]), 2, axis=-1)
      if training:
        emb_channeled = tf.random.normal(emb_mus.shape, mean=emb_mus, stddev=tf.exp(emb_logvars/2.))
      else:
        emb_channeled = emb_mus

      feature_embeddings.append(emb_channeled)
      kl_divergence_channels.append(tf.reduce_mean(tf.reduce_sum(0.5 * (tf.square(emb_mus) + tf.exp(emb_logvars) - emb_logvars - 1.), axis=-1)))

    self.add_loss(self.beta*tf.reduce_sum(kl_divergence_channels))
    self.add_metric(kl_divergence_channels, name='KL')
    self.add_metric(self.beta, name='beta')
    prediction = self.integration_network(tf.concat(feature_embeddings, -1))
    return prediction

class InfoBottleneckAnnealingCallback(tf.keras.callbacks.Callback):
  def __init__(self, 
               beta_start,
               beta_end,
               number_pretraining_steps,
               number_annealing_steps):
    super(InfoBottleneckAnnealingCallback, self).__init__()
    self.beta_start = beta_start
    self.beta_end = beta_end 
    self.number_pretraining_steps = number_pretraining_steps
    self.number_annealing_steps = number_annealing_steps
  def on_epoch_begin(self, epoch, logs=None):
    self.model.beta.assign(tf.exp(tf.math.log(self.beta_start)+tf.cast(max(epoch-self.number_pretraining_steps, 0), tf.float32)/self.number_annealing_steps*(tf.math.log(self.beta_end)-tf.math.log(self.beta_start))))