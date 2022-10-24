"""
Models:
- The positional encoding layer for puffing out the inputs to help with training
"""

import tensorflow as tf

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
