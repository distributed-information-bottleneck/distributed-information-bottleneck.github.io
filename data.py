"""
Data loading and preprocessing for:
- Boolean circuit
- Glassy rearrangement
- Mona Lisa (and other images)
- Double pendulum
- Tabular datasets

Coming soon!
"""
import numpy as np  
import tensorflow as tf  

def fetch_boolean_circuit(**kwargs):

  generate_random_circuit = kwargs.get('boolean_random_circuit', False)
  number_input_gates = kwargs.get('boolean_number_input_gates', 10)
  gates = [np.logical_and, np.logical_or, np.logical_xor]

  if generate_random_circuit:
    possible_inputs = [i for i in range(number_input_gates)]
    circuit_specification = possible_inputs.copy()
    while len(possible_inputs) > 1:
      # pick a gate at random
      gate_index = np.random.choice(len(gates))
      inps = np.random.choice(possible_inputs, size=2, replace=False)
      possible_inputs.append(len(circuit_specification))
      del possible_inputs[possible_inputs.index(inps[0])]
      del possible_inputs[possible_inputs.index(inps[1])]
      circuit_specification.append([gate_index, inps[0], inps[1]])
  else:
    # This is the circuit from the paper, formatted such that each intermediate output is defined by the contents of the brackets: [gate_id, input1, input2]
    circuit_specification = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, [1, 0, 1], [2, 8, 7], [0, 4, 3], [1, 11, 5], [2, 6, 12], [2, 13, 9], [1, 14, 10], [0, 15, 2], [0, 17, 16]]
    number_input_gates = 10

  def apply_gates(inputs):
    intermed = inputs
    for connection_spec in circuit_specification[inputs.shape[-1]:]:
      intermed = np.concatenate([intermed, np.int32(gates[connection_spec[0]](intermed[:, connection_spec[1]], intermed[:, connection_spec[2]]))[:, np.newaxis]], -1)
    return intermed

  # Evaluate the full truth table
  possible_inputs = np.meshgrid(*[[0, 1]]*number_input_gates)
  possible_inputs = np.stack(possible_inputs, -1)
  possible_inputs = np.reshape(possible_inputs, [-1, number_input_gates])

  truth_table = apply_gates(possible_inputs)

  x_train = 2*truth_table[:, :number_input_gates] - 1  ## Take x -> [-1, 1] to help w training
  y_train = truth_table[:, -1]

  x_valid = y_valid = x_valid_raw = None

  feature_dimensionalities = [1] * number_input_gates
  output_dimensionality = 1
  output_activation_fn = None
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  loss_is_info_based = True 
  metrics = ['accuracy']

  return dict(
  	x_train=x_train,
  	y_train=y_train,
  	x_valid=x_valid,
  	y_valid=y_valid,
  	x_valid_raw=x_valid_raw,
  	feature_dimensionalities=feature_dimensionalities,
  	number_features=len(feature_dimensionalities),
  	output_dimensionality=output_dimensionality,
  	output_activation_fn=output_activation_fn,
  	loss=loss,
  	loss_is_info_based=loss_is_info_based,
  	metrics=metrics,
  	)

def load(dataset, **kwargs):
	if dataset == 'boolean_circuit':
		return fetch_boolean_circuit(**kwargs)