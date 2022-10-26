"""
One stop shop for training on any of the datasets in the Distributed IB papers.
"""

import argparse

import utils, data, models


def get_args():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--dataset', default='bikeshare',
  	help='Choose the dataset.',
  	choices=['boolean_circuit', 'mona_lisa', 'glass',
  	'pendulum',
  	'microsoft', 'mimic2', 'mimic3', 'wine', 'bikeshare'])
  parser.add_argument('--datadir', type=str, default='./data/')
  parser.add_argument('--outdir', type=str, default='./training_artifacts/')
  parser.add_argument('--ib', type=bool, default=False, 
  	help='Whether to train as the basic information bottleneck, where'+
  	' all features are processed together into a single bottleneck.' + 
  	' Intended for the pendulum information loss and the DIB/IB Mona Lisa comparison.')
  parser.add_argument('--lr', type=float, default=0.02)
  parser.add_argument('--beta_start', type=float, default=1e-4, 
  	help='The bottleneck strength beta at the start of training. Recommend starting small and increasing.')
  parser.add_argument('--beta_end', type=float, default=3e0, 
  	help='The bottleneck strength at the end of training. Recommend passing 1 for information-based error metric' + 
  	' (e.g. cross entropy) or finding a suitable range otherwise (ending after all KL->0).')
  parser.add_argument('--number_pretraining_steps', type=int, default=10**4,
  	help='The number of training steps to warm up where beta=beta_start.')
  parser.add_argument('--number_annealing_steps', type=int, default=10**5,
  	help='The number of training steps to anneal beta from beta_start to beta_end.')
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--use_positional_encoding', type=bool, default=True,
  	help='Whether to preprocess each feature with a positional encoding layer.')
  parser.add_argument('--activation_fn', type=str, default='relu', help='Activation function for the feature encoders and'+
  	' integration network. Can be None.') 
  parser.add_argument('--feature_embedding_dimension', type=int, default=32, 
  	help='The embedding space dimensionality for each feature.')
  parser.add_argument('--optimizer', type=str, default='adam')
  parser.add_argument('--save_distinguishability_matrices_frequency', type=int, default=0,
  	help='Whether and how often to save distinguishability matrices, which show where the network' + 
  	' is allocating information.')

  parser.add_argument('--feature_encoder_architecture') 
  parser.add_argument('--positional_encoding_frequencies') 
  parser.add_argument('--integration_network_architecture')


  args = parser.parse_args()
  return args


def main():
  args = get_args()

  ## Load the data
  dataset_dict = data.load(args.dataset)

  print('Data loaded.')
  if args.ib:
  	# Extremely simple change: just treat everything as one feature
  	dataset_dict['feature_dimensionalities'] = [np.sum(dataset_dict['feature_dimensionalities'])]

  ## Build the model
  model = models.DistributedMLP(dataset_dict['feature_dimensionalities'],
    args.encoder_architecture,
    args.integration_network_architecture,
    dataset_dict['output_dimensionality'],
    use_positional_encoding=args.use_positional_encoding,
    positional_encoding_frequencies=args.positional_encoding_frequencies,
    activation_fn=args.activation_fn,
    feature_embedding_dimension=args.feature_embedding_dimension,
    output_activation_fn=dataset_dict['output_activation_fn'])

  optimizer = tf.keras.optimizers.get(args.optimizer)
  optimizer.lr = args.lr
  model.compile(
  	optimizer=optimizer,
  	loss=dataset_dict['loss'],
  	metrics=dataset_dict['metrics'],
  	)

  callbacks = [models.InfoBottleneckAnnealingCallback(
  	args.beta_start, args.beta_end, args.number_pretraining_steps, args.number_annealing_steps)]

  if args.save_distinguishability_matrices_frequency > 0:
  	callbacks.append(models.SaveDistinguishabilityMatricesCallback(
  		args.save_distinguishability_matrices_frequency,
  		dataset_dict['x_valid'],
  		dataset_dict['x_valid_raw'],
  		args.outdir))

  number_trainable_variables = 0
  for tensor in model.trainable_variables:
  	number_trainable_variables += np.product(tensor.shape)
  print(f'Number of trainable variables: {number_trainable_variables}')

  ## Train
  print('Model built, starting to train.')

  number_epochs = args.number_pretraining_steps + args.number_annealing_steps

  history = model.fit(
  	dataset_dict['x_train'],
  	dataset_dict['y_train'],
  	epochs=number_epochs,
  	shuffle=True,
  	batch_size=args.batch_size,
  	callbacks=[beta_annealing_callback],
  	verbose=False,
  	validation_data=(dataset_dict['x_valid'], dataset_dict['y_valid']),
  	)

  print('Finished training.')
  ## Inspect the training trajectory

  beta_series = np.float32(history.history['beta'])
  kl_series = np.stack([history.history[f'KL{feature_ind}'] for feature_ind in range(dataset_dict['number_features'])], -1)
  loss_series = np.float32(history.history['loss'])
  ## Get the original loss without the KL term
  loss_series -= beta_series * np.sum(kl_series, axis=-1)

  kl_series /= np.log(2)  ## get the KL values in bits
  if dataset_dict['loss_is_info_based']:
  	loss_series /= np.log(2)  ## get the loss values in bits

  visualization.save_distributed_info_plane(beta_series, kl_series, loss_series, outdir=args.outdir)


if __name__ == '__main__':
    main()