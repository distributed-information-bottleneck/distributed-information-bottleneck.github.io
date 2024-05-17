"""
One stop shop for training on any of the datasets in the Distributed IB papers.
"""

import argparse
import tensorflow as tf  
import numpy as np  
import os 

import utils, data, models, visualization

def get_args():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--dataset', default='boolean_circuit',
  	help='Choose the dataset.',
  	choices=['boolean_circuit', 'double_pendulum', 'mice_protein',
    'microsoft', 'credit', 'support2', 'wine', 'bikeshare'])
  parser.add_argument('--data_path', type=str, default='./data/')
  parser.add_argument('--artifact_outdir', type=str, default='./training_artifacts/')
  parser.add_argument('--ib', type=bool, default=False, 
  	help='Whether to train as the basic information bottleneck, where'+
  	' all features are processed together into a single bottleneck.' + 
  	' Intended for the pendulum information loss and the DIB/IB Mona Lisa comparison.')
  parser.add_argument('--learning_rate', type=float, default=3e-4)
  parser.add_argument('--beta_start', type=float, default=1e-4, 
  	help='The bottleneck strength beta at the start of training. Recommend starting small and increasing.')
  parser.add_argument('--beta_end', type=float, default=3e0, 
  	help='The bottleneck strength at the end of training. Recommend passing 1 for information-based error metric' + 
  	' (e.g. cross entropy) or finding a suitable range otherwise (ending after all KL->0).')
  parser.add_argument('--number_pretraining_epochs', type=int, default=10**3,
  	help='The number of training epochs to warm up where beta=beta_start.')
  parser.add_argument('--number_annealing_epochs', type=int, default=10**4,
  	help='The number of training epochs to anneal beta from beta_start to beta_end.')
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--use_positional_encoding', type=bool, default=True,
  	help='Whether to preprocess each feature with a positional encoding layer.')
  parser.add_argument('--activation_fn', type=str, default='relu', help='Activation function for the feature encoders and'+
  	' integration network. Can be None.') 
  parser.add_argument('--feature_embedding_dimension', type=int, default=32, 
  	help='The embedding space dimensionality for each feature.')
  parser.add_argument('--optimizer', type=str, default='adam')
  parser.add_argument('--save_compression_matrices_frequency', type=int, default=0,
  	help='Whether and how often to save compression matrices, which show where the network' + 
  	' is allocating information.')

  parser.add_argument('--feature_encoder_architecture', type=int, nargs='+', default=[128, 128],
    help='The specs for each feature encoder MLP.') 
  parser.add_argument('--number_positional_encoding_frequencies', type=int, default=5,
    help='The number of sinusoid frequencies used to puff out the low-dimensional feature data pre-MLP.') 
  parser.add_argument('--integration_network_architecture', type=int, nargs='+', default=[256, 256],
    help='The specs for the integration network that does the prediction using all the embeddings.')

  parser.add_argument('--infonce_loss', type=bool, default=False,
    help='Whether to train with an infoNCE loss to match up X and Y in a shared embedding space.')
  parser.add_argument('--infonce_shared_dimensionality', type=int, default=64,
    help='If training with infoNCE, this is the dimensionality of the shared embedding space.')
  parser.add_argument('--infonce_y_encoder_architecture', type=int, nargs='+', default=[128, 128],
    help='The specs for the Y encoder MLP.')
  parser.add_argument('--infonce_similarity', type=str, default='l2',
    help='The similarity measure to use in the shared embedding space for computing the infonce loss.')
  parser.add_argument('--infonce_temperature', type=float, default=1.,
    help='The temperature to rescale the similarities for use with infonce; only really impactful for cosine similarity.')

  ## Dataset specific
  parser.add_argument('--boolean_random_circuit', type=bool, default=False,
    help='Whether to generate a random boolean circuit for training, or use the one from the paper.')
  parser.add_argument('--boolean_number_input_gates', type=int, default=10,
    help='If training with a random boolean circuit: how many input gates to use.')

  parser.add_argument('--pendulum_time_delta', type=float, default=2,
    help='The time delay for predicting the future state of the double pendulum.')

  args = parser.parse_args()
  return args


def main():
  args = get_args()

  number_pretraining_epochs = args.number_pretraining_epochs
  number_annealing_epochs = args.number_annealing_epochs
  number_epochs = number_pretraining_epochs + number_annealing_epochs

  batch_size = args.batch_size

  beta_start = args.beta_start
  beta_end = args.beta_end

  use_positional_encoding = args.use_positional_encoding
  number_positional_encoding_frequencies = args.number_positional_encoding_frequencies
  activation_fn = args.activation_fn

  artifact_outdir = args.artifact_outdir
  if not os.path.exists(artifact_outdir):
    os.makedirs(artifact_outdir)
  save_compression_matrices_frequency = args.save_compression_matrices_frequency

  ## Load the data
  kwargs=dict(
    data_path=args.data_path,
    boolean_random_circuit=args.boolean_random_circuit,
    boolean_number_input_gates=args.boolean_number_input_gates,
    pendulum_time_delta=args.pendulum_time_delta,
    )
  dataset_dict = data.DATASETS[args.dataset](**kwargs)

  feature_labels = dataset_dict.get('feature_labels', 
    ['Feature {ind}' for ind in range(dataset_dict['number_features'])])

  print(f'Dataset {args.dataset} loaded.')
  if args.ib:
  	# Extremely simple change to train with IB instead of DIB: just treat everything as one feature
  	dataset_dict['feature_dimensionalities'] = [np.sum(dataset_dict['feature_dimensionalities'])]

  ## Build the model
  output_dimensionality = dataset_dict['output_dimensionality'] if not args.infonce_loss else args.infonce_space_dimensionality
  output_activation_fn = dataset_dict['output_activation_fn'] if not args.infonce_loss else None
  model = models.DistributedIBNet(dataset_dict['feature_dimensionalities'],
    args.feature_encoder_architecture,
    args.integration_network_architecture,
    output_dimensionality,
    use_positional_encoding=use_positional_encoding,
    number_positional_encoding_frequencies=number_positional_encoding_frequencies,
    activation_fn=activation_fn,
    feature_embedding_dimension=args.feature_embedding_dimension,
    output_activation_fn=output_activation_fn)

  optimizer = tf.keras.optimizers.get(args.optimizer)
  optimizer.learning_rate = args.learning_rate

  train_model_simple = not args.infonce_loss

  if train_model_simple:
    ################################ Keras API ######################################
    ## Most cases should be able to use the following code, which relies on keras Callbacks
    ## to update beta and monitor the information allocation (which is retrieved through
    ## the history object returned by model.fit()).
    model.compile(
    	optimizer=optimizer,
    	loss=dataset_dict['loss'],
    	metrics=dataset_dict['metrics'],
    	)

    beta_annealing_callback = models.InfoBottleneckAnnealingCallback(beta_start,
                 beta_end,
                 number_pretraining_epochs,
                 number_annealing_epochs)
    callbacks = [beta_annealing_callback]

    if save_compression_matrices_frequency > 0:
    	callbacks.append(models.SaveCompressionMatricesCallback(
    		save_compression_matrices_frequency,
    		dataset_dict['x_valid'],
    		dataset_dict.get('x_valid_raw', dataset_dict['x_valid']),
    		args.artifact_outdir))

    history = model.fit(
    	dataset_dict['x_train'],
    	dataset_dict['y_train'],
    	epochs=number_epochs,
    	shuffle=True,
    	batch_size=batch_size,
    	callbacks=callbacks,
    	verbose=False,
    	validation_data=(dataset_dict['x_valid'], dataset_dict['y_valid']),
    	)

    ## Grab the stats from the run
    beta_series = np.float32(history.history['beta'])
    kl_series = np.stack([history.history[f'KL{feature_ind}'] for feature_ind in range(dataset_dict['number_features'])], -1)
    loss_series = np.float32(history.history['loss'])
    loss_series_validation = np.float32(history.history['val_loss'])
    ## Get the original loss without the KL term
    loss_series -= beta_series * np.sum(kl_series, axis=-1)
    kl_series /= np.log(2)  ## convert the KL values to bits
    
    if dataset_dict['loss_is_info_based']:
      loss_series /= np.log(2)  ## convert the loss values to bits

  else: 
    ################################ Custom training loop ######################################
    ## If training requires additional models, or if you just want finer-grained control
    ## of everything, the following custom training loop is more versatile
    
    ## Here we assume the loss is infonce and create an encoder for the output variable
    output_encoder_layers = [tf.keras.layers.Input((dataset_dict['y_train'].shape[-1],))]
    if use_positional_encoding:
      positional_encoding_frequencies = 2**np.arange(1, number_positional_encoding_frequencies) 
      output_encoder_layers += [models.PositionalEncoding(positional_encoding_frequencies)]
    for num_units in args.infonce_y_encoder_architecture:
      output_encoder_layers += [tf.keras.layers.Dense(num_units, activation_fn)]
    output_encoder_layers += [tf.keras.layers.Dense(args.infonce_space_dimensionality)]
    output_encoder = tf.keras.Sequential(output_encoder_layers)
                    
    # Pass an input through the model to build it
    model(np.ones((1, dataset_dict['x_train'].shape[-1])))

    all_trainable_variables = model.trainable_variables + output_encoder.trainable_variables

    # Define the custom eval step, to be used for training and eval
    @tf.function
    def eval_batch_infonce(inps, outps, training=True):
      with tf.GradientTape() as tape:
        inps_encoding = model(inps, training=training)
        outps_encoding = output_encoder(outps, training=training)
        
        similarity_matrix = utils.get_scaled_similarity(inps_encoding, 
          outps_encoding, args.infonce_similarity, args.infonce_temperature)
        loss_infonce = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
            tf.range(batch_size), similarity_matrix, from_logits=True))
        loss_infonce += tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
            tf.range(batch_size), tf.transpose(similarity_matrix), from_logits=True))
        kl_loss = model.losses  ## this already has beta factored in
        loss = loss_infonce + tf.reduce_sum(kl_loss)
      if training:
        grads = tape.gradient(loss, all_trainable_variables)
        optimizer.apply_gradients(zip(grads, all_trainable_variables))
      return loss_infonce, kl_loss / model.beta

    # We make a tf.data.Dataset because we always want a full batch for consistency of the infonce loss
    # and we can use .repeat() to nicely roll over from epoch to epoch with full batches 
    dataset_length = dataset_dict['x_train'].shape[0]
    steps_per_epoch = dataset_length / batch_size
    tf_dataset = tf.data.Dataset.from_tensor_slices((dataset_dict['x_train'], dataset_dict['y_train']))
    tf_dataset = tf_dataset.repeat().shuffle(min(dataset_length, 10_000)).batch(batch_size)

    # For validation, we don't want partial batches, so just shuffle everything and round up to the next full batch
    validation_set_length = dataset_dict['x_valid'].shape[0]
    number_full_validation_batches = validation_set_length // batch_size

    tf_dataset_validation = tf.data.Dataset.from_tensor_slices((dataset_dict['x_valid'], dataset_dict['y_valid']))
    tf_dataset_validation = tf_dataset_validation.repeat().shuffle(min(validation_set_length, 10_000)).batch(batch_size).take(number_full_validation_batches+1)

    epoch_steps = np.round(steps_per_epoch*np.arange(number_epochs)).astype(np.int32)
    loss_infonce_series, loss_infonce_validation_series, loss_kl_series, loss_kl_validation_series = [[] for _ in range(4)] 

    loss_infonce_running, loss_infonce_validation_running, loss_kl_running, loss_kl_validation_running = [[] for _ in range(4)]
    beta_series = []
    for step_num, (batch_inps, batch_outps) in enumerate(tf_dataset.take(epoch_steps[-1])):
      loss_infonce, loss_kl = eval_batch_infonce(batch_inps, batch_outps, training=True)
      loss_infonce_running.append(loss_infonce)
      loss_kl_running.append(loss_kl)
      if step_num in epoch_steps:
        ## Update beta
        epoch_num = np.where(epoch_steps==step_num)[0][0]
        next_beta = np.exp(np.log(beta_start)+float(max(epoch_num-number_pretraining_epochs, 0))/number_annealing_epochs*(np.log(beta_end)-np.log(beta_start)))
        beta_series.append(next_beta)
        model.beta.assign(next_beta)
        if save_compression_matrices_frequency and epoch_num % save_compression_matrices_frequency == 0:
          ## Save the compression mats
          beta_value = model.beta.value()
          features_split = tf.split(dataset_dict['x_valid'], model.feature_dimensionalities, axis=-1)
          features_split_raw = tf.split(dataset_dict.get('x_valid_raw', dataset_dict['x_valid']), 
            model.feature_dimensionalities, axis=-1)

          for feature_ind in range(model.number_features):
            out_fname = os.path.join(artifact_outdir, f'feature_{feature_ind}_log10beta_{np.log10(beta_value):.3f}.png')
            visualization.save_compression_matrices(model.feature_encoders[feature_ind], 
              features_split[feature_ind], out_fname, inp_features_raw=features_split_raw[feature_ind])
        # Evaluate the validation set
        for validation_batch_inps, validation_batch_outps in tf_dataset_validation:
          # Note we are still using noise in the probabilistic encoder -- the encoding *is* the distribution, and 
          # we are interested in the information content of that distribution, not the mean
          loss_infonce_val, loss_kl_val = eval_batch_infonce(validation_batch_inps, validation_batch_outps, training=False)
          loss_infonce_validation_running.append(loss_infonce_val)
          loss_kl_validation_running.append(loss_kl_val)
        
        # Update metrics
        loss_infonce_series.append(np.mean(loss_infonce_running))
        loss_infonce_validation_series.append(np.mean(loss_infonce_validation_running))
        loss_kl_series.append(np.mean(loss_kl_running, axis=0))
        loss_kl_validation_series.append(np.mean(loss_kl_validation_running, axis=0))
        loss_infonce_running, loss_infonce_validation_running, loss_kl_running, loss_kl_validation_running = [[] for _ in range(4)]

    ## Grab the stats from the run
    beta_series = np.float32(beta_series)
    kl_series = np.stack(loss_kl_series, 0)
    loss_series = np.float32(loss_infonce_series)
    kl_series_validation = np.stack(loss_kl_validation_series, 0)
    loss_series_validation = np.float32(loss_infonce_validation_series)

    kl_series /= np.log(2)  ## convert the KL values to bits
    kl_series_validation /= np.log(2)
    
    if dataset_dict['loss_is_info_based']:
      loss_series /= np.log(2)  ## convert the loss values to bits
      loss_series_validation /= np.log(2)  ## convert the loss values to bits

  print('Finished training.')

  ## Plot the training trajectory in the info plane
  visualization.save_distributed_info_plane(kl_series_validation, loss_series_validation, artifact_outdir)


if __name__ == '__main__':
    main()