import numpy as np

import matplotlib.pyplot as plt
import os
import scipy.ndimage as nim
import tensorflow as tf

import utils

data_dir = './double_pendulum_simulation_data'
outdir = './double_pendulum_training_artifacts'
if not os.path.exists(outdir):
	os.makedirs(outdir)

energy_over_g = 4
## An IB will be run for each of the following time horizons
time_deltas_to_train_on = [0.5, 1, 2, 3, 4, 6]

batch_size = 256
learning_rate = 3e-4
beta_start = 5e-4
beta_end = 2e0
number_annealing_steps = 5*10**4
input_state_encoder_arch_spec = [128]*2
shared_encoder_present_arch_spec = [256]*2
shared_encoder_future_arch_spec = [256]*2
activation_fn = 'leaky_relu'
number_positional_encoding_frequencies = 8  ## how much to 'shuffle' the input space, for help w training when input features are low dimensional (see Tancik 2020)
positional_encoding_frequencies = 2**np.arange(number_positional_encoding_frequencies)

bottleneck_space_dimensionality = 32
infonce_similarity = 'l2'
infonce_temperature = 1.
infonce_space_dimensionality = 64

m1, m2 = 1, 1
L1, L2 = 1, 1

dt = 0.02
start_time = 50  ## to get past the initial state dependence; the runs are 100s long so take the second half
start_timestep_ind = int(start_time // dt) 
number_simulation_chunks = 2

eval_every = number_annealing_steps // 500
cut_down_eval_size_factor = 5  ## instead of evaluation on the whole validation set, which can be quite slow

## Turn the angles into sin cos to sidestep issues with periodicity
## This means we'll have to do a little more work to properly hand each feature encoder the right variables
num_components = 4
num_components_unrolled = 6
component_index_table = [[0, 1], [2], [3, 4], [5]]  ## for gathering the right indices to input to each feature encoder

def preprocess_angle_data(arr):
	return np.stack([np.sin(arr[:, :, 0]),
                 -np.cos(arr[:, :, 0]),
                 arr[:, :, 1],
                 np.sin(arr[:, :, 2]),
                 -np.cos(arr[:, :, 2]),
                 arr[:, :, 3],
                 ], -1)

train_val_split = 0  # Use 20% of the data for validation
data_arr, data_arr_validation = [[], []]
for simulation_chunk in range(number_simulation_chunks):
  
  data_fname = os.path.join(data_dir, f'E_{energy_over_g}_l1_{L1:.02f},l2_{L2:.02f},m1_{m1:.02f},m2_{m2:.02f}_{simulation_chunk}.npy')
  if simulation_chunk == train_val_split:
    data_arr_validation = np.load(data_fname)[:, start_timestep_ind:]
  else:
    data_arr.append(np.load(data_fname)[:, start_timestep_ind:])

data_arr = np.concatenate(data_arr, 0)
print(f'Train data loaded, shape = {data_arr.shape}')
print(f'Validation data loaded, shape = {data_arr_validation.shape}')
## Compute the entropy so that we can convert cross entropy to a mutual information estimate
entropy = np.log2(batch_size)
data_arr = preprocess_angle_data(data_arr)
data_arr_validation = preprocess_angle_data(data_arr_validation)

for time_delta in time_deltas_to_train_on:
	out_filename = os.path.join(outdir, f'IB_E={energy_over_g}_L1={L1}_L2={L2}_timedelay={time_delta:.2f}s_split={train_val_split}.npz')

	if os.path.exists(out_filename):
		print('Results already exist, continuing on.')
		continue

	time_delta_timesteps = int(time_delta/dt)
	print(f'Starting IB run for double pendulum with time delta = {time_delta}s aka {time_delta_timesteps} frames')

	## An encoder for the input state -> bottleneck space
	combined_component_encoder = tf.keras.Sequential([tf.keras.layers.Input((num_components_unrolled)), utils.PositionalEncoding(positional_encoding_frequencies)] + 
                            [tf.keras.layers.Dense(num_units, activation_fn) for num_units in input_state_encoder_arch_spec] + 
                            [tf.keras.layers.Dense(2*bottleneck_space_dimensionality)])
	all_trainable_vars = combined_component_encoder.trainable_variables

	## Another encoder to take representation -> InfoNCE space
	enc1 = tf.keras.Sequential([tf.keras.layers.Input(bottleneck_space_dimensionality)] + 
	                                                [tf.keras.layers.Dense(num_units, activation_fn) for num_units in shared_encoder_present_arch_spec] + 
	                                                [tf.keras.layers.Dense(infonce_space_dimensionality)])
	all_trainable_vars += enc1.trainable_variables 

	## An encoder for the future state -> InfoNCE space
	enc2 = tf.keras.Sequential([tf.keras.layers.Input((num_components_unrolled)), utils.PositionalEncoding(positional_encoding_frequencies)] + 
	                          [tf.keras.layers.Dense(num_units, activation_fn) for num_units in shared_encoder_future_arch_spec] + 
	                          [tf.keras.layers.Dense(infonce_space_dimensionality)])
	all_trainable_vars += enc2.trainable_variables

	optimizer = tf.keras.optimizers.Adam(learning_rate)
	beta_var = tf.Variable(beta_start, dtype=tf.float32, trainable=False)

	@tf.function
	def train_step_infonce_IB(state1_batch, state2_batch):
		with tf.GradientTape() as tape:
			embs_mus, embs_logvars = tf.split(combined_component_encoder(state1_batch), 2, axis=-1)
			reparameterized_embs = tf.random.normal(embs_mus.shape, mean=embs_mus, stddev=tf.exp(embs_logvars/2.))
			kl = tf.reduce_sum(0.5 * (tf.square(embs_mus) + tf.exp(embs_logvars) - embs_logvars - 1.), axis=-1)  ## [batch_size]

			full_emb_state1 = enc1(reparameterized_embs)
			full_emb_state2 = enc2(state2_batch)

			sim_mat = utils.get_scaled_similarity(full_emb_state1, full_emb_state2, infonce_similarity, infonce_temperature)
			loss_infonce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.range(batch_size), 
			                                                                              sim_mat,
			                                                                              from_logits=True))
			loss_infonce += tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.range(batch_size), 
			                                                                              tf.transpose(sim_mat),
			                                                                              from_logits=True))
			kl_batch_avged = tf.reduce_mean(kl, axis=0)
			loss = loss_infonce / 2. 
			loss += beta_var * tf.reduce_sum(kl_batch_avged)
		grads = tape.gradient(loss, all_trainable_vars)
		optimizer.apply_gradients(zip(grads, all_trainable_vars))
		return loss_infonce / 2., kl_batch_avged

	@tf.function
	def eval_step_infonce_IB(state1_batch, state2_batch):
		embs_mus, embs_logvars = tf.split(combined_component_encoder(state1_batch), 2, axis=-1)
		reparameterized_embs = tf.random.normal(embs_mus.shape, mean=embs_mus, stddev=tf.exp(embs_logvars/2.))
		kl = tf.reduce_sum(0.5 * (tf.square(embs_mus) + tf.exp(embs_logvars) - embs_logvars - 1.), axis=-1)  ## [batch_size]

		full_emb_state1 = enc1(reparameterized_embs)
		full_emb_state2 = enc2(state2_batch)

		sim_mat = utils.get_scaled_similarity(full_emb_state1, full_emb_state2, infonce_similarity, infonce_temperature)
		loss_infonce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.range(batch_size), 
		                                                                              sim_mat,
		                                                                              from_logits=True))
		loss_infonce += tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.range(batch_size), 
		                                                                              tf.transpose(sim_mat),
		                                                                              from_logits=True))
		kl_batch_avged = tf.reduce_mean(kl, axis=0)
		loss = loss_infonce / 2. 
		loss += beta_var * tf.reduce_sum(kl_batch_avged)
		return loss_infonce / 2., kl_batch_avged


	InfoNCE_series, InfoNCE_series_validation, KL_divergence_series, KL_divergence_series_validation = [[] for _ in range(4)]
	for step_num in range(number_annealing_steps):
		## ramp up beta logarithmically
		beta_var.assign(np.exp(np.log(beta_start)+float(step_num)/number_annealing_steps*(np.log(beta_end)-np.log(beta_start))))

		## grab some random run inds and time inds
		batch_inds = np.random.choice(data_arr.shape[0], size=batch_size)
		time_ind = np.random.randint(data_arr.shape[1]-time_delta_timesteps)

		state1_batch = data_arr[batch_inds][:, time_ind]  ## [batch_size, num_components]
		state2_batch = data_arr[batch_inds][:, time_ind+time_delta_timesteps]  ## [batch_size, num_components]

		loss_infonce, kl_batch_avged = train_step_infonce_IB(state1_batch, state2_batch)
		InfoNCE_series.append(loss_infonce.numpy())
		KL_divergence_series.append(kl_batch_avged.numpy())

		if (step_num+1) % eval_every == 0:
			print(f'Beginning evaluation, step {step_num+1}.')
			loss_infonce_evals, kl_evals = [[], []]
			for time_ind in range(0, data_arr_validation.shape[0]-time_delta_timesteps, data_arr_validation.shape[0]//cut_down_eval_size_factor):

				for batch_start in range(0, data_arr_validation.shape[0]-batch_size, batch_size):
          
					batch_inds = np.arange(batch_start, batch_start+batch_size)

					state1_batch = data_arr_validation[batch_inds][:, time_ind]  ## [batch_size, num_components]
					state2_batch = data_arr_validation[batch_inds][:, time_ind+time_delta_timesteps]  ## [batch_size, num_components]

					loss_infonce, kl_batch_avged = eval_step_infonce_IB(state1_batch, state2_batch)
					loss_infonce_evals.append(loss_infonce.numpy())
					kl_evals.append(kl_batch_avged.numpy())
				InfoNCE_series_validation.append(np.mean(loss_infonce_evals))
				KL_divergence_series_validation.append(np.mean(kl_evals, axis=0))

	## display the kls
	KL_divergence_series = np.stack(KL_divergence_series, 0)  ## [num_steps, 1]
	KL_divergence_series_validation = np.stack(KL_divergence_series_validation, 0)  ## [num_steps, 1]
	betas = np.exp(np.log(beta_start)+np.linspace(0, 1, number_annealing_steps)*(np.log(beta_end)-np.log(beta_start)))

	## Average a little bit over time to smooth things out; sieve the data points so that matplotlib has a lighter load
	smoothing_sigma = 20
	sieve_factor = number_annealing_steps // 250
	approx_info_inIB = nim.filters.gaussian_filter1d(KL_divergence_series, smoothing_sigma, axis=0)[::sieve_factor]/np.log(2)
	approx_info_outIB = entropy-nim.filters.gaussian_filter(InfoNCE_series, smoothing_sigma)[::sieve_factor]/np.log(2)

	info_out_display_max = 6
	info_in_display_max = 10

	plt.figure(figsize=(8, 6))
	plt.plot(np.squeeze(approx_info_inIB), approx_info_outIB, lw=2, color='k')
	plt.xlim(0, info_in_display_max)
	plt.ylim(0, info_out_display_max)
	plt.xlabel('Info in (Approx bits)', fontsize=15)
	plt.ylabel('Info out (Approx bits)', fontsize=15)

	plt.savefig(os.path.join(outdir, f'IB_E={energy_over_g}_L1={L1}_L2={L2}_timedelay={time_delta:.2f}s_infoplane.png'))

	## save the data
	pickle_dict = dict(betas=betas[::eval_every],
	                   kl_divergences_log2=KL_divergence_series_validation/np.log(2),
	                   entropy_minus_infonce_log2=entropy-np.float32(InfoNCE_series_validation)/np.log(2))
	for key in pickle_dict.keys():
		print(f'Saving {key} with shape {pickle_dict[key].shape}.')
	np.savez(out_filename, **pickle_dict)
	print(f'Saved results to {out_filename}.')