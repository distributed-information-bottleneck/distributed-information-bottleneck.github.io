# Want to find where the information is in your data?

## High level
The [Distributed Information Bottleneck (IB)](https://distributed-information-bottleneck.github.io) treats each feature of your data -- whether that feature is the age of a person visiting the hospital, the humidity on a day in Washington, D.C., or the density of a particular radial shell in a glassy material -- as information to be communicated before being used in a predictive model.  We place a cost on the information and that has the effect of locating the most informative features, and even the most informative distinctions within each feature, for different levels of predictive accuracy.

In typical machine learning scenarios, the primary products of a training run is peak performance on some metric and a model checkpoint. With the Distributed IB, the fruits of training are signals that map out the information in the data; the training history and artifacts along the way are the desiderata.

For more information and links to relevant papers, see the outward facing page of this repo, [distributed-information-bottleneck.github.io](https://distributed-information-bottleneck.github.io).

## Code overview
In practice, the (variational) Distributed IB is a probabilistic encoder for each feature and a KL divergence penalty that we increase gradually over the course of training.
Under the hood it looks very similar to a VAE.

For convenience, we have wrapped all the functionality into a `tf.keras.Model` subclass called `DistributedIBNet`. 
`DistributedIBNet` can be used in the standard `tf.keras.model` ways, e.g. with `model.compile(...)` and `model.fit(...)`. 
The bottleneck contribution to the loss is included through a call to `model.add_loss`.
The rest of the necessary functionality is achieved through custom `keras` callbacks:
- `InfoBottleneckAnnealingCallback` handles the logarithmic annealing of the bottleneck strength \beta
- `SaveCompressionMatricesCallback` saves the confusion matrices that represent the feature compression schemes learned during training

### Training example
A full training run can be accomplished with the following:
```python
model = DistributedIBNet(model_params)
model.compile(optimizer_and_loss_dict)
beta_annealing_callback = InfoBottleneckAnnealingCallback(annealing_params)

history = model.fit(x, y, 
	epochs=number_pretraining_epochs+number_annealing_epochs,
	callbacks=[beta_annealing_callback])
```

The data `x` is split by feature inside `DistributedIBNet`, which must be initialized with a list containing the dimensions of each feature (in the order that they appear in the data).

After training, the `history` object contains all the necessary values to plot the trajectory in the distributed information plane and to monitor the information allocation to the features.

### Model parameters

- `feature_dimensionalities`: List of ints specifying the dimension of each feature.
      **Example:** if the first feature is a scalar and the second is a one-hot with 4 values, 
      the feature_dimensionalities should be [1, 4].
- `feature_encoder_architecture`: List of ints specifying the number of units in each layer of the 
      feature encoders. **Example:** [64, 128] specifies that each feature encoder has 64 units in 
      the first layer and 128 in the second.
- `integration_network_architecture`: List of ints specifying the architecture of the MLP that
      integrates all of the feature embeddings into a prediction.
- `output_dimensionality`: Int specifying the dimensionality of the output. **Example:** If the task is 
      scalar regression, `output_dimensionality`=1; if it's classification, 
      `output_dimensionality`=number_classes; etc.
- `use_positional_encoding`: Boolean specifying whether to preprocess each feature with a 
      positional encoding layer.  Helps with training when using low-dimensional features.
      Default: True.
- `positional_encoding_frequencies`: List of floats specifying the frequencies to use in the
      positional encoding layer.
      Default: a few powers of 2, which we found to work well across the board.
- `activation_fn`: The activation function to use in the feature encoders and integration
      network, with the exception of the output layer.
      Default: `ReLU`.
- `feature_embedding_dimension`: Int specifying the embedding space dimensionality for each
      feature. 
      Default: 32.
- `output_activation_fn`: The activation function for the output. 
      Default: None.

### Running the (not distributed) Information Bottleneck
It is easy to run the vanilla IB, where the entire input X is bottlenecked at once: equivalent to viewing the data as one combined feature.
Instead of a multi-element list for `feature_dimensionalities`, simply pass a single element list that is the dimension of X.  
For example, if X has two 3-dimensional features, the Distributed IB can be run with `feature_dimensionalities = [3, 3]` and the IB with `feature_dimensionalities = [6]`.

### Data that isn't tabular
If some features of your data are time series, images, etc., where processing with an MLP doesn't make a lot of sense, you can use a subnetwork to process those features before inputting a distilled feature vector to `DistributedIBNet`.  It will most likely require defining a custom computation graph and using [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model), where you split the features in the computation graph, process whichever ones require it, and then re-concatenate the features along the last axis to feed them into an instance of `DistributedIBNet`.