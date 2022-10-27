# Code to use the Distributed Information Bottleneck to localize information in data

The Distributed IB is just a probabilistic encoder for each feature, with a KL divergence penalty that we increase gradually over the course of training.
It should look very familiar to anyone who has coded a VAE.

For convenience, we have wrapped all the functionality into a `tf.keras.Model` subclass called `DistributedIBNet`. 
`DistributedIBNet` can be used in the standard `tf.keras.model` ways, e.g. with `model.compile(...)` and `model.fit(...)`. 
The bottleneck contribution to the loss is accounted for as a `model.loss`.
The rest of the functionality is achieved through custom `keras` callbacks:
- `InfoBottleneckAnnealingCallback` handles the logarithmic annealing of the bottleneck strength \beta
- `MutualInfoEstimateCallback` estimates the upper and lower bounds of the mutual information of features (one callback per feature!) during training
- `SaveDistinguishabilityMatricesCallback` saves the feature value distinguishability matrices during training

A full training run can be accomplished with the following:
```python
model = DistributedIBNet(model_params)
model.compile(
	optimizer=tf.keras.optimizers.Adam(lr),
	loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	metrics=['accuracy']
	)

beta_annealing_callback = InfoBottleneckAnnealingCallback(annealing_params)

history = model.fit(x, y, epochs=number_pretraining_epochs+number_annealing_epochs,
	callbacks=[beta_annealing_callback], verbose=False)
```

The data `x` is split by feature inside `DistributedIBNet`, which must be initialized with a list containing the dimensions of each feature (in the order that they appear in the data).

After training, the `history` object contains all the necessary values to plot the trajectory in the distributed information plane and to monitor the information allocation to the features.