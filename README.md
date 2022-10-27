# Want to find where the information is in your data?

## High level
The [Distributed Information Bottleneck (IB)](https://distributed-information-bottleneck.github.io) treats each feature of your data -- whether that feature is the age of a person visiting the hospital, the humidity on a day in Washington, D.C., or the density of a particular radial shell in a glassy material -- as information to be communicated before being used in a predictive model.  We place a cost on the information and that has the effect of locating the most informative features, and even the most informative distinctions within each feature, for different levels of predictive accuracy.

In typical machine learning scenarios, the primary products of a training run is peak performance on some metric and a model checkpoint. With the Distributed IB, the fruits of training are signals that map out the information in the data; the training history and artifacts along the way are the desiderata.

## Code overview
In practice, the (variational) Distributed IB is a probabilistic encoder for each feature and a KL divergence penalty that we increase gradually over the course of training.
Under the hood it looks very similar to a VAE.

For convenience, we have wrapped all the functionality into a `tf.keras.Model` subclass called `DistributedIBNet`. 
`DistributedIBNet` can be used in the standard `tf.keras.model` ways, e.g. with `model.compile(...)` and `model.fit(...)`. 
The bottleneck contribution to the loss is included through a call to `model.add_loss`.
The rest of the necessary functionality is achieved through custom `keras` callbacks:
- `InfoBottleneckAnnealingCallback` handles the logarithmic annealing of the bottleneck strength \beta
- `SaveDistinguishabilityMatricesCallback` saves the feature value distinguishability matrices during training

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