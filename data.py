"""
Data loading and preprocessing for:
- Boolean circuit
- Glassy rearrangement
- Mona Lisa (and other images)
- Double pendulum
- Tabular datasets

Coming soon!
"""

def load(dataset_name):


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