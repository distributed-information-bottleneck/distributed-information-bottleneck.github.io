"""
Data loading and preprocessing for:
- Boolean circuit
- Double pendulum
- Tabular datasets (copied heavily from NODE-GAM, https://github.com/zzzace2000/nodegam)

"""
import numpy as np  
import tensorflow as tf  
import os

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from category_encoders import OneHotEncoder
import time

import nodegam

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

  x_valid = x_train
  y_valid = y_train

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
  	feature_dimensionalities=feature_dimensionalities,
  	number_features=len(feature_dimensionalities),
  	output_dimensionality=output_dimensionality,
  	output_activation_fn=output_activation_fn,
  	loss=loss,
  	loss_is_info_based=loss_is_info_based,
  	metrics=metrics,
  	)

def fetch_double_pendulum(**kwargs):

  data_path = kwargs.get('data_path', './data/')
  data_fname = os.path.join(data_path, 'double_pendulum.npy')
  if not os.path.exists(data_fname):
    print('Generating double pendulum data with default parameters.')
    import simulate_pendulum 
    simulate_pendulum.simulate_double_pendulum(data_path=data_path)

  data_arr = np.load(data_fname)

  time_delta = kwargs.get('pendulum_time_delta', 2.)

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

  validation_fraction = 0.1

  data_arr, data_arr_validation = np.split(data_arr, [int(data_arr.shape[0]*validation_fraction)], axis=0)

  print(f'Train data loaded, shape = {data_arr.shape}')
  print(f'Validation data loaded, shape = {data_arr_validation.shape}')

  data_arr = preprocess_angle_data(data_arr)
  data_arr_validation = preprocess_angle_data(data_arr_validation)

  dt_simulation = 0.02  ## hardcoded for now
  time_delta_timesteps = int(time_delta/dt_simulation)

  x_train = data_arr[:, :-time_delta_timesteps].reshape([-1, num_components_unrolled])
  y_train = data_arr[:, time_delta_timesteps:].reshape([-1, num_components_unrolled])
  x_valid = data_arr_validation[:, :-time_delta_timesteps].reshape([-1, num_components_unrolled])
  y_valid = data_arr_validation[:, time_delta_timesteps:].reshape([-1, num_components_unrolled])

  feature_dimensionalities = [2, 1, 2, 1]  ## because the angle of the first and second arm have been converted to unit vectors

  output_dimensionality = 6  ## only used if not infonce, and you want to use MSE straight on the output state vector
  output_activation_fn = None 
  loss = 'infonce'
  loss_is_info_based = True 
  feature_labels = ['theta1', 'theta1_dot', 'theta2', 'theta2_dot']

  return dict(
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    feature_dimensionalities=feature_dimensionalities,
    number_features=len(feature_dimensionalities),
    output_dimensionality=output_dimensionality,
    output_activation_fn=output_activation_fn,
    loss=loss,
    loss_is_info_based=loss_is_info_based,
    feature_labels=feature_labels,
    )

######################## Tabular datasets for "Interpretability with full complexity by constraining feature information"
# Mostly just the data loading framework from NODE-GAM, with minor modifications

def download(url, filename, delete_if_interrupted=True, chunk_size=4096):
  """It saves file from url to filename with a fancy progressbar."""
  try:
      with open(filename, "wb") as f:
          print("Downloading {} > {}".format(url, filename))
          response = requests.get(url, stream=True)
          total_length = response.headers.get('content-length')

          if total_length is None:  # no content length header
              f.write(response.content)
          else:
              total_length = int(total_length)
              with tqdm(total=total_length) as progressbar:
                  for data in response.iter_content(chunk_size=chunk_size):
                      if data:  # filter-out keep-alive chunks
                          f.write(data)
                          progressbar.update(len(data))
  except Exception as e:
      if delete_if_interrupted:
          print("Removing incomplete download {}.".format(filename))
          os.remove(filename)
      raise e
  return filename


# Modified NODEGAM preprocessor to use one hot instead of leave one out encoding
class MyPreprocessor:
  def __init__(self, random_state=1377, cat_features=None,
             y_normalize=False, quantile_transform=False,
             output_distribution='normal', n_quantiles=2000,
             quantile_noise=1e-3, one_hot=True):
    """Preprocessor does the data preprocessing like input and target normalization.
    Args:
        random_state: Global random seed for an experiment.
        cat_features: If passed in, it does the ordinal encoding for these features before other
            input normalization like quantile transformation. Default: None.
        y_normalize: If True, it standardizes the targets y by setting the mean and stdev to 0
            and 1. Useful in the regression setting.
        quantile_transform: If True, transforms the features to follow a normal or uniform
            distribution.
        output_distribution: Choose between ['normal', 'uniform']. Data is projected onto this
            distribution. See the same param of sklearn QuantileTransformer. 'normal' is better.
        n_quantiles: Number of quantiles to estimate the distribution. Default: 2000.
        quantile_noise: If specified, fits QuantileTransformer on data with added gaussian noise
            with std = :quantile_noise: * data.std; this will cause discrete values to be more
            separable. Please note that this transformation does NOT apply gaussian noise to the
            resulting data, the noise is only applied for QuantileTransformer.
    Example:
        >>> preprocessor = nodegam.mypreprocessor.MyPreprocessor(
        >>>     cat_features=['ethnicity', 'gender'],
        >>>     y_normalize=True,
        >>>     random_state=1337,
        >>> )
        >>> preprocessor.fit(X_train, y_train)
        >>> X_train, y_train = preprocessor.transform(X_train, y_train)
    """

    self.random_state = random_state
    self.cat_features = cat_features
    self.y_normalize = y_normalize
    self.quantile_transform = quantile_transform
    self.output_distribution = output_distribution
    self.quantile_noise = quantile_noise
    self.n_quantiles = n_quantiles
    self.one_hot = one_hot
    self.transformers = []
    self.y_mu, self.y_std = 0, 1
    self.feature_names = None

  def fit(self, X, y):
    """Fit the transformer.
    Args:
        X (pandas dataframe): Input data.
        y (numpy array): target y.
    """
    assert isinstance(X, pd.DataFrame), 'X is not a dataframe! %s' % type(X)
    self.feature_names = X.columns

    if self.cat_features is not None:
      if self.one_hot:
        cat_encoder = OneHotEncoder(cols=self.cat_features)
      else:
        cat_encoder = LeaveOneOutEncoder(cols=self.cat_features)
      cat_encoder.fit(X, y)
      self.transformers.append(cat_encoder)

    if self.quantile_transform:
      quantile_train = X.copy()
      if self.cat_features is not None:
        quantile_train = cat_encoder.transform(quantile_train)

      if self.quantile_noise:
        r = np.random.RandomState(self.random_state)
        stds = np.std(quantile_train.values, axis=0, keepdims=True)
        noise_std = self.quantile_noise / np.maximum(stds, self.quantile_noise)
        quantile_train += noise_std * r.randn(*quantile_train.shape)

      qt = QuantileTransformer(random_state=self.random_state,
                               n_quantiles=self.n_quantiles,
                               output_distribution=self.output_distribution,
                               copy=False)
      qt.fit(quantile_train)
      self.transformers.append(qt)

    if y is not None and self.y_normalize:
      self.y_mu, self.y_std = y.mean(axis=0), y.std(axis=0)
      print("Normalize y. mean = {}, std = {}".format(self.y_mu, self.y_std))

  def transform(self, *args):
    """Transform the data.
    Args:
        X (pandas daraframe): Input data.
        y (numpy array): Optional. If passed in, it will do target normalization.
    Returns:
        X (pandas daraframe): Normalized Input data.
        y (numpy array): Optional. Normalized y.
    """
    assert len(args) <= 2

    X = args[0]
    if len(self.transformers) > 0:
      X = X.copy()
      if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=self.feature_names)

      for i, t in enumerate(self.transformers):
        # Leave one out transform when it's training set
        X = t.transform(X)

    # Make everything as numpy and float32
    if isinstance(X, pd.DataFrame):
      X = X.values
    X = X.astype(np.float32)

    if len(args) == 1:
      return X

    y = args[1]
    if y is None:
      return X, None

    if self.y_normalize and self.y_mu is not None and self.y_std is not None:
      y = (y - self.y_mu) / self.y_std
      y = y.astype(np.float32)

    return X, y

def fetch_mice_protein(data_path='./data/', **kwargs):
  ## Copied from LassoNet (https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py)
  data_path = os.path.join(data_path, 'mice_protein', 'Data_Cortex_Nuclear.xls')

  if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
    download("https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls", data_path)

  filling_value = -100000
  X = np.genfromtxt(data_path, delimiter = ',', skip_header = 1, usecols = range(1, 78), filling_values = filling_value, encoding = 'UTF-8')
  classes = np.genfromtxt(data_path, delimiter = ',', skip_header = 1, usecols = range(78, 81), dtype = None, encoding = 'UTF-8')

  for i, row in enumerate(X):
    for j, val in enumerate(row):
      if val == filling_value:
        X[i, j] = np.mean([X[k, j] for k in range(classes.shape[0]) if np.all(classes[i] == classes[k])])

  DY = np.zeros((classes.shape[0]), dtype = np.uint8)
  for i, row in enumerate(classes):
    for j, (val, label) in enumerate(zip(row, ['Control', 'Memantine', 'C/S'])):
      DY[i] += (2 ** j) * (val == label)

  Y = np.zeros((DY.shape[0], np.unique(DY).shape[0]))
  for idx, val in enumerate(DY):
    Y[idx, val] = 1

  indices = np.arange(X.shape[0])
  np.random.shuffle(indices)
  X = X[indices]
  Y = Y[indices]
  DY = DY[indices]
  classes = classes[indices]

  Y = DY
      
  X = X.astype(np.float32)
  Y = Y.astype(np.int32)

  x_train=pd.DataFrame(X[: X.shape[0] * 4 // 5], columns =np.arange(X.shape[1])),
  y_train=Y[: X.shape[0] * 4 // 5],
  x_valid=pd.DataFrame(X[X.shape[0] * 4 // 5:], columns =np.arange(X.shape[1])),
  y_valid=Y[X.shape[0] * 4 // 5: ],

  problem = 'classification'
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  loss_is_info_based = True
  quantile_dist = 'normal'
  preprocessor = MyPreprocessor(
    cat_features=None,
    y_normalize=(problem == 'regression'),
    random_state=1337, quantile_transform=True,
    output_distribution=quantile_dist,
    quantile_noise=qn,
    one_hot=one_hot
  )
  preprocessor.fit(x_train, y_train)


  return dict(
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    feature_dimensionalities=feature_dimensionalities,
    number_features=len(feature_dimensionalities),
    output_dimensionality=output_dimensionality,
    output_activation_fn=output_activation_fn,
    loss=loss,
    loss_is_info_based=loss_is_info_based,
    metrics=metrics,
  )


def fetch_microsoft(data_path='./data/', **kwargs):
  dataset_dict = nodegam.data.DATASETS['MICROSOFT'](data_path)


  return 

def fetch_credit(data_path='./data/', **kwargs):
  dataset_dict = nodegam.data.DATASETS['CREDIT'](data_path)
  return 


def fetch_support2(data_path='./data/', **kwargs):
  dataset_dict = nodegam.data.DATASETS['SUPPORT2'](data_path)
  return 


def fetch_wine(data_path='./data/'):
  dataset_dict = nodegam.data.DATASETS['WINE'](data_path)
  return 


def fetch_BIKESHARE(data_path='./data/'):
  dataset_dict = nodegam.data.DATASETS['BIKESHARE'](data_path)
  return 

DATASETS = {
    'boolean_circuit': fetch_boolean_circuit,
    'double_pendulum': fetch_double_pendulum,
    'mice_protein': fetch_mice_protein,
    'microsoft': fetch_microsoft,
    'credit': fetch_credit,
    'support2': fetch_support2,
    'wine': fetch_wine,
    'bikeshare': fetch_bikeshare,
}