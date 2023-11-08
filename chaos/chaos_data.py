import numpy as np

def generate_data(system_name, number_iterations=1_000_000, number_skip_iterations=100_000, **system_params):
  '''Generate a long trajectory for a chaotic system.

  Args:
    system_name: Must be one of ['logistic', 'henon', 'ikeda'].
    number_iterations: The length of the trajectory to generate.
    number_skip_iterations: The number of steps to run before saving, in order to bypass transients.
    system_params: Parameter values for the specific systems, defined in the Supp. of the manuscript.
      logistic: r  |  default value is 3.7115 (used in the paper)
      henon: a, b  |  default values are 1.4, 0.3.
      ikeda: a, b, kappa, eta  |  default values are 1, 0.9, 0.4, 6.
  Returns: Numpy array of shape [number_iterations, state_dimensionality]
  '''
  if system_name == 'logistic':
    def iterate_logistic(x, r):
      return x * (1. - x) * r
    x = np.random.rand()
    r = system_params.get('r', 3.7115)
    data_arr = [x]
    for _ in range(number_iterations+number_skip_iterations-1):
      x = iterate_logistic(x, r)
      data_arr.append(x)
    data_arr = np.stack(data_arr[number_skip_iterations:]).reshape([-1, 1])
  elif system_name == 'henon':
    def iterate_henon(x, y, a, b):
      return 1 - a*x**2 + b*y, x
    x, y = np.random.rand(2)
    a = system_params.get('a', 1.4)
    b = system_params.get('b', 0.3)
    data_arr = [[x, y]]
    for _ in range(number_iterations+number_skip_iterations-1):
      x, y = iterate_henon(x, y, a, b)
      data_arr.append([x, y])
    data_arr = np.stack(data_arr[number_skip_iterations:], 0)
  elif system_name == 'ikeda':  # using notation from Davidchack et al. 2000
    def iterate_ikeda(x, y, a, b, kappa, eta):
      phi = kappa - eta / (1. + x**2 + y**2)
      x_new = a + b * (x * np.cos(phi) - y * np.sin(phi))
      y_new = b * (x * np.sin(phi) + y * np.cos(phi))
      return x_new, y_new
    a = system_params.get('a', 1.)
    b = system_params.get('b', 0.9)
    kappa = system_params.get('kappa', 0.4)
    eta = system_params.get('eta', 6)
    x, y = np.random.rand(2)
    data_arr = [[x, y]]
    for _ in range(number_iterations+number_skip_iterations-1):
      x, y = iterate_ikeda(x, y, a, b, kappa, eta)
      data_arr.append([x, y])
    data_arr = np.stack(data_arr[number_skip_iterations:], 0)
  else:
    raise ValueError(f'System {system_name} not implemented.')
  return data_arr
