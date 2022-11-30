"""
Simulates a double pendulum.

Modified from https://scipython.com/blog/the-double-pendulum/ 
"""
from scipy.integrate import odeint
import numpy as np
import os

def simulate_double_pendulum(data_path='./data/', simulation_params_dict=None):
  # The gravitational acceleration (m.s-2).  Everything is SI
  g = 9.81
  if simulation_params_dict is None:
    simulation_params_dict = {}
  m1 = simulation_params_dict.get('m1', 1)
  m2 = simulation_params_dict.get('m2', 1)
  L1 = simulation_params_dict.get('L1', 1)
  L2 = simulation_params_dict.get('L2', 1)

  energy_over_g = simulation_params_dict.get('energy_over_g', 4)

  # Maximum time, time point spacings
  initial_time = simulation_params_dict.get('initial_time', 50)  ## to run through initial condition transients
  simulation_time = simulation_params_dict.get('simulation_time', 50)
  dt_simulation = simulation_params_dict.get('dt_simulation', 1e-2)
  dt_saving = simulation_params_dict.get('dt_saving', 2e-2)

  number_trajectories = simulation_params_dict.get('number_trajectories', 1000)

  save_every = int(dt_saving // dt_simulation)

  def deriv(y, t, L1, L2, m1, m2):
      """Return the first derivatives of y = theta1, z1, theta2, z2."""
      theta1, z1, theta2, z2 = y

      c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

      theta1dot = z1
      z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
               (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
      theta2dot = z2
      z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
               m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
      return theta1dot, z1dot, theta2dot, z2dot

  def calc_E(y):
      """Return the total energy of the system."""
      th1, th1d, th2, th2d = y.T
      V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
      T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
              2*L1*L2*th1d*th2d*np.cos(th1-th2))
      return T + V
          
  data_arr = []
  successful_runs = unsuccessful_runs = 0
  while successful_runs < number_trajectories:
    # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
    # Sample theta1 uniformly, then solve for the theta2 that would give the prescribed potential energy with zero velocity
    theta1 = np.random.uniform()*2*np.pi
    height1 = L1 * (1. - np.cos(theta1))
    energy_over_g_contribution = m1*height1

    theta2 = np.arccos(1 - ((energy_over_g - energy_over_g_contribution) / m2 - height1) / L2) * (np.random.randint(2)*2-1)
    if np.isnan(theta2):
      ## Try again
      continue    

    y0 = np.array([
        theta1,
        0,  ## dtheta1/dt
        theta2,
        0,  ## dtheta2/dt        
    ])
          
    t = np.linspace(0, initial_time+simulation_time, int((initial_time+simulation_time)//dt_simulation))

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

    # Check that the calculation conserves total energy to within some tolerance.
    FRACTIONAL_EDRIFT_TOL = 1e-3
    # Total energy from the initial conditions
    E = calc_E(y0)
    if np.max(np.abs(calc_E(y) - E)/E) > FRACTIONAL_EDRIFT_TOL:
      unsuccessful_runs += 1
      continue

    run_arr = y[int(initial_time//dt_simulation)::save_every]  ## Sieve out to save a smaller file
    data_arr.append(run_arr)
    successful_runs += 1
    if successful_runs % 1000 == 0:
      print(f'{successful_runs}/{number_trajectories} completed, {unsuccessful_runs} unsuccessful')

  data_arr = np.stack(data_arr, 0)
  print(f'Saving simulation data, shape {data_arr.shape}')
  np.save(os.path.join(data_path, 'double_pendulum.npy'), data_arr)


if __name__ == '__main__':
  simulate_double_pendulum()