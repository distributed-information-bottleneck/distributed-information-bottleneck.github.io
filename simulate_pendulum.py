"""
Simulates a double pendulum.

Modified from https://scipython.com/blog/the-double-pendulum/ 
"""
from scipy.integrate import odeint
import numpy as np
import os

data_dir = './double_pendulum_simulation_data'
if not os.path.exists(data_dir):
  os.makedirs(data_dir)

# The gravitational acceleration (m.s-2).  Everything is SI
g = 9.81
m1, m2 = 1, 1
energy_over_g = 4
L1, L2 = 1, 1

# Maximum time, time point spacings
tmax = 100
dt_simulation = 1e-2
dt_saving = 2e-2
save_every = int(dt_saving // dt_simulation)

number_trajectories = 1000  ## might take a little while
data_chunk_size = 500

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

for simulation_chunk in range(number_trajectories//data_chunk_size):
        
  giant_arr = []
  successful_runs = unsuccessful_runs = 0
  while successful_runs < number_trajectories:
    # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
    # Sample theta1 uniformly, then solve for the theta2 that would give the prescribed potential energy
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
          
    t = np.arange(0, tmax+dt_simulation, dt_simulation)

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

    # Check that the calculation conserves total energy to within some tolerance.
    FRACTIONAL_EDRIFT_TOL = 1e-3
    # Total energy from the initial conditions
    E = calc_E(y0)
    if np.max(np.abs(calc_E(y) - E)/E) > FRACTIONAL_EDRIFT_TOL:
      unsuccessful_runs += 1
      continue
    # Unpack z and theta as a function of time
    theta1, theta2 = y[:,0], y[:,2]

    run_arr = y[::save_every]  ## Sieve out to save a smaller file
    giant_arr.append(run_arr)
    successful_runs += 1
    if successful_runs % 1000 == 0:
      print(f'{successful_runs}/{number_trajectories} completed, {unsuccessful_runs} unsuccessful')

  giant_arr = np.stack(giant_arr, 0)
  print(f'Saving simulation chunk, shape {giant_arr.shape}')
  np.save(os.path.join(data_dir, f'E_{energy_over_g}_l1_{L1:.02f},l2_{L2:.02f},m1_{m1:.02f},m2_{m2:.02f}_{simulation_chunk}.npy'), giant_arr)
