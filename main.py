from typing import cast

import os
import pickle
import time


import numpy as np
from scipy.interpolate import interp1d

import seaborn as sns

from solver_wrapper import EnvModelEstimator, ScenarioParameters, SolverParameters, default_optim_factory
from utils import get_logger

sns.color_palette("bright")

import torch
torch.set_default_tensor_type(torch.DoubleTensor)


from guilda.power_network import PowerNetwork
from guilda.generator import Generator


os.makedirs('fig', exist_ok=True)
logger = get_logger(__name__, 'fig/logfile.log')


# import data
with open('./run/saved.pkl', 'rb') as f:
    obj = pickle.load(f)
    net = cast(PowerNetwork, obj[0])
    result = cast(PowerNetwork, obj[3])
    
gen1 = cast(Generator, net.a_bus_dict[1].component)
gen2 = cast(Generator, net.a_bus_dict[2].component)

# define parameters

def get_gen_params(gen: Generator):
    # M, D, V*, P*
    p = gen.parameter
    Vf, Pm = gen.u_equilibrium.flatten()
    return np.array([p.M, p.D, Vf, Pm])

omega0_np = 2 * 60 * np.pi
L_1_np = get_gen_params(gen1)
L_2_np = get_gen_params(gen2)

L_1_np[3] = -0.513568531598284
L_2_np[3] = 0.486559381709619

# G, B
Xp = np.diag([
  gen1.parameter.Xd_prime,
  gen2.parameter.Xd_prime
])
Y = net.get_admittance_matrix()
G = Xp - 1j * (Xp @ np.conj(Y) @ Xp)
Y_red = -1j * np.linalg.inv(G)

Y_red_12 = Y_red[0, 1]

# G, B of reduced admittance matrix
L_Y_np = [Y_red_12.real, Y_red_12.imag]

# eq. state

X_eq_np = np.vstack(net.x_equilibrium)

t_orig = result.t
x_orig = np.hstack([
    result[1].x,
    result[2].x
])


# resample
interp_func = interp1d(t_orig, x_orig, axis=0)
smpl_rate = 20 # unit: Hz
t_np = np.linspace(0, 20, 20 * smpl_rate + 1)
x_np = interp_func(t_np)

factor = 1.2

scenario = ScenarioParameters(
  omega=omega0_np,
  
  M_1=L_1_np[0],
  D_1=L_1_np[1],
  V_field_1=L_1_np[2],
  P_mech_1=L_1_np[3],
  
  M_2=L_2_np[0] * factor,
  D_2=L_2_np[1] * factor,
  V_field_2=L_2_np[2] * factor,
  P_mech_2=L_2_np[3] * factor,
  
  G_12=L_Y_np[0],
  B_12=L_Y_np[1],
  
  t=t_np,
  true_x=x_np,
)

sol_params = SolverParameters(
  device='cuda',
  optim_factory=default_optim_factory,
  batch_size=20,
  batch_time=30,
  ode_params={
    'method': 'rk4'
  }
)


model = EnvModelEstimator(
  scenario,
  sol_params
)

model.init()
params = model.get_current_params()

for itr in range(0, 114514):
  
  if itr % 5 == 0:
    print('Params:', params)
    print()
  
  start = time.time()
  loss, grad_norm = model.iterate()
  end = time.time()
  params = model.get_current_params()
    
  
  logger.info(f'Iteration {itr:d} | Loss: {loss} | Grad Norm: {grad_norm} | Time Elapsed: {end - start:.4f}s')
  
  if loss < 1e-11:
    print('break')
    print(params)
    break
  
  