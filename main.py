from typing import cast

import os
import pickle
import time
from tqdm import tqdm
import shutil


import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_logger

sns.color_palette("bright")

import torch
import torchdiffeq
torch.set_default_tensor_type(torch.DoubleTensor)


from guilda.power_network import SimulationResult, PowerNetwork
from guilda.generator import Generator

from node import NODEMechanized, NODENeural, loss_func

os.makedirs('fig', exist_ok=True)
logger = get_logger(__name__, 'fig/logfile.log')

gpu = 0
device = 'cuda'

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
omega0 = torch.tensor(omega0_np)
L_1_np = get_gen_params(gen1)
L_2_np = get_gen_params(gen2)

# TODO this seems to be a bug of GUILDA
# L_1_np[2] = 2.04379374735951
# L_2_np[2] = 1.98829642487374
L_1_np[3] = -0.513568531598284
L_2_np[3] = 0.486559381709619

L_1 = torch.from_numpy(L_1_np) 

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

X_eq_np = np.vstack(net.x_equilibrium)

x0 = torch.from_numpy(X_eq_np + np.array([[np.pi / 6], [0], [0], [0]])).to(device)

t_orig = result.t
x_orig = np.hstack([
    result[1].x,
    result[2].x
])

# resample
interp_func = interp1d(t_orig, x_orig, axis=0)
smpl_rate = 30 # unit: Hz
t_np = np.linspace(0, 20, 20 * smpl_rate + 1)
x_np = interp_func(t_np)

t = torch.from_numpy(t_np).to(device)

true_x = torch.from_numpy(x_np).to(device)

true_omega1 = true_x[:, 1]
true_omega2 = true_x[:, 3]

# Visualization

def visualize(t, true_x, pred_x, itr):
    if not viz:
        return
    
    fig, ((x_delta1, x_delta2), (x_omega1, x_omega2)) = plt.subplots(2, 2, figsize=(12, 8), facecolor='white')

    plots = [x_delta1, x_omega1, x_delta2, x_omega2]
    with torch.no_grad():
        for i, plot in enumerate(plots):
            
            plot.plot(t.cpu().numpy(), true_x.cpu().numpy()[:,i], 'g-', lw=1, label='True')
            plot.plot(t.cpu().numpy(), pred_x.cpu().numpy()[:,i], 'b-', lw=1, label='Predicted')
        plot.legend()

    x_delta1.set_title('Local Subsystem Model')
    x_delta2.set_title('Environment Model')

    x_delta1.set_ylabel('Rotor Angle')
    x_omega1.set_ylabel('Frequency Deviation')

    x_omega1.set_xlabel('Time [s]')
    x_omega2.set_xlabel(f'Iteration: {itr}')
    
    
    fig_cur = f'fig/{itr}.png'
    for ax in x_delta1, x_omega1, x_delta2, x_omega2:
        ax.grid(True)
    plt.savefig(fig_cur, dpi=300, bbox_inches='tight')
    if itr >= 0:
        shutil.copy(f'fig/{itr}.png', 'fig/_latest.png')
    return plt.close()


# Configuration

viz = True

data_size = t_np.size
batch_time = smpl_rate * 1

niters = 114514
batch_size = 20
per_slice_training = False
test_freq = 5 if per_slice_training else 10


freq_weight = 0.5
freq_wnd = torch.hann_window(batch_time).reshape(-1, 1, 1).to(device)
time_weight = (torch.linspace(0.5, 2, batch_time) ** 2).reshape(-1, 1, 1).to(device)

def get_batch():
    s = np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False)
    batch_x0 = true_x[s]
    batch_t = t[:batch_time]
    batch_x = torch.stack([true_x[t0: t0 + batch_time] for t0 in s], dim=1)
    return s, batch_x0.to(device), batch_t.to(device), batch_x.to(device)

tol_options = dict( 
    rtol = 1e-8,
    atol = 1e-8,
)
solver_options = dict( 
    **tol_options,
    method = 'rk4',
)

# configure params

real_params = np.concatenate([L_2_np, L_Y_np]).flatten()

# init_params_np = np.random.random((6, )) * 2
init_params_np = real_params + real_params * 0.5 * np.random.random((6, ))
# init_params_np[:2] /= 10

init_params = torch.from_numpy(
    init_params_np
)


# func = NODEMechanized(
#     omega0, L_1, 
#     init_params[:2], 
#     init_params[2:4],
#     init_params[4:]
# ).to(device)

# param_groups = [
#     {'params': func.params0, 'lr': 1e-2}, # M, D
#     {'params': func.params1, 'lr': 1e-3}, # V, P
#     {'params': func.params2, 'lr': 5e-4} # B, G
# ]

func = NODENeural(omega0).to(device)
param_groups = [{
    'params': func.parameters(), 'lr': 0.01
}]

optimizer = torch.optim.RMSprop(
    param_groups,
)
# optimizer = torch.optim.Adam(
#     param_groups,
# )

# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer, 
#     step_size = 100, gamma=0.5, verbose=False
# )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', factor=0.5, 
    patience=40 if per_slice_training else 60, 
    verbose=False,
    min_lr=1e-7
)
clamp_min = 0.1

# main cycle

grad_norm_values = []
parameters_data = []
start_time = time.time()

true_x0 = x0.T
true_x_ = true_x[:, None, :]
def pred_and_calc_loss():
    with torch.no_grad():
        loss, pred_x = loss_func(
            func, 
            t, 
            true_x0, 
            true_x_, 
            solver_options,
        )
    return pred_x[:, 0, :], loss

pred_x_total, loss_total = pred_and_calc_loss()


for itr in range(0, niters):
    
    freq_weight_ = 0 # freq_weight if itr < 75 else 0
    
    if itr % test_freq == 0:
        with torch.no_grad():
            visualize(t, true_x, pred_x_total, itr)
            if not isinstance(func, NODENeural):
                logger.info('Current Parameters:')
                for name, param in func.named_parameters():
                    logger.info(f"{name}: {param.data}")
                logger.info('------')
        # assert False
            
    start = time.time()
            
    s, batch_x0, batch_t, batch_x = get_batch()
    optimizer.zero_grad()
    
    loss_batch = 0
    
    if not per_slice_training:
        loss, _ = loss_func(
            func, 
            batch_t, 
            batch_x0, 
            batch_x, 
            solver_options,
            freq_weight = freq_weight_,
            freq_wnd = freq_wnd,
            time_weight = time_weight,
        )
        loss.backward() # Calculate the dloss/dparameters
        optimizer.step() # Update value of parameters
        visualize(batch_t, batch_x[:, 0], _[:, 0], -1)
        if isinstance(func, NODEMechanized):
            with torch.no_grad():
                func.params1.clamp_(min=clamp_min)
                func.params2[:1].clamp_(min=clamp_min) # make sure values are positive
        loss_batch = loss.item()
    
    for batch_n in range(0, batch_size) if per_slice_training else []:
        
        loss_minibatch, _ = loss_func(
            func, 
            batch_t, 
            batch_x0[batch_n: batch_n + 1], 
            batch_x[:, batch_n: batch_n + 1], 
            solver_options,
            freq_weight = freq_weight_,
            freq_wnd = freq_wnd,
            time_weight = time_weight,
        )
        loss_minibatch.backward() # Calculate the dloss/dparameters
        optimizer.step() # Update value of parameters

        with torch.no_grad():
            loss_batch += loss_minibatch.item()
            if isinstance(func, NODEMechanized):
                func.params0.clamp_(min=clamp_min)
                func.params1.clamp_(min=clamp_min)
                # make sure values are positive

        for param in func.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()**2
        grad_norm = np.sqrt(grad_norm)
        grad_norm_values.append(grad_norm)
    
    if per_slice_training:
        loss_batch /= batch_size
    
    pred_x_total, loss_total = pred_and_calc_loss()
    
    # scheduler.step()
    scheduler.step(loss_total)
    end = time.time()
    
    logger.info(f'Iteration {itr:d} | Loss T: {loss_total.item():.12f} B: {loss_batch:.12f} | LR: {optimizer.param_groups[0]["lr"]} | Time Elapsed: {end - start:.4f}s')
    
    if loss_total < 1e-6:
        logger.info('break')
        for name, param in func.named_parameters():
            logger.info(f"{name}: {param.data}")
        break
    