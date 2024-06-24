from typing import cast

import os
import pickle
import time


import numpy as np
from scipy.interpolate import interp1d

import seaborn as sns

from solver_wrapper import create_estimator
from utils import USE_CACHE, get_logger

sns.color_palette("bright")

import torch
torch.set_default_tensor_type(torch.DoubleTensor)
logger = get_logger(__name__, 'run/logfile.log')


model = create_estimator(USE_CACHE)

model.init()
params = model.get_current_params()

for itr in range(0, 2000):
  
  if itr % 5 == 0:
    logger.info(f'Params: {params}')
    logger.info('')
  
  start = time.time()
  loss, grad_norm = model.iterate()
  end = time.time()
  params = model.get_current_params()
    
  
  logger.info(f'Iteration {itr:d} | Loss: {loss} | Grad Norm: {grad_norm} | Time Elapsed: {end - start:.4f}s')
  
  if loss < 4e-10:
    logger.info('break')
    logger.info(params)
    break
  
  