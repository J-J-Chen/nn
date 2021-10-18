import numpy as np
import torch
import torch.optim as optim
import time
import random

class mhmc():
  def __init__(self, functions, dims, args=None):
    self.functions = functions
    self.args = args
    self.dims = dims
    self.step = 0
    self.prev = torch.random.normal(dims)
    self.prev_eval = self.functions(self.prev)
    
    self.transition_model = lambda x: [x[0], torch.random.normal(x[1], 0.5, (1,))]

  def integrate():
    pass

  def mh_step():
    x_new = self.transition_model(x)
    x_like = self.man_log_like_norm(x, self.prev)
    x_update_like = self.man_log_like_norm(x_new, data)
    self.step += 1

  def prior(x):
    return 0 if x[1] <= 0 else 1

  def man_log_like_norm(x, data):
    return torch.sum(-torch.log(x[1]*torch.sqrt(2*np.pi))-((data-x[0])**2)\
        / (2*x[1]**2))

  def random_walk(x, acceptance_rate=0.2):
    _, num_dim = x.shape
    for dim in range(num_dim):
      dim_range = self.dims[dim][1] - self.dims[dim][0]
      x[:][dim] += torch.normal(0, dim_range/5, size=x.shape)
    if self.prev.shape != x.shape:
      print(f"ERROR: Trying to update integral of {self.prev.shape} with integral of {x.shape}.")
    for i, el in enumerate(x):
      #TODO: Add optional arguments
      vals = self.functions(x, self.args)
      for i, el in enumerate(vals):
        if el/self.prev_eval[i] >= 1 or random.uniform(0,1) < acceptance_rate:
          self.prev[i] = el
      self.prev_eval = self.functions(self.prev, self.args)

  def dist(x):
    return torch.sqrt(torch.sum(x**2))

  def norm_const():
    pass

