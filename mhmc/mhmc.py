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
    self.prev_eval = functions[0](self.prev) * functions[1](self.prev)
    
  def importance_w(psi1, psi2):
    return torch.mean(torch.abs(psi1)*torch.abs(psi2))

  def integrate():
    self.random_walk(self.prev)
    return self.prev_eval

  def integration():
    vals = self.functions[0](x) * self.functions[1](x)
    return torch.mean(vals/self.importance_w(psi))

  def random_walk(x, acceptance_rate=0.2):
    _, num_dim = x.shape
    for dim in range(num_dim):
      dim_range = self.dims[dim][1] - self.dims[dim][0]
      x[:][dim] += torch.normal(0, dim_range/5, size=x.shape)
    if self.prev.shape != x.shape:
        exit()
        #print(f"ERROR: Trying to update integral of {self.prev.shape} with integral of {x.shape}.")
    for i, el in enumerate(x):
      #TODO: Add optional arguments
      #vals = self.functions(x, self.args)
      #vals = self.functions[0](x) * self.functions[1](x)
      vals = self.integration()
      for i, el in enumerate(vals):
        if el/self.prev_eval[i] >= 1 or random.uniform(0,1) < acceptance_rate:
          self.prev[i] = el
      #self.prev_eval = self.functions(self.prev, self.args)
      self.prev_eval = self.integration()
      self.step += 1

