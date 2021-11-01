import numpy as np
import torch
import torch.optim as optim
import time

from mhmc import mhmc

integrator = mhmc((torch.sin, torch.cos), 1)

for i in range(10):
  val = integrator.integrate()
  print("VAL:\n")
  print(val)
  print("\n Samples:\n")
  print(val.prev)


