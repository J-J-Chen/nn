## Import

import torch
import numpy as np
import matplotlib.pyplot as plt
import motraining
from motraining.training_instance import training_instance

## Training 1d

nDim, nOutput, nLayers, nNeurons, nEpochs, nSplit, lr = 1, 1, 4, 100, 9000, 500, 8e-4
name = "ti1_1"
time_boundaries = (-6,6)
training = training_instance(nDim, nOutput, "box", time_boundaries, name)
#training_instances[name] = training
training.create_model(nLayers, nNeurons, lr, name)
training.loadModel(name, False)
training.train(nEpochs, nSplit, minLoss=1e-10, liveplot=100)

### Graph 1d
for index in range(nOutput):
  N = 1000
  time_boundaries = (-6,6)
  xTest = torch.linspace(time_boundaries[0], time_boundaries[1], N, requires_grad=True).reshape(-1,1)

  print(xTest.shape)
  psisTests = training.parametric_eq(xTest).unbind(-1)
  psiTest = psisTests[index].cpu().detach().numpy().reshape(-1,1)

  xTest = xTest.detach().cpu().numpy()

  print(xTest.shape)
  plt.figure()
  plt.plot(xTest, psiTest, '--b', label='x', alpha=0.5);
  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("E: " +str(3.503491857184)) #str(training.curr_model.es[index].item()))
  plt.savefig("figures/multi_"+str(index)+".png")
  plt.close()

## Training instances

training_instances = {}

## Training 2d qho

nDim, nOutput, nLayers, nNeurons, nEpochs, nSplit, lr = 2, 3, 4, 80, 9000, 500, 8e-3
name = "qho_dim2_out3_1"
time_boundaries = (-6,-6,6,6)
training = training_instance(nDim, nOutput, "qho", time_boundaries, name)
training_instances[name] = training
training.create_model(nLayers, nNeurons, lr, name)
training.loadModel(name)
loss, time = training.train(nEpochs, nSplit, minLoss=1e-3, liveplot=100)

## Graph 2d qho

model = training.curr_model
for index in range(nOutput):
  N = 1000
  xTest = torch.linspace(-6, 6, N).reshape(-1, 1).repeat(1,N).reshape(-1, 1)
  yTest = torch.linspace(-6, 6, N).repeat(N).reshape(-1, 1)
  xTest.requires_grad = True
  yTest.requires_grad = True

  t_val = torch.cat((xTest, yTest), -1)
  psiTest = training.parametric_eq(t_val)
  psi_tests = psiTest.unbind(-1)
  psiTest = psi_tests[index].cpu().detach().numpy().reshape(-1)
  psiTest = np.absolute(psiTest)

  plt.figure()
  plt.imshow(psiTest.reshape(-1,N), extent=[-6,6,-6,6])
  plt.colorbar().set_label("psi")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.savefig("figures/"+str(training.curr_model_name) + "_" + str(index) +".png")
  plt.close()

## Training 2d box

nDim, nOutput, nLayers, nNeurons, nEpochs, nSplit, lr = 2, 2, 4, 80, 8000, 500, 8e-4
name = "ti2_3out"
time_boundaries = (0,0,1,1)
training = training_instance(nDim, nOutput, "box", time_boundaries, name)
training_instances[name] = training
training.create_model(nLayers, nNeurons, lr, name)
training.loadModel(name)
loss, time = training.train(nEpochs, nSplit, minLoss=1e-3, liveplot=100)

## Graph 2d box

model = training.curr_model
for index in range(nOutput):
  N = 100
  xTest = torch.linspace(0, 1, N).reshape(-1, 1).repeat(1,N).reshape(-1, 1)
  yTest = torch.linspace(0, 1, N).repeat(N).reshape(-1, 1)
  xTest.requires_grad = True
  yTest.requires_grad = True

  time_boundaries = (0, 0, 1, 1)

  t_val = torch.cat((xTest, yTest), -1)
  psiTest = training.parametric_eq(t_val)
  psi_tests = psiTest.unbind(-1)
  psiTest = psi_tests[index].cpu().detach().numpy().reshape(-1)
  psiTest = np.absolute(psiTest)

  plt.figure()
  plt.imshow(psiTest.reshape(-1,N), extent=[0,1,0,1])
  plt.colorbar().set_label("psi")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.savefig("figures/"+str(training.curr_model_name) + "_" + str(index) +".png")
  plt.close()

## Solve 3-d

nDim, nOutput, nLayers, nNeurons, nEpochs, nSplit, lr = 3, 1, 4, 100, 5000, 500, 8e-3
name = "ti3_1_1"
time_boundaries = (0,0,0,1,1,1)
training = training_instance(nDim, nOutput, "box", time_boundaries, name)
training.create_model(nLayers, nNeurons, lr, name)
training.loadModel(name, False)
training.train(nEpochs, nSplit, minLoss=1e-3, liveplot=100)

## Graph 3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

model = training.curr_model

for index in range(nOutput):
  N = 100
  xTest = torch.linspace(0, 1, N).reshape(-1, 1).repeat(1,N).reshape(-1, 1)
  yTest = torch.linspace(0, 1, N).repeat(N).reshape(-1, 1)
  zTest = 0.5*torch.ones_like(yTest)
  xTest.requires_grad = True
  yTest.requires_grad = True
  zTest.requires_grad = True

  time_boundaries = (0,0,0,1,1,1)

  t_val = torch.cat((xTest, yTest, zTest), -1)
  psiTest = training.parametric_eq(t_val)
  psi_tests = psiTest.unbind(-1)
  psiTest = psi_tests[index].cpu().detach().numpy().reshape(-1)
  psiTest = np.absolute(psiTest)

  plt.figure()
  plt.imshow(psiTest.reshape(-1,N), extent=[0,1,0,1])
  plt.colorbar().set_label("psi")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.savefig("figures/"+str(training.curr_model_name) + "_" + str(index) +".png")
  plt.close()

