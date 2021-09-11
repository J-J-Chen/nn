## Import

import torch
import numpy as np
import matplotlib.pyplot as plt
import motraining
from motraining.training_instance import training_instance

## Consts

dimensions_arr = [1,2,3]
nOutput_arr = [1,2,3,4,5,6]

## Training
f = open("results.csv", "a")
f.write("dim,nOutput,loss,time,reach_min\n")

for dim in dimensions_arr:
  for output in nOutput_arr:
    nDim, nOutput, nLayers, nNeurons, nEpochs, nSplit, lr = dim, output, 4, 100, 500, 2000, 8e-4
    name = "mo_dim" + str(dim) + "_output" + str(output)
    time_boundaries = [item for item in [0,1] for i in range(dim)]
    training = training_instance(nDim, nOutput, "box", time_boundaries, name)
    training.create_model(nLayers, nNeurons, lr, name)
    training.loadModel(name, False)
    loss, time = training.train(nEpochs, nSplit, minLoss=1e-10)
    f.write(str(dim)+","+str(output)+","+str(sum(loss))+","+str(time)+","+str(training.reach_min)+"\n")

