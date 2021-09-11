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
nDim, nOutput, nLayers, nNeurons, nEpochs, nSplit, lr = 1, 1, 4, 100, 500, 5000, 8e-4
name = "mo_dim" + str(dim) + "_output" + str(output)
time_boundaries = [item for item in [0,1] for i in range(1)]
training = training_instance(nDim, nOutput, "box", time_boundaries, name)
training.create_model(nLayers, nNeurons, lr, name)
training.loadModel(name, False)
loss, time = training.train(nEpochs, nSplit, minLoss=1e-10)

