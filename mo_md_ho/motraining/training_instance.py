import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
from mpl_toolkits import mplot3d
import math
import time
import copy
from os import path
from os import mkdir
from numpy.random import uniform
import sys

from torchquad import MonteCarlo, Trapezoid, enable_cuda, set_log_level

from odeNet import odeNet

dtype = torch.float

set_log_level("CRITICAL")

if not path.isdir("models"): mkdir("models")
if not path.isdir("loss"): mkdir("loss")
if not path.isdir("figures"): mkdir("figures")

# Check to see if gpu is available. If it is, use it else use the cpu
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    enable_cuda()
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('No GPU found, using cpu')


def dfx(x,f):
  #return grad([f], [x], grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
  return grad(f, x, grad_outputs=torch.ones(f.shape, dtype=dtype), create_graph=True)[0]

class training_instance():
  def __init__(self, nDim, nOutput, eq_type, time_boundaries, name):
    self.bounds = [float(a) for a in time_boundaries]
    self.nDim = nDim
    self.nOutput = nOutput
    self.nSplit = None
    self.name = name
    self.models = {}
    self.optimizers = {}
    self.integrator = None
    self.curr_model = None
    self.curr_optimizer = None
    self.curr_model_name = None
    self.psis_to_integrate = None
    self.reach_min = None
    if self.nDim == 1 and eq_type == "box":
      self.loss_fn = self.Eqs_Loss_1d_box
    elif self.nDim == 2 and eq_type == "box":
      self.loss_fn = self.Eqs_Loss_2d_box
    elif self.nDim == 3 and eq_type == "box":
      self.loss_fn = self.Eqs_Loss_3d_box
    elif self.nDim == 1 and eq_type == "qho":
      self.loss_fn = self.Eqs_Loss_1d_box
    elif self.nDim == 2 and eq_type == "qho":
      self.loss_fn = self.Eqs_Loss_2d_qho
    elif self.nDim == 3 and eq_type == "qho":
      self.loss_fn = self.Eqs_Loss_3d_qho
    else:
      print(f"{nDim} is not supported or {eq_type} does not exist.")
      exit()
    self.parametric_eq = self.parametricSolutions_bounded if eq_type == "box" else\
        self.parametricSolutions_unbound

  def generate_test(self, numPoints):
    x = torch.linspace(self.bounds[0], self.bounds[2], numPoints, requires_grad=True).T
    y = torch.linspace(self.bounds[1], self.bounds[3], numPoints, requires_grad=True).T
    return torch.cat((x.unsqueeze(1),y.unsqueeze(1)),-1)

  def generate_points(self, numPoints=None):
    boundR = torch.tensor(self.bounds[self.nDim:], requires_grad=True).reshape(-1,1)
    boundL = torch.tensor(self.bounds[:self.nDim], requires_grad=True).reshape(-1,1)
    num_points = self.nSplit if numPoints == None else numPoints
    return torch.unbind(torch.rand(self.nDim, num_points, requires_grad=True) * (boundR-boundL) + boundL)
    #return torch.unbind(torch.rand(self.nDim, num_points, requires_grad=True).sort(dim=-1)[0] * (boundR-boundL) + boundL)
  
  def parametricSolutions_bounded(self, t):
    t_points = torch.stack(t,-1) if isinstance(t, tuple) else t
    boundR = torch.tensor(self.bounds[self.nDim:])
    boundL = torch.tensor(self.bounds[:self.nDim])
    return (1-torch.exp(-t_points+boundL)).prod(-1,keepdim=True) *\
    (1-torch.exp(t_points-boundR)).prod(-1,keepdim=True)*self.curr_model.forward(t_points)
    #x,y = t_points.unbind(-1)
    #x = x.reshape(-1,1)
    #y = y.reshape(-1,1)
    #ans = ((1-torch.exp(-x)) * (1-torch.exp(x-torch.ones_like(x))) * (1-torch.exp(-y)) * (1-torch.exp(y-torch.ones_like(y)))).reshape(-1,1).repeat(1,self.nOutput) * self.curr_model.forward(t_points)
    #return ans

  def parametricSolutions_unbound(self, t):
    t_points = torch.stack(t,-1) if isinstance(t, tuple) else t
    return 1/(t_points**2+1).prod(-1,keepdim=True)*self.curr_model.forward(t_points)
    #boundR = torch.tensor(self.bounds[self.nDim:])
    #boundL = torch.tensor(self.bounds[:self.nDim])
    #return (1-torch.exp(-t_points+boundL)).prod(-1,keepdim=True) *\
    #(1-torch.exp(t_points-boundR)).prod(-1,keepdim=True)*self.curr_model.forward(t_points)

  def parametricIntegration(self, t):
    psis = self.parametric_eq(t)
    psis_separate = psis.unbind(-1)
    #psis_separate = [psi.reshape(-1) for psi in psis.split(1)]
    return psis_separate[self.psis_to_integrate[0]]*psis_separate[self.psis_to_integrate[1]]

  def Eqs_Loss_1d_box(self):
    t = self.generate_points()
    x = t[0]
    psis = self.parametric_eq(t).unbind(-1)
    loss = 0
    norm_loss = 0
    orthos = 0
    for i,psi in enumerate(psis):
      xdot = dfx(x, psi)
      x2dot = dfx(x, xdot)
      loss += (x2dot - x**2*psi + 2*3.5*psi).pow(2).mean()
      #loss += (x2dot - x**2*psi + 2*self.curr_model.es[i]*psi).pow(2).mean()
      self.psis_to_integrate = (i,i)
      norm_loss += (self.integrator.integrate(self.parametricIntegration, self.nDim, N=self.nSplit,\
          integration_domain=torch.tensor(self.bounds).reshape(2,-1).T)-1).pow(2).mean()
    #print(f"loss: {loss}, norm_loss: {norm_loss}, e: {self.curr_model.es[0]}, psi: {psis[0].max()}")
    for i in range(self.nOutput-1):
      for j in range(i+1, self.nOutput):
        self.psis_to_integrate = (i,j)
        orthos += self.integrator.integrate(self.parametricIntegration, dim=self.nDim, N=self.nSplit, integration_domain=torch.tensor(self.bounds).reshape(2,-1).T).pow(2)
    if orthos == 0: orthos = torch.tensor(0., requires_grad=True)
    #last_layer = (self.curr_model.output[0].weight).pow(2).sum() + (self.curr_model.output[0].bias).pow(2).sum()
    return 2*loss, 10*norm_loss, orthos

  def Eqs_Loss_2d_box(self):
    t = self.generate_points()
    x,y = t
    psis = self.parametric_eq(t).unbind(-1)
    loss = 0
    norm_loss = 0
    for i, psi in enumerate(psis):
      xdot = dfx(x, psi)
      x2dot = dfx(x, xdot)
      ydot = dfx(y, psi)
      y2dot = dfx(y, ydot)
      #lol = [9.869, 26.674, 26.674]
      #loss += (x2dot + y2dot + 2*lol[i]*psi).pow(2).mean()
      loss += (x2dot + y2dot + 2*self.curr_model.es[i]*psi).pow(2).mean()
      self.psis_to_integrate = (i,i)
      norm_loss += (self.integrator.integrate(self.parametricIntegration,\
          dim=self.nDim, N=self.nSplit, integration_domain=torch.tensor(self.bounds).reshape(2,-1).T)-1).pow(2).mean()
    B = 0
    for i in range(len(psis)-1):
      for j in range(i+1, len(psis)):
        self.psis_to_integrate = (i,j)
        B += self.integrator.integrate(self.parametricIntegration, dim=self.nDim, N=self.nSplit,\
            integration_domain=torch.tensor(self.bounds).reshape(2,-1).T).pow(2)#.mean()
    #last_layer = (self.curr_model.output[0].weight).pow(2).sum() + (self.curr_model.output[0].bias).pow(2).sum()
    orthos = torch.tensor(0.,requires_grad=True) if B==0 else B #(last_layer-1)**2 + 700*(B)
    #sum_weights = 0
    #for i in range(self.curr_model.layers+2):
      #sum_weights += self.curr_model.ffn[i][0].weight.pow(2).sum()
    return 1.2*loss, 300*norm_loss, 70*orthos

  def Eqs_Loss_3d_box(self):
    t = self.generate_points()
    x,y,z = t
    psis = self.parametric_eq(t).unbind(-1)
    loss = 0
    norm_loss = 0
    for i, psi in enumerate(psis):
      xdot = dfx(x, psi)
      x2dot = dfx(x, xdot)
      ydot = dfx(y, psi)
      y2dot = dfx(y, ydot)
      zdot = dfx(z, psi)
      z2dot = dfx(z, zdot)
      f = x2dot + y2dot + z2dot + np.pi**2*self.curr_model.es[i]*psi
      loss += (f.pow(2)).mean()
      self.psis_to_integrate = (i,i)
      norm_loss += (self.integrator.integrate(self.parametricIntegration,\
          dim=self.nDim, N=self.nSplit, integration_domain=torch.tensor(self.bounds).reshape(2,-1).T)-1).pow(2).mean()
    B = 0
    for i in range(len(psis)-1):
      for j in range(i+1, len(psis)):
        self.psis_to_integrate = (i,j)
        B += self.integrator.integrate(self.parametricIntegration, dim=self.nDim, N=self.nSplit,\
            integration_domain=torch.tensor(self.bounds).reshape(2,-1).T).pow(2).mean()
    #last_layer = (self.curr_model.output[0].weight).pow(2).sum() + (self.curr_model.output[0].bias).pow(2).sum()
    orthos = torch.tensor(0.,requires_grad=True) if B==0 else B #(last_layer-1)**2 + 700*(B)
    #sum_weights = 0
    #for i in range(self.curr_model.layers+2):
    #  sum_weights += self.curr_model.ffn[i][0].weight.pow(2).sum()
    return 2*loss, 950*norm_loss, 10*orthos

  def Eqs_Loss_2d_qho(self):
    t = self.generate_points()
    x,y = t
    psis = self.parametric_eq(t).unbind(-1)
    loss = 0
    norm_loss = 0
    for i, psi in enumerate(psis):
      xdot = dfx(x, psi)
      x2dot = dfx(x, xdot)
      ydot = dfx(y, psi)
      y2dot = dfx(y, ydot)
      f = x2dot + y2dot + (-x**2 - y**2 + 2*(self.curr_model.es[i]))*psi
      loss += (f.pow(2)).mean()
      self.psis_to_integrate = (i,i)
      norm_loss += (self.integrator.integrate(self.parametricIntegration,\
          dim=self.nDim, N=self.nSplit, integration_domain=torch.tensor(self.bounds).reshape(2,-1).T)-1).pow(2).mean()
    B = 0
    for i in range(len(psis)-1):
      for j in range(i+1, len(psis)):
        self.psis_to_integrate = (i,j)
        B += self.integrator.integrate(self.parametricIntegration, dim=self.nDim, N=self.nSplit,\
            integration_domain=torch.tensor(self.bounds).reshape(2,-1).T).pow(2)#.mean()
    #last_layer = (self.curr_model.output[0].weight).pow(2).sum() + (self.curr_model.output[0].bias).pow(2).sum()
    orthos = torch.tensor(0.,requires_grad=True) if B == 0 else B #(last_layer-1)**2 + 700*(B)
    #sum_weights = 0
    #for i in range(self.curr_model.layers+2):
      #sum_weights += self.curr_model.ffn[i][0].weight.pow(2).sum()
    return loss, 100*norm_loss, 80*orthos

  def create_model(self, nLayers, nNeurons, lr, name, activation=None, optimizer=None, integrator=None):
    self.models[name] = odeNet(activation = activation, input = self.nDim, layers=nLayers, output=self.nOutput, D_hid=nNeurons)
    #self.optimizers[name] = optim.SGD(self.models[name].parameters(), lr=lr, momentum=0.9)
    self.optimizers[name] = optim.Adam(self.models[name].parameters(), lr=lr, betas=[0.99,0.999])
    self.integrator = MonteCarlo() if integrator == None else integrator

  def loadModel(self, name, load_weights=False):
    if name in self.models:
      PATH = "models/" + name
      self.curr_model = self.models[name]
      self.curr_optimizer = self.optimizers[name]
      self.curr_model_name = name
      if path.exists(PATH) and load_weights:
        checkpoint = torch.load(PATH)
        fc0 = self.models[self.curr_model]
        optimizer = self.optimizer
        fc0.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        fc0.train()
    else:
      print(f"{name} does not exist. Run create_model first")
      sys.exit()

  def train(self, nEpochs, nSplit, minLoss=1e-3, liveplot=0, min_loss=1e-2):
    self.nSplit = nSplit

    Loss_history = []
    Llim =  1 
    normLossHistory = []
    equationLossHistory = []
    t_old=0
    Ltot = 0    
    PATH = "models/"+str(self.name)
    
    fc1 = copy.copy(self.curr_model)
    TeP0 = time.time()
    using_gpu = torch.cuda.is_available()
    if using_gpu:
      scaler = torch.cuda.amp.GradScaler()
    for tt in range(t_old,int(nEpochs)+t_old ):              
      loss = 0.0
      normLoss = 0.0
      equationLoss = 0.0
      orthosLoss = 0.0
      
      if using_gpu:
        with torch.cuda.amp.autocast():
          equationLoss, normLoss, orthosLoss = self.loss_fn()
          Ltot = equationLoss + normLoss + orthosLoss 
        scaler.scale(Ltot).backward(retain_graph=False)
        scaler.step(self.curr_optimizer)
        scaler.update()
      else:
        equationLoss, normLoss, orthosLoss = self.loss_fn()
        Ltot = equationLoss + normLoss + orthosLoss
        Ltot.backward(retain_graph=False); #True
        self.curr_optimizer.step();
                 
      self.curr_optimizer.zero_grad()

      loss = (equationLoss, normLoss, orthosLoss)

      detached = [L.item() for L in loss]
      Loss_history.append(detached)
      #normLossHistory.append(normLoss)
      #equationLossHistory.append(equationLoss)

      if liveplot != 0:
        if (tt+1)%liveplot == 0 and tt != 0:
          energy = ""
          for e in range(len(self.curr_model.es)):
            energy = energy + ", E" + str(e+1) + ": " + str(self.curr_model.es[e].item())[0:5]
          print(f'Loss: {Ltot:.3f}, Norm: {normLoss:.3f}, Orthos: {orthosLoss:.3f}, Step: {tt+1}', energy, end='\r')     

      if Ltot < min_loss:
        self.reach_min = tt

#Keep the best model (lowest loss) by using a deep copy
      if  tt > 0.8*nEpochs and Ltot < Llim:
        fc1 =  copy.copy(self.curr_model)
        Llim=Ltot 

# break the training after a thresold of accuracy
      if Ltot < minLoss :
        fc1 =  copy.copy(self.curr_model)
        print('Reach minimum requested loss')
        break

    TePf = time.time()
    runTime = TePf - TeP0     
    
    torch.save({
    'epoch': tt,
    'model_state_dict': fc1.state_dict(),
    'optimizer_state_dict': self.curr_optimizer.state_dict(),
    'loss': Ltot,
    }, "models/"+str(self.curr_model_name))

    self.get_loss(runTime, Loss_history)
    return Loss_history[-1], runTime/60

  def reached_min(self):
    return self.reach_min

  def get_loss(self, runTime, loss):
    loss = torch.tensor(loss)
    print('Training time (minutes):', runTime/60)
    print('Training Loss: ',  loss[-1] )
    plt.figure()
    color_cycle = ['-r', '-g', '-y', '-c', '-m']
    losses = ["eqs", "norm", "ortho"]
    for i, lossT in enumerate(loss.transpose(0,1)):
      plt.loglog(lossT.cpu(), color_cycle[i], label=losses[i], alpha=0.975)
    plt.loglog(torch.sum(loss.cpu(),-1), '-b', label='total', alpha=0.975)
    #plt.loglog(loss,'-b',alpha=0.975);                
    plt.tight_layout()
    plt.ylabel('Loss');plt.xlabel('t')
    plt.legend()
    plt.title(self.curr_model_name)

    loss_file = "loss/" + str(self.curr_model_name) + '.png'
    plt.savefig(loss_file)
    plt.close()
    np.savetxt("loss/" + str(self.curr_model_name) + ".txt", loss.cpu())

