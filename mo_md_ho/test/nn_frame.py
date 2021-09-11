## start
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import os
from os import path
import sys
import gc
# Path


def boundEnv(vecT, c, bounds):
    boundL, boundR = bounds
    ans = (1-exp(-vecT+boundL)).prod(-1,keepdim=True) * (1-exp(vecT-boundR)).prod(-1,keepdim=True)
    return ans

def unboundEnv(vecT, c, bounds):
    return 1/(((vecT-c)**2+1).prod(-1,keepdim=True))

def tanhEnv(vecT, c, bounds):
    return (torch.tanh(0.6+c+vecT)+torch.tanh(0.6-c-vecT)).prod(-1,keepdim=True)

def freeEnv(vecT, c, bounds):
    return 1

dir = "2D_Box_Multi"
if not path.exists(dir): os.mkdir(dir)
nStates = 3

# Training parameters
epochs = int(10)
n_train = 5
lr = 1e-2

adam = lambda p : optim.Adam(p, lr = lr, betas = [0.999, 0.9999])
sgd = lambda p : optim.SGD(p, lr = lr, momentum = 0.9)
# Parameters
params = {}
params['nDim'] = 2
params['nParams'] = nStates
params['neurons'] = [2,80,80,nStates]
params['bounds'] = [0.,0.], [1.,1.]  # left bounds, right bounds
params['envelope'] = boundEnv
params['mesh'] = True
params['unbundle'] = []
params['bundle'] = [], []  # bundle mins, bundle maxes
params['optimizer'] = adam
params['c'] = 0

# Calculate the derivatice with auto-differention
#ones = torch.ones((n_train,1), dtype=torch.float)
def dfx(f, x):
    print(f"dfx: {x.shape}")
    return grad(f, x, grad_outputs=torch.ones(f.shape, dtype=torch.float), create_graph=True)[0]

# Loss Function
def loss_fn(psi, vec, Area, bounds, unbundle, bundle, training, epoch):
    x, y = vec

    psiT = psi.transpose(0,-1).transpose(1,2)
    px = [dfx(psiT[i], x) for i in range(nStates)]
    py = [dfx(psiT[i], y) for i in range(nStates)]
    ppx = [dfx(px[i], x) for i in range(nStates)]
    ppy = [dfx(py[i], y) for i in range(nStates)]
    f = 0
    weight = 0
    normal = 0
    for i in range(nStates):
        f += (ppx[i] + ppy[i] + (np.pi**2/Area*training[i])*psiT[i]).pow(2).mean()/(psiT[i]**2).mean()
        weight += training[i]**2
        #normal += (1 - torch.trapz(torch.trapz(psiT[i].pow(2), x, dim = 0), y[0], dim = 0)).pow(2)
    
    ortho = 0
    for s1 in range(nStates):
        for s2 in range(s1+1,nStates):
            ortho += torch.trapz(torch.trapz(psiT[s1]*psiT[s2], x, dim = 0), y[0], dim = 0).pow(2)
    return f, 10*ortho, 1e4*weight/(epoch**2+1) #,5*normal
# Check and use gpu if available
if torch.cuda.is_available():
    device = torch.device('cuda'); torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device('cpu'); torch.set_default_tensor_type(torch.FloatTensor)

# Python optimization
sin = torch.sin
exp = torch.exp
cat = torch.cat

"""# Network Architecture

Define some helper functions
"""

# sin() activation function
class mySin(nn.Module):
    @staticmethod
    def forward(input):
        return sin(input)

"""Network architecture"""

class odeNet(nn.Module):
    def __init__(self, nDim, neurons, nParams):
        super(odeNet,self).__init__()

        self.actF = mySin()        
        # layers
        self.layers = nn.ModuleList()
        for i in range(len(neurons)-2):
            self.layers.append(nn.Linear(neurons[i], neurons[i+1]))
        self.params = nn.ParameterList([nn.Parameter(torch.rand(1, requires_grad=True)*nParams) for i in range(nParams)])
        self.Lin_out = nn.Linear(neurons[-2], neurons[-1])
    
    def forward(self, vec):
        for layer in self.layers:
            vec = self.actF(layer(vec))
        return self.Lin_out(vec)

def run_odeNet(loss_fn, epochs, n_train, params, loadWeights, minLoss, PATH):
    # minor optimization
    rand = torch.rand
    randn = torch.randn
    stack = torch.stack
    unbind = torch.unbind
    
    nDim = params['nDim']
    nParams = params['nParams']
    neurons = params['neurons']
    env = params['envelope']
    optimizer = params['optimizer']

    mesh = params['mesh']
    unbundle = params['unbundle']
    boundL, boundR = params['bounds']
    bundleL, bundleR = params['bundle']
    boundL = torch.tensor(boundL).reshape(-1,1)
    boundR = torch.tensor(boundR).reshape(-1,1)
    bounds = boundL.squeeze(), boundR.squeeze()
    bundleL = torch.tensor(bundleL)
    bundleR = torch.tensor(bundleR)
    c = torch.tensor(params['c'])
    
    if mesh:
        def vec_gen():
            vec = torch.meshgrid(unbind(torch.rand(nDim, n_train, requires_grad = True).sort(dim=-1)[0] * (boundR - boundL) + boundL))
            return vec
    else:
        def vec_gen():
            vec = unbind(rand(nDim, n_train, requires_grad = True) * (boundR - boundL) + boundL)
            return vec

    Area = (boundR-boundL).prod().item()
    bundles = rand(epochs, len(bundleL)) * (bundleR - bundleL) + bundleL
    
    fc0 = odeNet(nDim + len(bundleL), neurons, nParams)
    fc1 = deepcopy(fc0) # network with lowest loss

    optimizer = optimizer(fc0.parameters())
    Loss_history = []; Llim = 1
    t = 0
    
    # load existing weights
    if path.exists(PATH) and loadWeights == True:
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        t = checkpoint['epoch']
        loss = checkpoint['loss']
        fc0.train()
    
    TeP0 = time.time()
    training = fc0.params
    if mesh:
        reps = [n_train]*nDim+[1]
    else:
        reps = n_train,1
    
    for tt in range(t, t + epochs):
        vec = vec_gen()
        vecT = torch.stack(vec,-1) # vectors stacked
        
        bundle = bundles[tt-t]
        # network solutions
        #psi = env(vecT, c, bounds) * fc0(cat((vecT,bundle.repeat(reps)),-1))
        a = env(vecT, c, bounds) 
        b=fc0(cat((vecT,bundle.repeat(reps)),-1))
        print(f"av: {(a*b).mean()}, a: {a.mean()}, b: {b.mean()}")
        psi = a*b
        # loss
        loss = loss_fn(psi, vec, Area, c, unbundle, bundle, training, tt)
        Ltot = sum(loss)
        # optimization
        Ltot.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()

        # loss history
        detached = [L.item() for L in loss]
        Loss_history.append(detached)
        if tt%100 == 0:
            print(f'\rLoss: {detached}, Step: {tt}', end='')
            gc.collect()
        # best model
        if  tt-t > 0.8*epochs and Ltot < Llim:
            fc1 = deepcopy(fc0)
            Llim = Ltot.item()
        # thresold accuracy
        if Ltot < minLoss:
            fc1 = deepcopy(fc0)
            print('\nReach minimum requested loss',end='')
            break
    print()
    TePf = time.time()
    runTime = TePf - TeP0
    
    torch.save({
    'epoch': tt,
    'model_state_dict': fc1.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, "TEST1")

    return fc1, Loss_history, runTime, Llim, tt

def trainModel(loss_fn, epochs, n_train, params, loadWeights, minLoss, PATH):
    model, loss, runTime, Llim, steps = run_odeNet(loss_fn, epochs, n_train, params, loadWeights, minLoss, 'models/'+PATH)

    np.savetxt(PATH + '/loss.txt', loss)
    
    print('Training time (minutes):', runTime/60)
    print('Training Loss: ', loss[-1])
    print('Lowest Loss: ', Llim)
    print('Steps: ', steps+1)

    with open(PATH+'/profile.txt','w') as f:
        f.write(f'Final Loss: {loss}\n')
        f.write('Envelope: '+params['envelope'].__name__+'\n')
        f.write('Optimizer: '+params['optimizer'].__name__+'\n')
        f.write('Learning Rate: '+str(params['optimizer'](model.parameters()).param_groups[0]['lr'])+'\n')
        f.write(f'Epochs: {steps}\n')
        f.write(f'Runtime: {Llim}\n')
        f.write(f'Lowest Loss: {Llim}\n')
        f.write(f'Final Loss: {loss}\n')
        if len(model.params) > 0:
            f.write('Parameters: '+str([param[0].item() for param in model.params]))

def loadModel(PATH, params):
    if path.exists(PATH):
        fc0 = odeNet(params['nDim']+len(params['bundle'][0]), params['neurons'], params['nParams'])
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        fc0.train()
    else:
        print('Warning: There is not any trained model. Terminate')
        sys.exit()
    return fc0

"""# Analysis"""

def test_points(model, params, N):
    nDim = params['nDim']
    nParams = params['nParams']
    env = params['envelope']
    mesh = params['mesh']
    boundL, boundR = params['bounds']
    bundleL, bundleR = params['bundle']
    bounds = torch.tensor(boundL).squeeze(), torch.tensor(boundR).squeeze()
    bundleL = torch.tensor(bundleL)
    bundleR = torch.tensor(bundleR)
    c = torch.tensor(params['c'])

    vec = torch.meshgrid([torch.linspace(boundL[i], boundR[i], N) for i in range(nDim)])
    vecT = torch.stack(vec,-1)
    psi = env(vec, c, bounds) * model(vecT)

    return vec, psi.detach().numpy()

"""Plots"""

def plot_full(params, vec, psiTest, end=''):
    dim = params['nDim']
    if dim == 1:
        x_net = vec
        lineW = 4
        plt.figure(figsize=(10,8))
        plt.plot(x_net, psiTest, '--b', label='x', linewidth=lineW, alpha=.5)
        plt.ylabel('x(t)'); plt.xlabel('t')
        plt.legend()

    elif dim == 2:
        from mpl_toolkits.mplot3d import Axes3D

        x_net, y_net = vec
        lineW = 4 # Line thickness
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x_net, y_net, psiTest, label='x', alpha=.2)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('psi')
        plt.legend()

    if dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        
        x, y, z = vec
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        img = ax.scatter(x, y, z, c=psiTest, cmap=plt.hot())
        #img = ax.scatter(x[indices], y[indices], z[indices], c=psiTest[indices], cmap=plt.hot())
        fig.colorbar(img)
        plt.show()

    plt.savefig(f'{dir}/plots{end}.png')

def plot_color(params, vec, psiTest, end=''):
    dim = params['nDim']
    if dim == 2:
        plt.figure(figsize=(10,8))
        plt.imshow(psiTest.reshape(-1,N))

    if dim == 3:
        x, y, z = vec
        # , cmap=plt.get_cmap('binary')
        plt.figure(figsize=(10,8))
        plt.subplot(2,2,1)
        plt.scatter(x,y,c=psiTest)
        plt.xlabel('x');plt.ylabel('y')

        plt.subplot(2,2,2)
        plt.scatter(y,z,c=psiTest)
        plt.xlabel('y');plt.ylabel('z')

        plt.subplot(2,2,3)
        plt.scatter(z,x,c=psiTest)
        plt.xlabel('z');plt.ylabel('x')

    plt.savefig(f'{dir}/plots_color{end}.png')

def plot_bundle(dim, model, params):
    if dim == 1 and len(params['bundle'][0]) == 1:
        fig = go.Figure()
        inter = np.arange(0, 5, 0.25)
        for step in inter:
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color="#00CED1", width=6),
                    name="𝜈 = " + str(step),
                    x = t_net.reshape(-1),
                    y = parametricSolutions(model, tTest, torch.Tensor([step]).expand(N).reshape(-1,1), c).reshape(-1).cpu().detach().numpy()))

        # Make 10th trace visible
        fig.data[0].visible = True

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                    {"title": "Lambda set to: " + str(inter[i])}],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Lambda: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders
        )

        fig.show()

trainModel(loss_fn, epochs, n_train, params, loadWeights=False, minLoss=1e-3, PATH=dir)

## Graphing
model = loadModel("TEST", params)
vec, psiTest = test_points(model, params, 100)
plot_full(params, vec, psiTest)


