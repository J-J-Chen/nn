import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
import copy
from os import path
import sys

MODEL_PATH="~/nn/"
FIGURE_PATH="~/nn/"

dtype = torch.float

# Calculate the derivatice with auto-differention
def dfx(x,f):
    return grad([f], [x], grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]

def perturbPoints(grid,t0,tf,sig=0.5):
#   stochastic perturbation of the evaluation points
#   force t[0]=t0  & force points to be in the t-interval
    delta_t = grid[1] - grid[0]  
    noise = delta_t * torch.randn_like(grid)*sig
    t = grid + noise
    t.data[2] = torch.ones(1,1)*(-1)
    t.data[t<t0]=t0 - t.data[t<t0]
    t.data[t>tf]=2*tf - t.data[t>tf]
    # t.data[0] = torch.ones(1,1)*t0
    t.requires_grad = False
    return t

def parametricSolutions(t, nn, X0):
    # parametric solutions
    L0, L1 = X0[1]*torch.ones(1,1),X0[2]*torch.ones(1,1)
    fL, fr = X0[7]*torch.ones(1,1),X0[8]*torch.ones(1,1)
    #N1  = nn(t)
    #dt =t-t0
#### THERE ARE TWO PARAMETRIC SOLUTIONS. Uncomment f=dt 
    #f = (1-torch.exp(-dt))
    #f=dt
    #x_hat  = x0  + N1 * f**2 + v0 * f
    #x_hat = 0 + f*N1
    #print("L0:", L0, "L1:", L1, "fL:", fL, "fr", fr)

    norm = (1-torch.exp(L0-L1))
    gL = (1-torch.exp((t-L1)))/norm
    gR = (1-torch.exp(-(t-L0)))/norm
    N1 = nn(t)
    f = fL*gL + fr*gR + gL*gR*N1
    return f

numTimes = 0
def Eqs_Loss(t,x1, X0):
    global numTimes
    # Define the loss function by  Eqs.
    xdot = dfx(t,x1)
    x2dot = dfx(t,xdot)
    E = X0[6]
    f1 = x2dot - t.pow(2)*x1 + 2*E*x1
    L  = (f1.pow(2)).mean() + 1/(x1.pow(2).mean()+1e-4)
    return L


# A two hidden layer NN, 1 input & 1 output
class odeNet(torch.nn.Module):
    def __init__(self, D_hid=10):
        super(odeNet,self).__init__()

        # Define the Activation
        self.actF = torch.nn.Sigmoid()   
#         self.actF = mySin()
        
        # define layers
        self.Lin_1   = torch.nn.Linear(1, D_hid)
        self.Lin_2   = torch.nn.Linear(D_hid, D_hid)
#        self.Lin_3   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_out = torch.nn.Linear(D_hid, 1)

    def forward(self,t):
        # layer 1
        l = self.Lin_1(t);    h = self.actF(l)
        # layer 2
        l = self.Lin_2(h);    h = self.actF(l)
        # layer 3
#        l = self.Lin_2(h);    h = self.actF(l)
        # output layer
        netOut = self.Lin_out(h)
        return netOut



# Train the NN
def run_odeNet(X0, tf, neurons, epochs, n_train,lr, PATH= "", loadWeights=False,
                    minibatch_number = 1, minLoss=1e-3):
                    
    PATH = MODEL_PATH + PATH
    fc0 = odeNet(neurons)
    fc1 =  copy.deepcopy(fc0) # fc1 is a deepcopy of the network with the lowest training loss
    # optimizer
    betas = [0.999, 0.9999]
    
    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=betas)
    Loss_history = [];     Llim =  1 
        
    t0=X0[0];
    grid = torch.linspace(t0, tf, n_train).reshape(-1,1)

    
    
        
## LOADING WEIGHTS PART if PATH file exists and loadWeights=True
    if path.exists(PATH) and loadWeights==True:
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        tt = checkpoint['epoch']
        Ltot = checkpoint['loss']
        fc0.train(); # or model.eval
    
    
## TRAINING ITERATION    
    TeP0 = time.time()
    using_gpu = torch.cuda.is_available()
    if using_gpu:
        scaler = torch.cuda.amp.GradScaler()
    for tt in range(epochs):                
# Perturbing the evaluation points & forcing t[0]=t0
        # t=perturbPoints(grid,t0,tf,sig=.03*tf)
        t=perturbPoints(grid,t0,tf,sig= 0.3*tf)
            
# BATCHING
        batch_size = int(n_train/minibatch_number)
        batch_start, batch_end = 0, batch_size

        idx = np.random.permutation(n_train)
        t_b = t[idx]
        t_b.requires_grad = True

        loss=0.0
        for nbatch in range(minibatch_number): 
# batch time set
            t_mb = t_b[batch_start:batch_end]
#  Network solutions 
            x = parametricSolutions(t_mb,fc0,X0)
# LOSS
#  Loss function defined by Hamilton Eqs. (symplectic): Writing explicitely the Eqs (faster)
            if using_gpu:
                with torch.cuda.amp.autocast():
                    Ltot = Eqs_Loss(t_mb,x, X0)
            else:
                Ltot = Eqs_Loss(t_mb,x, X0)
            

#  Loss function defined by Hamilton Eqs. (symplectic): Calculating with auto-diff the Eqs (slower)
#             Ltot = hamEqs_Loss_byH(t_mb,x,y,px,py,lam)
    

# OPTIMIZER
            if using_gpu:
                scaler.scale(Ltot).backward(retain_graph=False)
                scaler.step(optimizer)
                scaler.update()
            else:
                Ltot.backward(retain_graph=False); #True
                optimizer.step(); 
            loss += Ltot.detach()
       
            optimizer.zero_grad()

            batch_start +=batch_size
            batch_end +=batch_size

# keep the loss function history
        Loss_history.append(loss)       

#Keep the best model (lowest loss) by using a deep copy
        if  tt > 0.8*epochs  and Ltot < Llim:
            fc1 =  copy.deepcopy(fc0)
            Llim=Ltot 

# break the training after a thresold of accuracy
        if Ltot < minLoss :
            fc1 =  copy.deepcopy(fc0)
            print('Reach minimum requested loss')
            break



    TePf = time.time()
    runTime = TePf - TeP0     
    
    
    torch.save({
    'epoch': tt,
    'model_state_dict': fc1.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': Ltot,
    }, PATH)

    return fc1, Loss_history, runTime



def trainModel(X0, t_max, neurons, epochs, n_train, lr,  loadWeights=True, minLoss=1e-6, showLoss=True, PATH =""):
    model,loss,runTime = run_odeNet(X0, t_max, neurons, epochs, n_train,lr, PATH,  loadWeights=loadWeights, minLoss=minLoss, minibatch_number=1)

    np.savetxt('loss.txt',loss)
    
    if showLoss==True :
        print('Training time (minutes):', runTime/60)
        print('Training Loss: ',  loss[-1] )
        plt.figure()
        plt.loglog(loss,'-b',alpha=0.975);                
        plt.tight_layout()
        plt.ylabel('Loss');plt.xlabel('t')
    
        # plt.savefig('HHsystem/HenonHeiles_loss.png')
        plt.savefig('simple_expDE_loss.png')
    

def loadModel(PATH):
    PATH = MODEL_PATH + PATH
    if path.exists(PATH):
        fc0 = odeNet(neurons)
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        fc0.train(); # or model.eval
    else:
        print('Warning: There is not any trained model. Terminate')
        sys.exit()

    return fc0    




# TRAIN THE NETWORK. 
# Set the time range and the training points N
t0, t_max, N = -5,  5, 3000;     
# Set the initial state. lam controls the nonlinearity
L0=-5
L1=5
k = 1
m = 1
E = 0.5 #*hbar*((k/m)**0.5)
hbar=1
fL = 0
fR = 0
X0 = [t0, L0,L1,k,m,hbar,E,fL,fR]

# Here, we use one mini-batch. NO significant different in using more
n_train, neurons, epochs, lr = N, 80, int( 1e2 ), 8e-3
trainModel(X0, t_max, neurons, epochs, n_train, lr, PATH="mod_bound_median",  loadWeights=False, minLoss=1e-6, showLoss=True)
model = loadModel("mod_bound_median")

nTest = N ; t_max_test = 1.0*t_max
tTest = torch.linspace(t0,t_max_test,nTest)

tTest = tTest.reshape(-1,1);
tTest.requires_grad=True
t_net = tTest.detach().cpu().numpy()


xTest=parametricSolutions(tTest,model,X0)
xdotTest=dfx(tTest,xTest)

xTest=xTest.cpu().data.numpy()
xdotTest=xdotTest.cpu().data.numpy()


################
# Median
#################

lineW = 4 # Line thickness
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.plot(t_net, exact_wf(t_net,0)[0],'-g', label='Ground Truth', linewidth=lineW);
plt.plot(t_net, xTest,'--b', label='x',linewidth=lineW, alpha=.5); 
plt.ylabel('x(t)');plt.xlabel('t')
plt.legend()


plt.subplot(2,2,2)
#plt.plot(t_net, xdot_exact,'-g', label='Ground Truth', linewidth=lineW);
plt.plot(t_net, xdotTest,'--b', label='x',linewidth=lineW, alpha=.5); 
plt.ylabel('dx\dt');plt.xlabel('t')
plt.legend()



# #plt.savefig('../results/HenonHeiles_trajectories.png')
plt.savefig(FIGURE_PATH+'simpleExp.png')
