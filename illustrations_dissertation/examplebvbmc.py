#Import necessary packages
import torch
from src.variational_boosting_bmc import VariationalBoosting

torch.manual_seed(100) #For reproducibility

#Approximating nnormalized 2-d Cauchy
def logjoint(theta):
    return torch.sum(-torch.log(1+theta**2),dim=-1)

#Set up parameters
dim=2 #Dimension of problem
samples = torch.randn(20,dim) #Initial samples
mu0 = torch.zeros(dim) #Initial mean
cov0 = 20.0*torch.ones(dim) #Initial covariance
acquisition = "prospective" #Acquisition function

#Initialize algorithm
vb = VariationalBoosting(dim,logjoint,samples,mu0,cov0)
vb.optimize_bmc_model() #Optimize GP model
vb.update_full() #Fit first component

#Training loop
for i in range(100):
    _ = vb.update() #Choose new boosting component
    vb.update_bmcmodel(acquisition=acquisition) #Choose new evaluation
    vb.cutweights(1e-3) #Weights prunning
    if ((i+1)%20) == 0:
        vb.update_full(cutoff=1e-3) #Joint parameter updating

vb.save_distribution("finaldistrib") #Save distribution
#%%
import math
distrib = torch.load("finaldistrib")
nplot = 21
x,y = torch.linspace(-6,6,nplot),torch.linspace(-6,6,nplot)
X,Y = torch.meshgrid(x,y)
Z1 = logjoint(torch.stack([X,Y],dim=-1).reshape(-1,2)).reshape(*X.shape)-\
        2*math.log(math.pi)
Z2 = distrib.logprob(torch.stack([X,Y],dim=-1).reshape(-1,2)).reshape(*X.shape)
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
fig1 = plt.figure()
ax1 = fig1.add_subplot(111,projection="3d")
ax1.plot_wireframe(X.numpy(),Y.numpy(),torch.exp(Z1).numpy())
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.set_zlabel("$g(x)$")
ax1.set_zlim(bottom=0.0,top=0.125)
fig1.savefig("/home/danilo/Danilo/Dissertação/Tex (copy 1)/figs/examplebvbmctrue",
             bbox_inches = "tight")
fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection="3d")
ax2.plot_wireframe(X.numpy(),Y.numpy(),torch.exp(Z2).numpy(),color="red")
ax2.set_xlabel("$x$")
ax2.set_ylabel("$y$")
ax2.set_zlabel("$q(x)$")
ax2.set_zlim(bottom=0.0,top=0.125)
fig2.savefig("/home/danilo/Danilo/Dissertação/Tex (copy 1)/figs/examplebvbmcestimated",
             bbox_inches = "tight")






