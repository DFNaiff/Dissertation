# -*- coding: utf-8 -*-

#Mixture of gaussians
#In this examples, the 1d case will be shown

# -*- coding: utf-8 -*-

# =============================================================================
# Here we will be testing with mixtures of student-t, 1d
# =============================================================================
import sys
sys.path.insert(0,"../../src")
import math
import functools
import time
import os

import torch
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.variational_boosting_bmc import VariationalBoosting
from src import vb_utils
from src import sampling
from src.utils import TicToc
from source_2d_likelihood_fn import compute_log_likelihood

np.random.seed(100)
torch.manual_seed(100)
tictoc = TicToc()
#%% priors
#x0 : Unif(0,1)
#Ts : Unif(0,0.5)
lambd_rho = 20.0 #rho : Exp(20)
lambd_q = 0.2 #q : Exp(0.2)
#%%%
def logit_t(x,a=0,b=1):
    return torch.log(((x-a)/(b-a))/(1.0-(x-a)/(b-a)))
def sigmoid(x,a=0,b=1):
    return (b-a)*1.0/(1.0+np.exp(-x)) + a
def dsigmoid(x,a=0,b=1):
    return (b-a)*np.exp(x)/((1+np.exp(x))**2)
def exp(x):
    return np.exp(x)
def dexp(x):
    return np.exp(x)

def unwarped_logjoint_np(x0,y0,Ts,q0,rho):
    rho = 0.05
    Ts = 0.3
    q0 = 6.366197723675814
    ll = compute_log_likelihood(x0,y0,rho,q0,Ts)
#    ll += -np.log(1+(q0/10.0)**2)
#    ll += -np.log(1+(rho/0.1)**2)
    return ll

def logjoint_np(x):
#    x0,y0,Ts,q0,rho = x[0],x[1],x[2],x[3],x[4]
    x0,y0,Ts,q0,rho = x[0],x[1],0.0,0.0,0.0
    ll = unwarped_logjoint_np(sigmoid(x0),sigmoid(y0),
                              sigmoid(Ts,b=0.4),
                              exp(q0),exp(rho)) + \
         np.log(dsigmoid(x0)) + np.log(dsigmoid(y0))
#         np.log(dsigmoid(Ts,b=0.4)) + \
#         np.log(dexp(q0)) + np.log(dexp(rho))
    return ll

def logjoint(sample):
    return torch.tensor(logjoint_np(sample.flatten().numpy()))
#%%

#torch.randn(20)
dim = 2
nsamples = 20*dim
def sampling1(nsamples):
    X1 = logit_t(torch.rand(nsamples,1))
    X2 = logit_t(torch.rand(nsamples,1))
#    X3 = logit_t(torch.rand(nsamples,1)*0.4,b=0.4)
#    X4 = torch.log(torch.distributions.HalfCauchy(scale=10.0).rsample((nsamples,1)))
#    X5 = torch.log(torch.distributions.HalfCauchy(scale=0.1).rsample((nsamples,1)))
    sample = torch.cat([X1,X2],dim=1)
    return sample
samples = sampling1(nsamples)
mu0 = torch.zeros(dim)
cov0 = 20.0*torch.ones(dim)

#%%
#samples = vb.samples
training_interval = 20
acquisitions = ["prospective","mmlt"]
vb = VariationalBoosting(dim,logjoint,samples,mu0,cov0,
                         bmc_type="FM",normalization_mode="normalize",
                         training_space="gspace",noise=1e-4,
                         kernel_function="PMat",matern_coef=1.5,
                         numstab=-50.0,degree_hermite=50)
vb.optimize_bmc_model(maxiter=500,verbose=1,
                      lr=0.05)    

#%%
elbo_list = [vb.evidence_lower_bound(nsamples=10000).cpu().numpy()]
kl_list = [vb.kullback_proposal_bmc(10000).item()]
step_list = [0]
time_list = [0.0]
#%%
print("Active sampling...")
for i in range(10*dim):
    vb.update_bmcmodel(acquisition="mmlt",mode="optimizing",vreg=1e-2)
vb.update_full()
folder_name = "testheat1c"
try:
    os.mkdir(folder_name)
except FileExistsError:
    pass
vb.save_distribution("%s/mvn%i"%(folder_name,0))
#%%
print(torch.sigmoid(vb.currentq_mean))
print(elbo_list[-1])
print(kl_list[-1])
#%%
for i in range(100):
    tictoc.tic()
    _ = vb.update(maxiter_nc=300,lr_nc=0.1,b_sn=0.1,
                  n_samples_nc=500,n_samples_sn=300,n_iter_sn=300,
                  max_alpha=1.0,verbose=0)
    try:
        acquisition = np.random.choice(acquisitions)
        vb.update_bmcmodel(acquisition=acquisition,mode="optimizing",
                           acq_reg=1e-1,verbose=0)
    except:
        continue
    if ((i+1)%training_interval) == 0:
        vb.update_full()
    elapsed = tictoc.toc(printing=False)
    elbo_list.append(vb.evidence_lower_bound(nsamples=10000).cpu().numpy())
    kl_list.append(vb.kullback_proposal_bmc(10000).item())
    step_list.append(i+1)
    time_list.append(elapsed)
    print(torch.sigmoid(vb.currentq_mean))
    print(acquisition)
    print(elbo_list[-1])
    print(kl_list[-1])
    print(time_list[-1])
    print("Step %i"%(i+1))
    vb.save_distribution("%s/mvn%i"%(folder_name,i+1))

elbo_np = np.array(elbo_list).astype(float)
step_list_np = np.array(step_list).astype(int)
times_np = np.array(time_list)
np.savez("%s/tracking"%folder_name,
         time=times_np,
         elbo=elbo_np,
         steps=step_list_np)
#%%
Nplot = 41
x,y = torch.linspace(-10,10,Nplot),torch.linspace(-10,10,Nplot)
X,Y = torch.meshgrid(x,y)
XY_ = torch.stack([X,Y],dim=-1).reshape(-1,2)
Zgp_ = vb.scaled_prediction(XY_)
Zgp = Zgp_.reshape(*X.shape)
Zq_ = vb.current_logq(XY_)
Zq = Zq_.reshape(*X.shape)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X.numpy(),Y.numpy(),Zgp.numpy())
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X.numpy(),Y.numpy(),Zq.numpy())
plt.show()
#%%
Nplot = 41
x = torch.linspace(-5,5,Nplot).reshape(-1,1)
xy_ = torch.cat([x,1.0986122886681098*torch.ones_like(x)],dim=-1)
zgp = vb.scaled_prediction(xy_)
zvb = vb.current_logq(xy_)
fig,ax = plt.subplots()
plt.plot(x.numpy(),zgp.numpy())
fig,ax = plt.subplots()
plt.plot(x.numpy(),zvb.detach().numpy())
#%%
Nplot = 21
x,y = torch.linspace(-10,10,Nplot),torch.linspace(-10,10,Nplot)
X,Y = torch.meshgrid(x,y)
XY_ = torch.stack([X,Y],dim=-1).reshape(-1,2)
Ztrue_ = np.array([logjoint_np(xy) for xy in XY_.numpy()]).reshape(-1,1)
Ztrue = Ztrue_.reshape(*X.shape)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(X.numpy(),Y.numpy(),Ztrue)
plt.show()
#Zgp_ = vb.scaled_prediction(XY_)
#Zgp = Zgp_.reshape(*X.shape)
#Zq_ = vb.current_logq(XY_)
#Zq = Zq_.reshape(*X.shape)
#
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111, projection='3d')
#ax1.plot_surface(X.numpy(),Y.numpy(),Zgp.numpy())
#plt.show()
#
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111, projection='3d')
#ax2.plot_surface(X.numpy(),Y.numpy(),Zq.numpy())
#plt.show()
