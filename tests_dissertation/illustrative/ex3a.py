# -*- coding: utf-8 -*-

# =============================================================================
# In this example, a 1d mixture of Gaussians is created, 
# and training is made with active learning
# =============================================================================
import sys
sys.path.insert(0,"../../src")
import os
import math
import functools
import time

import torch
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.variational_boosting_bmc import VariationalBoosting
from src import vb_utils
from src import sampling
from src.utils import TicToc #TicToc is an auxiliary class for time tracking

#Here, the seeds are set to ensure reproduciblity
np.random.seed(100)
torch.manual_seed(100)

tictoc = TicToc()

#%% Needed auxiliary functions
def logexpsum(expoents,weights):
    #Expoents : (n,d). Weights : (1,d)
    max_expoent = torch.max(expoents,dim=1,keepdim=True)[0]
    expoents_star = expoents - max_expoent
    sumexp_star = torch.sum(weights*torch.exp(expoents_star),dim=1,keepdim=True)
    return max_expoent + torch.log(sumexp_star)

def logjointmix(x,logps,alpha):
    if x.dim() == 1:
        x = x.reshape(1,-1)
    ys = torch.cat([logp(x) for logp in logps],dim=1)
    return logexpsum(ys,alpha)

def loggaussian(x,mu,sigma2):
    return -0.5*torch.sum((x-mu)**2/sigma2,dim=1,keepdim=True) + \
           -0.5*torch.sum(torch.log(sigma2)) + \
           -0.5*math.log(2*math.pi)

def tndmeancov(mu_list,sigma2_list,weights):
    cov2_list = torch.stack([torch.diag(sigma2_list[i]) for i,_ in enumerate(sigma2)],dim=0)
    mu = (weights.reshape(-1,1)*mu_list).sum(dim=0)
    moment2_ = cov2_list + torch.bmm(mu_list.unsqueeze(2),mu_list.unsqueeze(1))
    moment2 = (moment2_*weights.reshape(-1,1,1)).sum(dim=0)
    cov = moment2 - torch.ger(mu,mu)
    return mu,cov

def samplergaussian(mu,sigma2,weights,nsamples):
    inds = torch.multinomial(weights,nsamples,replacement=True)
    mu_ = mu[inds,:]
    sigma2_ = sigma2[inds,:]
    samples = torch.distributions.Normal(mu_,torch.sqrt(sigma2_)).rsample()
    return samples

#%% Creating the distribution
dim = 1
num_mixtures = 12
alpha = 1/num_mixtures*torch.ones(num_mixtures)
mu = torch.randn(num_mixtures,dim)*5
sigma2 = torch.ones(num_mixtures,dim)
logps = [functools.partial(loggaussian,mu=mu[i],sigma2=sigma2[i]) 
         for i in range(num_mixtures)]
logjoint_ = functools.partial(logjointmix,logps=logps,alpha=alpha)
def logjoint(sample):
    return logjoint_(sample)
true_mean,true_cov = tndmeancov(mu,sigma2,alpha)
sampler = lambda nsamples : samplergaussian(mu,sigma2,alpha,nsamples)
#%%
nsamples = 5
ninducing = 50
samples=sampling.sampling1(nsamples,dim,scale=10.0)
mu0 = torch.zeros(dim)
cov0 = (20.0/3)**2*torch.ones(dim)
acquisitions = ["uncertainty_sampling","prospective","mmlt","mmlt_prospective"]
for acquisition in acquisitions:
    folder_name = "ex3a_acq_%s_data"%acquisition
    try:
        os.mkdir(folder_name)
    except:
        pass
    #%% The main training class is initiated, it's gp model optimized,
    #   and initial component set
    vb = VariationalBoosting(dim,logjoint,samples,mu0,cov0,
                             kernel_function="PMat",
                             matern_coef=2.5,
                             degree_hermite=60)
    vb.optimize_bmc_model(maxiter=100,verbose=False)
    #%% Tracking devices
    vb.save_distribution("%s/distrib%i"%(folder_name,0))
    nplot = 201
    delta_x = torch.linspace(-20,20,nplot)
    delta_x_np = delta_x.flatten().numpy()
    tp_np = (logjoint(delta_x.reshape(-1,1)).cpu()).flatten().numpy()
    dmeans = [torch.norm(vb.currentq_mean.to("cpu")-true_mean).numpy()]
    dcovs = [torch.norm(vb.currentq_mean.to("cpu")-true_cov,2).numpy()]
    elbo_list = [vb.evidence_lower_bound(nsamples=10000).cpu().numpy()]
    step_list = [0]
    time_list = [0]
    vbp_list = []
    vbp = vb.current_logq(delta_x.reshape(-1,1)).cpu().flatten().numpy().astype(float)
    vbp_list.append(vbp)
    #%% Main loop
    for i in range(50):
        tictoc.tic()
        _ = vb.update(maxiter_nc=300,lr_nc=0.01,
                      n_samples_nc=500,n_samples_sn=300,n_iter_sn=300)
        try:
            vb.update_bmcmodel(acquisition=acquisition,vreg=1e-3)
        except:
            continue
        if ((i+1)%10) == 0:
            vb.update_full()
        #%% Save trackings
        elapsed = tictoc.toc(printing=False)
        dmeans.append(np.linalg.norm(vb.currentq_mean.cpu()-true_mean,2))
        dcovs.append(np.linalg.norm(vb.currentq_cov.cpu()-true_cov,2))
        elbo_list.append(vb.evidence_lower_bound(nsamples=10000).cpu().numpy())
        step_list.append(i+1)
        time_list.append(elapsed)
        vbp = vb.current_logq(delta_x.reshape(-1,1)).cpu().flatten().numpy().astype(float)
        vbp_list.append(vbp)
        vb.save_distribution("%s/distrib%i"%(folder_name,i+1))
        print(vb_utils.kl_vb_bmc(vb,1000))
        print(dmeans[-1])
        print(dcovs[-1])
        print(elbo_list[-1])
        print(time_list[-1])
        print("Step %i"%(i+1))
    #%% Final tracking
    prediction_np = (vb.bmcmodel.prediction(delta_x.reshape(-1,1),cov="none")*vb.evals_std + vb.evals_mean).\
                    numpy().astype(float)        
    dmeans_np = np.array(dmeans).astype(float)
    dcovs_np = np.array(dcovs).astype(float)
    elbo_np = np.array(elbo_list).astype(float)
    step_list_np = np.array(step_list).astype(int)
    vbp_np = np.array(vbp_list)
    time_np = np.array(time_list)
    samples_np = vb.samples.numpy()
    evaluations_np = vb.evaluations.numpy()
    np.savez("%s/tracking"%folder_name,means=dmeans_np,covs=dcovs_np,
             time=time_np,
             elbo=elbo_np,steps=step_list_np,
             true_posterior=tp_np,bmc_pred=prediction_np,
             vb_posterior=vbp_np,xplot=delta_x_np,
             samples=samples_np,evaluations=evaluations_np)