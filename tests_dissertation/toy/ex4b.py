# -*- coding: utf-8 -*-

# =============================================================================
# Here we will be testing anisotropic Gaussian
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
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.variational_boosting_bmc import VariationalBoosting
from src import vb_utils
from src import sampling
from src.utils import TicToc

np.random.seed(100)
torch.manual_seed(100)
tictoc = TicToc()
#%%
def maximum_mean_discrepancy(sx1,sx2,sx3,sy1,sy2,sy3,l=10.0,theta=100):
    e1 = torch.exp(-0.5*((sx1-sx2)/l).pow(2).sum(dim=1)).mean()
    e2 = torch.exp(-0.5*((sy1-sy2)/l).pow(2).sum(dim=1)).mean()
    e3 = torch.exp(-0.5*((sx3-sy3)/l).pow(2).sum(dim=1)).mean()
    return theta*(e1 + e2 - 2*e3)

def mmd_vb_sampler(vb,sampler,nsamples,l=10.0,theta=100.0):
    X1 = vb.samples_proposal(nsamples)
    X2 = vb.samples_proposal(nsamples)
    X3 = vb.samples_proposal(nsamples)
    Y1 = sampler(nsamples)
    Y2 = sampler(nsamples)
    Y3 = sampler(nsamples)
    return maximum_mean_discrepancy(X1,X2,X3,Y1,Y2,Y3,l,theta)
#%%
def logmvn(x,mu,cov):
    x_ = x.reshape(-1,1) - mu
    L = torch.cholesky(cov)
    invLx = torch.triangular_solve(x_,L,upper=False)[0]
    expoent = -0.5*invLx.pow(2).sum()
    return expoent

def sampler_mvn(mu,cov,nsamples):
    L = torch.cholesky(cov,upper=False)
    Z = torch.randn(mu.shape[0],nsamples)
    X = L@Z + mu
    return X.t()
#%%%
training_conditions = [2]
for dim in training_conditions:
    np.random.seed(100)
    torch.manual_seed(100)
    folder_name = "ex4b_d%i"%dim
    try:
        os.mkdir(folder_name)
    except:
        pass
    lambd = 0.1*torch.ones(dim)
    lambd[0] = 10.0
    Lambda = torch.diag(lambd)
    Q = torch.tensor(special_ortho_group.rvs(dim)).float()
    cov = Q@Lambda@Q.t()
    mu = torch.zeros(dim,1)
    logjoint_ = functools.partial(logmvn,mu=mu,cov=cov)
    device = "cpu"
    def logjoint(sample):
        return logjoint_(sample.to("cpu")).to(device)
    true_mean,true_cov = mu.flatten().clone(),cov.clone()
    sampler = lambda nsamples : sampler_mvn(mu,cov,nsamples)
    #%%
    nsamples = 10*dim
    samples=sampling.sampling1(nsamples,dim,scale=5.0,device=device)
    mu0 = torch.zeros(dim).to(device)
    cov0 = (20.0/3)**2*torch.ones(dim).to(device)
    
    #%%
    #samples = vb.samples
    training_interval = 20
    acquisitions = ["prospective","mmlt"]
    vb = VariationalBoosting(dim,logjoint,samples,mu0,cov0,
                             kernel_function="PMat",
                             matern_coef=2.5,
                             degree_hermite=60)
    vb.optimize_bmc_model(maxiter=200)
    vb.update_full()
    #%%
    dmeans = [torch.norm(vb.currentq_mean.to("cpu")-true_mean).numpy()]
    dcovs = [torch.norm(vb.currentq_mean.to("cpu")-true_cov,2).numpy()]
    mmds = [mmd_vb_sampler(vb,sampler,100000)]
    weights = [vb.weights.cpu().numpy()]
    elbo_list = [vb.evidence_lower_bound(nsamples=10000).cpu().numpy()]
    step_list = [0]
    time_list = [0.0]
    for i in range(100):
        tictoc.tic()
        _ = vb.update(maxiter_nc=300,lr_nc=0.01,
                      n_samples_nc=500,n_samples_sn=300,n_iter_sn=300)
        try:
            acquisition = np.random.choice(acquisitions)
            vb.update_bmcmodel(acquisition=acquisition,vreg=1e-1)
        except:
            continue
        vb.cutweights(1e-3)
        if ((i+1)%training_interval) == 0:
            vb.update_full(cutoff=1e-3)
        elapsed = tictoc.toc(printing=False)
        dmeans.append(torch.norm(vb.currentq_mean.cpu()-true_mean).numpy())
        dcovs.append(np.linalg.norm(vb.currentq_cov.cpu()-true_cov))
        mmds.append(mmd_vb_sampler(vb,sampler,100000))
        elbo_list.append(vb.evidence_lower_bound(nsamples=10000).cpu().numpy())
        step_list.append(i+1)
        time_list.append(elapsed)
        print(dmeans[-1])
        print(dcovs[-1])
        print(elbo_list[-1])
        print(time_list[-1])
        print("Step %i"%(i+1),dim)
        
    dmeans_np = np.array(dmeans).astype(float)
    dcovs_np = np.array(dcovs).astype(float)
    elbo_np = np.array(elbo_list).astype(float)
    mmds_np = np.array(mmds)
    step_list_np = np.array(step_list).astype(int)
    samples_np = vb.samples.numpy()
    evaluations_np = vb.evaluations.numpy()
    np.savez("%s/tracking"%folder_name,means=dmeans_np,covs=dcovs_np,
             mmds=mmds_np,true_mean=true_mean.numpy(),
             true_cov=true_cov.numpy(),
             elbo=elbo_np,steps=step_list_np)
    np.savez("%s/distribdata"%folder_name,cov=cov.numpy())
    vb.save_distribution("%s/vbdistrib"%folder_name)