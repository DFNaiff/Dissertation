# -*- coding: utf-8 -*-

# =============================================================================
# Here we will be tested multivariate T
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

tictoc = TicToc()
#%%
def logexpsum(expoents,weights):
    #Expoents : (n,d). Weights : (1,d)
    max_expoent = torch.max(expoents,dim=1,keepdim=True)[0]
    expoents_star = expoents - max_expoent
    sumexp_star = torch.sum(weights*torch.exp(expoents_star),dim=1,keepdim=True)
    return max_expoent + torch.log(sumexp_star)

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
def logtnd(x,nu):
    return torch.sum(-0.5*(nu+1)*torch.log(1.0 + (1.0/nu)*(x)**2))

def logrotated(x,f,A):
    return f(torch.solve(x.reshape(-1,1),A)[0].flatten())

def samplertndrot(nsamples,nu,A,dim):
    samples_ = torch.distributions.StudentT(0.5).rsample((dim,nsamples))
    samples = (A@samples_).t()
    return samples

#%%%
dims = [2]
for dim in dims:
    np.random.seed(100)
    torch.manual_seed(100)
    folder_name = "ex5b_d%i"%dim
    try:
        os.mkdir(folder_name)
    except:
        pass
    nu= torch.rand(dim)*(2+0.5*dim-2.5)+2.5
    A = torch.eye(dim)
    logjoint_ = functools.partial(logrotated,
                                  f=functools.partial(logtnd,nu=nu),
                                  A=A)
    device = "cpu"
    def logjoint(sample):
        return logjoint_(sample.to("cpu")).to(device)
    true_mean = torch.zeros(dim)
    true_cov = A@torch.diag(nu/(nu-2)*torch.ones(dim))@A.t()
    sampler = lambda nsamples : samplertndrot(nsamples,nu,A,dim)
    #%%
    nsamples = 10*dim
    samples=sampling.sampling1(nsamples,dim,scale=5.0,device=device)
    mu0 = torch.zeros(dim).to(device)
    cov0 = (20.0/3)**2*torch.ones(dim).to(device)
    #%%
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
    np.savez("%s/distribdata"%folder_name,A=A.numpy())
    vb.save_distribution("%s/vbdistrib"%folder_name)