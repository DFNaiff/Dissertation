# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,"../../src")
import math
import functools
import time

import torch
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import emcee
import pandas as pd
import seaborn as sns
import pymc3

from source_1d_likelihood_fn import compute_log_likelihood

np.random.seed(100)
torch.manual_seed(100)
#%% This part of the code is not important.
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
def unwarped_logjoint_np(x0,Ts,q0,rho):
#    x0,Ts,q0,rho = x[0],x[1],x[2],x[3]
    ll = compute_log_likelihood(x0,Ts,q0,rho)
    ll += -np.log(1+(q0/10.0)**2)
    ll += -np.log(1+(rho/0.1)**2)
#    ll += -lambd_q*q0
#    ll += np.exp(-lambd_rho*rho) + np.exp(-lambd_q*q0)
    return ll

def logjoint_np(x):
    x0,Ts,q0,rho = x[0],x[1],x[2],x[3]
    ll = unwarped_logjoint_np(sigmoid(x0),sigmoid(Ts,b=0.4),
                              exp(q0),exp(rho)) + \
         np.log(dsigmoid(x0)) + np.log(dsigmoid(Ts,b=0.4)) + \
         np.log(dexp(q0)) + np.log(dexp(rho))
    return ll

counter=0
def logjoint_emcee(x):
    global counter
    counter += 1
    return logjoint_np(x)

#%%
ndim, nwalkers = 4, 10
p0 = [np.random.rand(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, logjoint_emcee)
sampler.run_mcmc(p0, 1)
#np.savez("testheat_1g_emcee_3",sampler=sampler)
#%% Data generation start here
sampler = np.load("testheat_1a_emcee.npz")["sampler"][()]
def sample_sampler(nsamples,sampler,burnin):
    chain = sampler.flatchain[burnin:,:]
    choices = np.random.choice(chain.shape[0],nsamples)
    return chain[choices,:]
samples = sample_sampler(5000,sampler,10000)
samples[:,0] = sigmoid(samples[:,0])
samples[:,1] = sigmoid(samples[:,1],b=0.4)
samples[:,2] = exp(samples[:,2])
samples[:,3] = exp(samples[:,3])

for i in range(4):
    print("Data %i"%i)
    mean = samples[:,i].mean()
    print(pymc3.stats.hpd(samples[:,i],0.3))

names = [r"$x_0$",r"$t_s$",r"$q_0$",r"$\rho$"]
datadict = dict([(names[i],samples[:,i]) for i in range(len(names))])
dataframe = pd.DataFrame(datadict)

lims = [(-0.2,1.2),
        (-0.2,0.5),
        (-0.1,21.0),
        (-0.1,2.1)]

g = sns.PairGrid(dataframe)
def set_lims_pairgrid(g,lims):
    shape = g.axes.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i == j:
                g.axes[i,j].set_xlim(lims[i])
            elif i > j:
                g.axes[i,j].set_xlim(lims[j])
                g.axes[i,j].set_ylim(lims[i])
#            else:
#                g.axes[i,j].set_xlim(lims[i])
#                g.axes[i,j].set_ylim(lims[j])
g.map_diag(sns.kdeplot)
g.map_lower(sns.kdeplot, n_levels=10);
set_lims_pairgrid(g,lims)
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i, j].set_visible(False)
g.savefig("../../tex/figs/sourceproblemhistogramsemcee")