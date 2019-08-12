# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,"../../src2")
import math
import functools
import time

import torch
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import emcee

from source_1d_likelihood_fn import compute_log_likelihood_2

np.random.seed(100)
torch.manual_seed(100)
#%%
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
    ll = compute_log_likelihood_2(x0,Ts,q0,rho)
    ll += -np.log(1+(q0/10.0)**2)
    ll += -np.log(1+(rho/0.1)**2)
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
    print(counter)
    return logjoint_np(x)

#%%
ndim, nwalkers = 4, 10
p0 = [np.random.rand(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, logjoint_emcee)
sampler.run_mcmc(p0, 10000)
np.savez("testheat_1a_emcee",sampler=sampler)
#%%
