# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as spstats
import torch

def uniform_sphere(nsamples,ndim):
    X = np.random.randn(nsamples,ndim)
    X = X/np.sqrt(np.square(X).sum(axis=1,keepdims=True))
    return X

def sampling1(nsamples,ndim,scale=1.0,
              to_tensor=True,device="cpu",**kwargs):
    
    res = scale*spstats.halfnorm().rvs((nsamples,1))*uniform_sphere(nsamples,ndim)
    if to_tensor:
        res = torch.tensor(res.astype(np.float32),device=device)
    return res

def _between_any(x,I):
    """
        x : (,). I : (n,d)
    """
    return np.any((x>=I[:,0]) & (x<=I[:,1]))