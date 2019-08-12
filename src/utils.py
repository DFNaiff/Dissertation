#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:18:28 2019

@author: danilo
"""
import math
import functools
import time

import numpy as np
import torch


def batch_trtrs(X,L,upper=False):
    """
        X : (t,d,n)
        L : (t,d,d)
        returns : (t,d,n)
    """
    t = X.shape[0]
    sollist = [torch.triangular_solve(X[i,:,:],L[i,:,:],upper=upper)[0] for i in range(t)]
    result = torch.stack(sollist,dim=0)
    return result


def batch_diag1(A):
    """
        A :(t,d,d)
        returns : (t,d)
    """
    t = A.shape[0]
    sollist = [torch.diag(A[i,:,:]) for i in range(t)]
    result = torch.stack(sollist,dim=0)
    return result


def expand(A,B,C):
    """
        B : (n,m)
        C : (m,m)
    """
    D1 = torch.cat([A,B],dim=1)
    D2 = torch.cat([B.transpose(1,0),C],dim=1)
    res = torch.cat([D1,D2])
    return res


def expand_cholesky_lower(L,B,C):
    """
       L : lower cholesky factor of a positive-definite matrix K (n,n)
       B : (m,n)
       C : (m,m)
       
       Assumes that [K b^T, b c] will be positive-definite
       
       returns : lower cholesky factor of the matrix
           K b^T
           b c
    """
    res = expand_cholesky_upper(L.transpose(1,0),B.transpose(1,0),
                                C).transpose(1,0)
    return res


def expand_cholesky_upper(U,B,C):
    """
       U : upper cholesky factor of a positive-definite matrix K (n,n)
       B : (n,m)
       C : (m,m)
       
       Assumes that [K b, b^T c] will be positive-definite
       
       returns : upper cholesky factor of the matrix
           K b^T
           b c
    """
    S11 = U
    S21 = torch.triangular_solve(B,S11,upper=True,transpose=True)[0] #(n,m)
    S22 = torch.cholesky(C - torch.matmul(S21.t(),S21),upper=True)
    return expand_upper(S11,S21,S22)
    

def expand_lower(A,B,C):
    """
        B : (m,n)
        C : (m,m)
    """
    D1 = torch.cat([A,torch.zeros_like(B).transpose(1,0)],dim=1)
    D2 = torch.cat([B,C],dim=1)
    res = torch.cat([D1,D2])
    return res


def expand_upper(A,B,C):
    """
        B : (n,m)
        C : (m,m)
    """
    D1 = torch.cat([A,B],dim=1)
    D2 = torch.cat([torch.zeros_like(B).transpose(1,0),C],dim=1)
    res = torch.cat([D1,D2])
    return res

def logexpsum(expoents,weights=None,flatten = False,dim=-1):
    #Expoents : (*,d). Weights : (d,) or (1,d), (1,1,d)...
    #returns (*,1) or #(*,), depending on flatten
    max_expoent = torch.max(expoents,dim=dim,keepdim=True)[0]
    expoents_star = expoents - max_expoent
    sumexp_star = torch.sum(weights*torch.exp(expoents_star),dim=dim,keepdim=True)
    res = max_expoent + torch.log(sumexp_star)
    if flatten:
        res = res.squeeze(dim)
    return res

def logmvn(x,mu,cov):
    """
        x : (n_samples,n_dim) tensor or (n_dim) tensor
        mu : (n_dim,) tensor
        cov : (n_dim,) tensor
    """
    if x.dim() == 1:
        x = x.reshape(1,-1)
    m = x - mu #(nsamples,ndim)
    term1 = -0.5*torch.sum(m**2/cov,dim=1,keepdim=True) #(nsamples,1)
    term2 = -0.5*torch.sum(torch.log(cov)) #(,)
    term3 = -0.5*mu.numel()*math.log(2*math.pi) #(,)
    return term1 + term2 + term3 #(nsamples,1)

def logmvnbatch(x,mu,cov):
    """
        inputs
            x : (*,n_dim)
            mu : (n_q,n_dim)
            cov : (n_q,n_dim)
        outputs 
            res : (*,n_q) tensor where 
                res[:,:i] = logmvn(x,mu[i,:],cov[i,:]) 
    """
    D = x.shape[-1]
    x_ = x.unsqueeze(-2) #(*,1,D)
    z_ = (x_-mu)**2/cov #(*,Q,D)
    term1 = -0.5*torch.sum(z_,dim=-1) #(*,Q)
    term2 = -0.5*torch.sum(torch.log(cov),dim=-1) #(Q,)
    term3 = -0.5*D*math.log(2*math.pi) #(,)
    res = term1 + term2 + term3 #(*,Q)
    return res

def is_none(var):
    return type(var) == type(None)


def integral_vector(X,theta,l,mu,cov):
    """
        X : (n,d) tensor
        theta : 0d tensor or float
        l : (d,) tensor
        mu : (d,) tensor
        cov : (d,d) tensor
        outputs (n,) tensor
    """
    C = cov + torch.diag(l**2)
    L = torch.cholesky(C,upper=False)
    Xm = X - mu #nxd#
    LX = torch.triangular_solve(Xm.transpose(1,0),L,upper=False)[0] #d x n
    expoent = -0.5*torch.sum(LX**2,dim=0) #(n,)
    det = torch.prod(1/l**2)*torch.prod(torch.diag(L))**2 #|I + A^-1B|
    vec = theta/torch.sqrt(det)*torch.exp(expoent) #(n,)
    return vec


def inv_quad_chol_lower(x,L,y):
    """
        Computes x (L L^T) y
        x : (m x n)
        L : (n x n)
        y : (n x k)
    """
    m = x.shape[0]
    if m == 1:
        xy = torch.cat([x.reshape(-1,1),y],dim=1)
        return torch.triangular_solve(xy,L,upper=False)[0].prod(dim=1).sum(dim=0).reshape(1,1)
    else:
        z1 = torch.triangular_solve(x.transpose(1,0),L,upper=False)[0].transpose(1,0) #m x n
        z2 = torch.triangular_solve(y,L,upper=False)[0] #n x k
        return torch.matmul(z1,z2)


def inv_quad_chol_lower_2(L,y):
    """
        Computes y^T (L L^T) y
        L : (n x n)
        y : (n x m)
    """
    z = torch.triangular_solve(y,L,upper=False)[0]
    if z.shape[1] == 0:
        return torch.sum(z**2).reshape(1,1)
    else:
        return torch.matmul(z.transpose(1,0),z)


def inv_quad_chol_lower_3(L,y):
    """
        Computes diagonal of y^T (L L^T) y
        L : (n x n)
        y : (n x m)
    """
    z = torch.triangular_solve(y,L,upper=False)[0]
    return torch.sum(z**2,dim=0).reshape(-1,1) #(m,1)


def jitterize(K,j,proportional=True):
    if proportional:
        jitter_factor = torch.mean(torch.diag(K)).item()*j
    else:
        jitter_factor = j
    K[range(len(K)),range(len(K))] += jitter_factor
    return K    


def kernel_matrix(X,Y,k,symmetrize=False):
    Xd,Yd = meshgrid3d(X,Y)
    K = k(Xd,Yd,keepdim=False)
    if symmetrize:
        K = 0.5*(K + K.transpose(1,0))
    return K


def meshgrid3d(X,Y):
    """
        input:
            X : (m,d) tensor
            Y : (n,d) tensor
        returns:
            Xd : (m x n x d) tensor
            Yd : (m x n x d) tensor
    """
    Xd = X.reshape(X.shape[0],1,X.shape[1])
    Yd = Y.reshape(1,Y.shape[0],Y.shape[1])
    Xd = Xd.repeat([1,Y.shape[0],1])
    Yd = Yd.repeat([X.shape[0],1,1])
    return Xd,Yd


def potrs(y,L,upper=False):
    """
        Since there is no derivative for potrs, we make two trtrs
    """
    z_ = torch.triangular_solve(y,L,upper=upper,transpose=upper)[0]
    z = torch.triangular_solve(z_,L,upper=upper,transpose=not upper)[0]
    return z


def psddet(M):
    L = torch.cholesky(M)
    d = torch.prod(torch.diag(L))**2
    return d


def quadratic_mean_lsq(X,y):
    """
        Fit sum((x-c)**2/l**2)
    """
    d = X.shape[1]
    A = torch.cat([torch.ones_like(y),
                   X,X**2],dim=1)
    Q,R = torch.qr(A)
    coefs = torch.triangular_solve(torch.matmul(Q.transpose(1,0),y),R,upper=True)[0].flatten()
    a = coefs[d+1:]
    b = coefs[1:d+1]
    c = coefs[0]
    lengthscales = torch.sqrt(-1.0/(2*a))
    center = b*lengthscales**2
    constant = c + 0.5*torch.sum(center**2/(lengthscales**2))
    return lengthscales,center,constant


def rbf_kernel_f(x,y,l,theta,keepdim=True):
    return theta*torch.exp(-0.5*torch.sum((x-y)**2/(l**2),dim=-1,
                                          keepdim=keepdim))


def rbf_kernel(X,Y,theta,l,symmetrize=False,diag=False):
    k = functools.partial(rbf_kernel_f,l=l,theta=theta)
    if diag:
        return k(X,Y)
    else:
        return kernel_matrix(X,Y,k,symmetrize)

def sum_matern_kernel_f(x,y,l,theta,nu=0.5,keepdim=True):
    if nu == 0.5:
        ks = torch.exp(-torch.abs(x-y)/l) #(*,D)
    else:
        raise NotImplementedError
    res = theta*torch.sum(ks,dim=-1,keepdim=keepdim)
    return res

def sum_matern_kernel(X,Y,theta,l,nu,symmetrize=False,diag=False):
    k = functools.partial(sum_matern_kernel_f,l=l,theta=theta,nu=nu)
    if diag:
        return k(X,Y)
    else:
        return kernel_matrix(X,Y,k,symmetrize)

def prod_matern_kernel_f(x,y,l,theta,nu=0.5,keepdim=True):
    r = torch.abs(x-y)/l
    if nu == 0.5:
        ks = torch.exp(-r) #(*,D)
    elif nu == 1.5:
        ks = (1+math.sqrt(3)*r)*torch.exp(-math.sqrt(3)*r)
    elif nu == 2.5:
        ks = (1+math.sqrt(5)*r+5.0/3.0*r**2)*torch.exp(-math.sqrt(5)*r)
    else:
        raise NotImplementedError
    res = theta*torch.prod(ks,dim=-1,keepdim=keepdim)
    return res

def prod_matern_kernel(X,Y,theta,l,nu,symmetrize=False,diag=False):
    k = functools.partial(prod_matern_kernel_f,l=l,theta=theta,nu=nu)
    if diag:
        return k(X,Y)
    else:
        return kernel_matrix(X,Y,k,symmetrize)

def sm_kernel_f(x,y,weights,l,mu,keepdim=True):
    """
        x : (*,D)
        y : (*,D)
        weights : (Q,)
        l : (Q,D)
        mu : (Q,D)
        returns : (*,D) or (*,) (depending on keepdim)
    """
    
    prev_shape = x.shape[:-1] #will have to fit y
    prev_ndims = len(prev_shape)
    Q = l.shape[0]
    D = l.shape[1]
    tau = (x - y).unsqueeze(-2).repeat((1,)*prev_ndims+(Q,1)) #(*,Q,D)
    scales = (1.0/l)**2
#    l = l.repeat([N,1,1]) #(N,Q,D)
#    mu = mu.repeat([N,1,1]) #(N,Q,D)
#    weights = weights.repeat([N,1]) #(N,Q)
    res = torch.exp(-2*math.pi**2*(tau**2)*scales)*torch.cos(2*math.pi*mu*tau) #(*,Q,D)
    res = res.prod(dim=-1) #(N,Q)
    res = (weights*res).sum(dim=-1,keepdim=keepdim)
    return res


def sm_kernel(X,Y,weights,l,mu,symmetrize=False,diag=False):
    k = functools.partial(sm_kernel_f,weights=weights,l=l,mu=mu)
    if diag:
        return k(X,Y)
    else:
        return kernel_matrix(X,Y,k,symmetrize)


def softplus(x,minvalue=1e-6):
    return torch.log(torch.exp(x)+1) + minvalue


def invsoftplus(x,minvalue=1e-6):
    return torch.log(torch.exp(x-minvalue)-1)

# =============================================================================
# K-Means due to https://github.com/overshiki
# =============================================================================

def _pairwise_distance(data1, data2=None, device=-1):
	r'''
	using broadcast mechanism to calculate pairwise ecludian distance of data
	the input data is N*M matrix, where M is the dimension
	we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
	then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
	'''
	if data2 is None:
		data2 = data1 

	if device!=-1:
		data1, data2 = data1.cuda(device), data2.cuda(device)

	#N*1*M
	A = data1.unsqueeze(dim=1)

	#1*N*M
	B = data2.unsqueeze(dim=0)

	dis = (A-B)**2.0
	#return N*N matrix for pairwise distance
	dis = dis.sum(dim=-1).squeeze()
	return dis

def _group_pairwise(X, groups, device=0, fun=lambda r,c: _pairwise_distance(r, c).cpu()):
    group_dict = {}
    for group_index_r,group_r in enumerate(groups):
        for group_index_c,group_c in enumerate(groups):
            R,C = X[group_r],X[group_c]
            if device != 1:
                R = R.cuda(device)
                C = C.cuda(device)
            group_dict[(group_index_r,group_index_c)] = fun(R,C)
    return group_dict


def forgy(X, n_clusters):
	_len = len(X)
	indices = np.random.choice(_len, n_clusters)
	initial_state = X[indices]
	return initial_state


def lloyd(X,nclusters,tol=1e-4,maxiter=100):
    initial_state = forgy(X,nclusters)
    for i in range(maxiter):
        dis = _pairwise_distance(X,initial_state)
        choice_cluster = torch.argmin(dis,dim=1)
        current_clusters = torch.unique(choice_cluster,sorted=True)
        initial_state = initial_state[current_clusters,:] #Remove empty clusters
        initial_state_pre = initial_state.clone()
        for j,index in enumerate(current_clusters):
            selected = torch.nonzero(choice_cluster==index).squeeze()
            selected = torch.index_select(X,0,selected)            
            initial_state[j] = selected.mean(dim=0)
        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state-initial_state_pre).pow(2),dim=1)))
        if center_shift**2 < tol:
            break
    return choice_cluster,initial_state


class TicToc(object):
    def __init__(self):
        pass
    
    def tic(self):
        self._tic_time = time.time()
    
    def toc(self,printing=True):
        elapsed = time.time() - self._tic_time
        if printing:
            print("Elapsed time :%f"%elapsed)
        else:
            return elapsed
        

#TODO : Extend this to an actual Torch distribution
class MixtureNormalDiagCov(object):
    def __init__(self,mu,cov,weights):
        """
            mu : (N,dim) tensor
            cov : (N,dim) tensor
            weights : (N,) tensor
        """
        self.mu_t = mu
        self.cov_t = cov
        self.weights = weights
        
    def sample(self,nsamples=1,flatten=False):
        inds = torch.multinomial(self.weights.flatten(),
                                 nsamples,replacement=True)
        Z = torch.randn(nsamples,self.dim,device=self.mu_t.device)
        samples = torch.sqrt(self.cov_t[inds,:])*Z + \
                  self.mu_t[inds,:]
        if flatten and nsamples == 1:
            samples = samples.flatten()
        return samples

    def mean(self):
        return (self.mu_t*self.weights.reshape(-1,1)).sum(dim=0)
    
    def cov(self):
        moment2_ = self.cov2_t + torch.bmm(self.mu_t.unsqueeze(2),
                                           self.mu_t.unsqueeze(1))
        moment2 = (moment2_*self.weights.reshape(-1,1,1)).sum(dim=0)
        cov = moment2 - torch.ger(self.mean(),self.mean())
        return cov
    
    def logprob(self,theta):
        """
            Arguments:
                theta : and (nsamples,ndim) tensor or a (ndim,) tensor
            Returns:
                (nsamples,1) evaluations of current log density of proposal at theta
        """
        if theta.dim() == 1:
            theta = theta.reshape(1,-1)
        #theta : (*,D)
        logqs = logmvnbatch(theta,self.mu_t,self.cov_t) #(*,Q)
        weights = self.weights #(Q,)
        res = logexpsum(logqs,weights,flatten=False) #(*,1)
        return res
    
    @property
    def cov2_t(self):
        return torch.diag_embed(self.cov_t)
    
    @property
    def num_mixtures(self):
        return self.weights.numel()
    
    @property
    def dim(self):
        return self.mu_t.shape[1]