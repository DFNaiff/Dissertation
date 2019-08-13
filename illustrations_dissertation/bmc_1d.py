# -*- coding: utf-8 -*-
import functools

import numpy as np
import matplotlib.pyplot as plt
#%%
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
    Xd = np.tile(Xd,[1,Y.shape[0],1])
    Yd = np.tile(Yd,[X.shape[0],1,1])
    return Xd,Yd

def kernel_matrix(X,Y,k,symmetrize=False):
    Xd,Yd = meshgrid3d(X,Y)
    K = k(Xd,Yd,keepdims=False)
    if symmetrize:
        K = 0.5*(K + K.transpose(1,0))
    return K

def rbf_kernel_f(x,y,l,theta,keepdims=True):
    return theta*np.exp(-0.5*np.sum((x-y)**2/(l**2),axis=-1,
                                          keepdims=keepdims))

def bmc(xsamples,ysamples,l,theta,mu,sigma2):
    k = functools.partial(rbf_kernel_f,l=l,theta=theta)
    K = kernel_matrix(xsamples,xsamples,k)
    z = theta/np.sqrt(1 + sigma2/(l**2))*\
            np.exp(-0.5*(xsamples-mu)**2/(sigma2+l**2))
    mean = z.transpose()@np.linalg.solve(K,ysamples)
    var = theta/np.sqrt(1 + 2*sigma2/(l**2)) - \
            z.transpose()@np.linalg.solve(K,z)
    return mean,var

def gp_mean_var(xsamples,ysamples,l,theta):
    k = functools.partial(rbf_kernel_f,l=l,theta=theta)
    K = kernel_matrix(xsamples,xsamples,k)
    L = np.linalg.cholesky(K)
    def m(x):
        kx = kernel_matrix(xsamples,x,k)
        mean = np.linalg.solve(L,kx).transpose()@np.linalg.solve(L,ysamples)
        return mean
    def v(x):
        kx = kernel_matrix(xsamples,x,k)
        cov = kernel_matrix(x,x,k) - \
            np.linalg.solve(L,kx).transpose()@np.linalg.solve(L,kx)
        var = np.diag(cov)
        return var.reshape(-1,1)
    return m,v
#%%
l = 1.0
theta = 1.0
mean = 0.0
var = 0.5

xsamples = np.array([-2.0,-1.0,0.0,1.0,2.0]).reshape(-1,1)
ysamples = -xsamples**2

mean_bmc,var_bmc = bmc(xsamples,ysamples,l,theta,mean,var)
mean_gp,var_gp = gp_mean_var(xsamples,ysamples,l,theta)
#%%
xplot = np.linspace(-4,4).reshape(-1,1)
ytrue = -xplot**2
ygp = mean_gp(xplot)
vargp = var_gp(xplot)
lowgp = (ygp-2*np.sqrt(vargp)).flatten()
highgp = (ygp+2*np.sqrt(vargp)).flatten()
fdistrib = 1/np.sqrt(2*np.pi*var)*np.exp(-0.5*(xplot-mean)**2/var)
#%%
fig,ax1 = plt.subplots()
color="tab:blue"
ax1.plot(xplot,ygp,color=color,label=r"$m_\mathcal{D}(x)$")
ax1.fill_between(xplot.flatten(),lowgp,highgp,color=color,
                 alpha=0.5)
ax1.plot(xplot,ytrue,color="black",linestyle='-.',
         label=r"$f(x)$")



ax1.set_xlabel("$x$")
ax1.set_ylabel("y",color=color)
ax1.set_ylim(bottom=-6.0)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc="upper left")
ax1.text(0.7, 1.4, '$\int f(x) p(x) dx = %.3f$'%(-var))
ax1.text(0.7, 0.8, '$E[Z_\mathcal{D}]=%.3f$'%mean_bmc)
ax1.text(0.7, 0.2, '$Var[Z_\mathcal{D}]=%.3f \cdot 10^{-5}$'%(1e5*var_bmc))
color = 'tab:red'
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(xplot,fdistrib,color=color,linestyle='--')
ax2.set_ylabel("$p(x)$",color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.savefig("/home/danilo/Danilo/Dissertação/Tex/figs/exbmc")
