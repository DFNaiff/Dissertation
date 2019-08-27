# -*- coding: utf-8 -*-
import functools

import numpy as np
import matplotlib.pyplot as plt

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

#%% Priors
theta = 1.0
l = 1.0
X = np.linspace(-2,2)
kfunc = functools.partial(rbf_kernel_f,l=l,theta=theta)
K = kernel_matrix(X.reshape(-1,1),X.reshape(-1,1),kfunc)
ncurves = 200
y = np.random.multivariate_normal(np.zeros(len(X)),K,ncurves)
for yy in y:
    plt.plot(X.flatten(),yy,'b--',alpha=0.2)
plt.fill_between(X,- 2*np.sqrt(np.diag(K)),
                    2*np.sqrt(np.diag(K)),
                   color='red',alpha=0.2)

plt.savefig("/home/danilo/Danilo/Dissertação/Tex/figs/gprex1a")
plt.show()
#%% Posterior
def f(x):
    return np.sin(np.pi*x)
xpoints = np.linspace(-0.1,1.5,5)
ypoints = f(xpoints)
Kxx = kernel_matrix(xpoints.reshape(-1,1),xpoints.reshape(-1,1),kfunc)
Kxx_inv = np.linalg.inv(Kxx)
Kplot = kernel_matrix(X.reshape(-1,1),xpoints.reshape(-1,1),kfunc)
Kplot2 = kernel_matrix(X.reshape(-1,1),X.reshape(-1,1),kfunc)
pos_mean = Kplot@Kxx_inv@(ypoints.reshape(-1,1))
pos_cov = Kplot2 - Kplot@Kxx_inv@Kplot.T
y2 = np.random.multivariate_normal(pos_mean.flatten(),pos_cov,ncurves)
plt.fill_between(X,pos_mean.flatten() - 2*np.sqrt(np.diag(pos_cov)),
                   pos_mean.flatten() + 2*np.sqrt(np.diag(pos_cov)),
                   color='red',alpha=0.2)
#plt.plot(X,pos_mean,'red',alpha=0.5)
for yy2 in y2:
    plt.plot(X.flatten(),yy2,'b--',alpha=0.2)
plt.plot(X,f(X),'black',linestyle='--',alpha=0.8)
plt.plot(xpoints,ypoints,'go')
plt.savefig("../tex/figs/gprex1b")
plt.show()