# -*- coding: utf-8 -*-
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def f(x,Sigma):
    return 1.0/np.linalg.det(2*np.pi*Sigma)*np.exp(-0.5*x.T@np.linalg.inv(Sigma)@x)

def fplot(X,Y,Sigma):
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xvec = np.array([X[i,j],Y[i,j]]).reshape(-1,1)
            Z[i,j] = f(xvec,Sigma)
    return Z
Sigma = np.array([[1.0,0.9],
                  [0.9,1.0]])

xplot = np.linspace(-3,3)
yplot = xplot.copy()
Xplot,Yplot = np.meshgrid(xplot,yplot)
Zplot = fplot(Xplot,Yplot,Sigma)
plt.contour(Xplot,Yplot,Zplot,levels=5,colors='blue',label='p(x,y)')

#%% Minimization D_KL(q||p)
def softplus(x):
    return np.log(1+np.exp(x))
def kl_mvn(Sigma1,Sigma2): #D_KL(n(Sigma1)||n(Sigma2))
    iSigma2 = np.linalg.inv(Sigma2)
    return 0.5*(-np.log(np.linalg.det(iSigma2@Sigma1)) - Sigma1.shape[0] + \
                np.trace(iSigma2@Sigma1))
def objectiveqp(s,Sigmap):
    s_ = softplus(s)
    Sigmaq = np.diag(s_)
    return kl_mvn(Sigmaq,Sigmap)

optqp = minimize(lambda s : objectiveqp(s,Sigma),[0.1,0.1])
Sigma2qp = np.diag(softplus(optqp.x))
Zplotqp = fplot(Xplot,Yplot,Sigma2qp)
plt.contour(Xplot,Yplot,Zplotqp,levels=5,colors='red',label='q(x,y)',alpha=0.75)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('/home/danilo/Danilo/Dissertação/Tex/figs/klil3a')
#%%
plt.figure()
plt.contour(Xplot,Yplot,Zplot,levels=5,colors='blue')
def objectivepq(s,Sigmap):
    s_ = softplus(s)
    Sigmaq = np.diag(s_)
    return kl_mvn(Sigmap,Sigmaq)

optpq = minimize(lambda s : objectivepq(s,Sigma),[0.1,0.1])
Sigma2pq = np.diag(softplus(optpq.x))
Zplotpq = fplot(Xplot,Yplot,Sigma2pq)
plt.contour(Xplot,Yplot,Zplotpq,levels=5,colors='red',alpha=0.75)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('/home/danilo/Danilo/Dissertação/Tex/figs/klil3b')
