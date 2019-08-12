# -*- coding: utf-8 -*-
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

# Both cases: true = 0.5*N(-2,1.) + 0.5*N(2,1.0) is proposal
# N(mu,sigma2) is proposal
#%% 1 : D_KL(q||p)
dmean = 5
np.random.seed(150)
torch.manual_seed(120)
def true_f(x):
    return 1/math.sqrt(2*math.pi)*(0.5*torch.exp(-0.5*(x-dmean/2)**2) + \
                                   0.5*torch.exp(-0.5*(x+dmean/2)**2))

def samples_true_f(nsamples):
    weights = torch.tensor([0.5,0.5])
    means_ = torch.multinomial(weights,nsamples,replacement=True).float()
    means = means_*dmean - dmean/2
    return torch.randn(nsamples)+means

def q_f(x,mu,sigma):
    return 1/torch.sqrt(2*math.pi*(sigma**2))*\
             torch.exp(-0.5*(x-mu)**2/(sigma**2))


mu_a = torch.randn(1)
sigma_a = torch.tensor(1.0)
mu_a.requires_grad = True
optimizer = torch.optim.Adam([mu_a])
#sigma_a.requires_grad = True
#optimizer = torch.optim.Adam([mu_a,sigma_a])

nsamples = 10000
for i in range(1000):
    samples = sigma_a*torch.randn(nsamples) + mu_a
    d_kl = -torch.mean(torch.log(true_f(samples)) - 
                       torch.log(q_f(samples,mu_a,sigma_a)))
    d_kl.backward()
    optimizer.step()
print(d_kl)
mu_a.requires_grad = False
sigma_a.requires_grad = False
xplot = torch.linspace(-6,6)
yplot1 = true_f(xplot)
plt.plot(xplot.numpy(),yplot1.numpy(),label=r"$p(\theta)$")
xplot = torch.linspace(-6,6)
yplot2 = q_f(xplot,mu_a,sigma_a)
plt.plot(xplot.numpy(),yplot2.numpy(),label=r"$\mathcal{N}(\theta;\mu^*,1)$")
plt.legend()
plt.xlabel(r"$\theta$")
plt.savefig("/home/danilo/Danilo/Dissertação/Tex/figs/dklil1a")
muplot = torch.linspace(-6,6)
dkls = torch.zeros_like(torch.linspace(-8,8))
for i,mu_ in enumerate(muplot):
    samples = sigma_a*torch.randn(nsamples) + mu_
    dkls[i] = -torch.mean(torch.log(true_f(samples)) - 
                       torch.log(q_f(samples,mu_,sigma_a)))
plt.figure()
plt.plot(muplot.numpy(),dkls.numpy(),'o')
plt.xlabel(r"$\mu$")
plt.ylabel(r"$D_{KL}(\mathcal{N}(\theta;\mu,1)||p(\theta))$")
plt.savefig("/home/danilo/Danilo/Dissertação/Tex/figs/dklil1b")
#%% 2 : D_KL(q||p)
plt.figure()

mu_a = torch.randn(1)
mu_a.requires_grad = True
optimizer = torch.optim.Adam([mu_a])
#sigma_a.requires_grad = True
#optimizer = torch.optim.Adam([mu_a,sigma_a])

nsamples = 10000
for i in range(1000):
    samples = samples_true_f(nsamples)
    d_kl = -torch.mean(torch.log(q_f(samples,mu_a,sigma_a))/true_f(samples))
    d_kl.backward()
    optimizer.step()
print(d_kl)
mu_a.requires_grad = False
sigma_a.requires_grad = False
xplot = torch.linspace(-6,6)
yplot1 = true_f(xplot)
plt.plot(xplot.numpy(),yplot1.numpy(),label=r"$p(\theta)$")
xplot = torch.linspace(-6,6)
yplot2 = q_f(xplot,mu_a,sigma_a)
plt.plot(xplot.numpy(),yplot2.numpy(),label=r"$\mathcal{N}(\theta;\mu^*,1)$")
plt.legend()
plt.xlabel(r"$\theta$")
plt.savefig("/home/danilo/Danilo/Dissertação/Tex/figs/dklil2a")

muplot = torch.linspace(-6,6)
dkls = torch.zeros_like(muplot)
for i,mu_ in enumerate(muplot):
    samples = samples_true_f(nsamples)
    dkls[i] = -torch.mean(-torch.log(true_f(samples)) + 
                       torch.log(q_f(samples,mu_,sigma_a)))
plt.figure()
plt.plot(muplot.numpy(),dkls.numpy(),'o')
plt.xlabel(r"$\mu$")
plt.ylabel(r"$D_{KL}(p(\theta)||\mathcal{N}(\theta;\mu,1))$")
plt.savefig("/home/danilo/Danilo/Dissertação/Tex/figs/dklil2b")

