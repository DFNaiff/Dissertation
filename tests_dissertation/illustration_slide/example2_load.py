# -*- coding: utf-8 -*-

# =============================================================================
# Here we will be testing with mixtures of student-t, 1d
# =============================================================================
import sys
sys.path.insert(0,"../../src2")
import math
import functools

import torch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
torch.manual_seed(100)

training_interval = 10
data_folder = "example3"
data = np.load("%s/tracking.npz"%data_folder,allow_pickle=True)

xplot = data["xplot"]
true_posterior = data["true_posterior"]
bmc_pred = data["bmc_pred"]
vb_posterior = data["vb_posterior"]
samples_dict = data["samples_dict"][()]
#means = data["means"]
#covs = data["covs"]
#elbo = data["elbo"]
#steps = data["steps"]
#time = data["time"]
#true_posterior = data["true_posterior"]
#bmc_pred = data["bmc_pred"]
#vb_posterior = data["vb_posterior"]
#xplot = data["xplot"]
order = 1
def savefigtitle(fig):
    global order
    fig.savefig("../../tex_pres/figs/exampleintro%i"%order)
    order += 1
fig,ax = plt.subplots()
ax.plot(xplot,np.exp(true_posterior))
ax.set_ylim(bottom=-0.01,top=0.255)
ax.set_title("True posterior")
savefigtitle(fig)
fig,ax = plt.subplots()
ax.plot(xplot,np.exp(true_posterior))
ax.plot(samples_dict[0][0],np.exp(samples_dict[0][1]),'ro')
ax.set_ylim(bottom=-0.01,top=0.255)
ax.set_title("Evaluations (%i samples)"%samples_dict[0][0].shape[0])
savefigtitle(fig)
fig,ax = plt.subplots()
ax.plot(xplot,np.exp(true_posterior),alpha=0.5)
ax.plot(samples_dict[0][0],np.exp(samples_dict[0][1]),'ro')
ax.plot(xplot,np.exp(bmc_pred[10,:]),'m')
ax.set_ylim(bottom=-0.01,top=0.255)
ax.set_title("GP approximation (%i samples)"%samples_dict[0][0].shape[0])
savefigtitle(fig)

for i in [1,2,5,10,19,20,29,30,39]:
    fig,ax = plt.subplots()
    ax.plot(xplot,np.exp(true_posterior),alpha=0.5)
    alpha_bmc = 1.0
    if i%10 == 0:
        ax.plot(samples_dict[i][0][:-1],np.exp(samples_dict[i][1][:-1]),'ro')
        if i != 100:
            ax.plot(samples_dict[i][0][-1],np.exp(samples_dict[i][1][-1]),'go')
        ax.set_title("GP approximation (%i samples)"%samples_dict[i][0].shape[0])
    if (i%10 != 0) or i == 100:
        alpha_bmc = 0.5
        ax.plot(xplot,np.exp(vb_posterior[i,:]))
        ax.set_title("Variational approximation (%i proposed components)"%(i+1))
    ax.plot(xplot,np.exp(bmc_pred[i,:]),'m',alpha=alpha_bmc)
    ax.set_ylim(bottom=-0.01,top=0.255)
    savefigtitle(fig)
i = 100
fig,ax = plt.subplots()
ax.plot(xplot,np.exp(true_posterior),alpha=0.5)
ax.plot(samples_dict[i][0],np.exp(samples_dict[i][1]),'ro')
ax.plot(xplot,np.exp(bmc_pred[i,:]),'m')
#ax.plot(xplot,np.exp(vb_posterior[i,:]))
ax.set_ylim(bottom=-0.01,top=0.255)
ax.set_title("GP approximation (%i samples)"%samples_dict[i][0].shape[0])
savefigtitle(fig)

fig,ax = plt.subplots()
ax.plot(xplot,np.exp(true_posterior),alpha=0.5)
#ax.plot(xplot,np.exp(bmc_pred[i,:]),'m',alpha=0.5)
ax.plot(xplot,np.exp(vb_posterior[i,:]))
ax.set_ylim(bottom=-0.01,top=0.255)
ax.set_title("Variational approximation (%i proposed components)"%(i+1))
savefigtitle(fig)

#    ax3i.set_title("Step %i"%i)




