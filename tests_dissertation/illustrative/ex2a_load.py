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
from scipy.special import gamma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.variational_boosting_bmc import VariationalBoosting
from src import vb_utils
from src import sampling

np.random.seed(100)
torch.manual_seed(100)

training_interval = 1
data = np.load("ex2a_jopt%i_data/tracking.npz"%training_interval,allow_pickle=True)
means = data["means"]
covs = data["covs"]
elbo = data["elbo"]
steps = data["steps"]
time = data["time"]
true_posterior = data["true_posterior"]
bmc_pred = data["bmc_pred"]
vb_posterior = data["vb_posterior"]
xplot = data["xplot"]

fig1,ax1 = plt.subplots()
color = 'tab:blue'
ax1.plot(steps,np.log10(means),color=color,marker='.',linestyle=' ')
ax1.set_xlabel("iteration")
ax1.set_ylabel(r"$\log_{10}|\mu - \mu_0|_2$",color=color)
ax1.set_ylim(bottom=-2.5,top=1.0)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.plot(steps,np.log10(covs/25.0316),color=color,marker='x',linestyle=' ')
ax2.set_ylabel(r"$\log_{10} |\sigma^2 - \sigma^2_0|_2/|\sigma^2_0|$",color=color)
ax2.set_ylim(bottom=-4.0,top=1.0)
ax2.tick_params(axis='y', labelcolor=color)


fig1.savefig("/home/danilo/Danilo/Dissertação/Tex/figs/dmcil1ati%i.png"%training_interval)

fig3,ax3 = plt.subplots()
#inds_of_interest = [0,5,10,1]]
inds_of_interest = [50]

for i,ind in enumerate(inds_of_interest):
    alpha = 0.8*(i+1)/len(inds_of_interest)+0.2
    ax3.plot(xplot,np.exp(vb_posterior[ind,:]),linestyle='--',alpha=1.0,
             label=" Variational distribution")
    
ax3.plot(xplot,np.exp(true_posterior),"b-",label="True distribution")
ax3.plot(xplot,np.exp(bmc_pred),"m--",label="GP prediction")
ax3.legend()
ax3.set_xlabel("x")
ax3.set_ylabel("f(x)")
fig3.savefig("/home/danilo/Danilo/Dissertação/Tex/figs/convgraphil1ati%i.png"%training_interval)
fig4,ax4 = plt.subplots()
cumtime = np.cumsum(time)
ax4.plot(steps,cumtime,'ro')
ax4.set_xlabel("iteration")
ax4.set_ylabel("time (s)")
ax4.set_title("Running time for algorithm")
ax4.set_ylim(bottom=0.0,top=280.0)
