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
ninit = 5

training_interval = 1000
acquisition = ["uncertainty_sampling","prospective","mmlt","mmlt_prospective"][3]
data = np.load("ex3a_acq_%s_data/tracking.npz"%(acquisition),allow_pickle=True)
means = data["means"]
covs = data["covs"]
elbo = data["elbo"]
steps = data["steps"]
true_posterior = data["true_posterior"]
bmc_pred = data["bmc_pred"]
vb_posterior = data["vb_posterior"]
xplot = data["xplot"]
samples = data["samples"]
evaluations = data["evaluations"]

#iters = np.arange(len(dmeans))
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
fig1.savefig("../../tex/figs/dmcil1g_aq_%s.png"%acquisition)

fig3,ax3 = plt.subplots()
#inds_of_interest = [0,5,10,1]]
inds_of_interest = [50]

for i,ind in enumerate(inds_of_interest):
    alpha = 0.8*(i+1)/len(inds_of_interest)+0.2
    ax3.plot(xplot,np.exp(vb_posterior[ind,:]),linestyle='--',alpha=1.0,
             label="Variational distribution")
    ax3.plot(xplot,np.exp(bmc_pred),label="GP prediction")
ax3.plot(xplot,np.exp(true_posterior),"b-",label="True distribution")
ax3.legend()
ax3.set_xlabel("x")
ax3.set_ylabel("f(x)")
fig3.savefig("../../tex/figs/convgraphil1g_aq_%s.png"%acquisition)

fig4,ax4 = plt.subplots()
ax4.plot(samples[:ninit],np.exp(evaluations)[:ninit],'ro',
         samples[ninit:],np.exp(evaluations)[ninit:],'bx')
ax4.set_xlim([-20,20])
ax4.set_xlabel("x")
ax4.set_ylabel("f(x)")
ax4.legend(["Initial sampling","Active sampling"])
#fig4.savefig("../../tex/figs/explopattern1g_aq_%s.png"%acquisition)