# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,"../../src")
sys.path.insert(0,"../credibleinterval")
import math
import functools
import time

import torch
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
import pymc3

from credible_estimator import credible_ball_estimator
#from src.variational_boosting_bmc import VariationalBoosting
#from src import vb_utils
#from src import sampling

def sigmoid(x,a=0,b=1):
    return (b-a)*1.0/(1.0+np.exp(-x)) + a
def exp(x):
    return np.exp(x)

np.random.seed(100)
torch.manual_seed(100)

tag = 1
data = np.load("testheat1b/tracking.npz",allow_pickle=True)
elbo = data["elbo"]
steps = data["steps"]
time = data["time"]

#fig4,ax4 = plt.subplots()
#cumtime = np.cumsum(time)
#ax4.plot(steps,cumtime,'ro')
#ax4.set_xlabel("iteration")
#ax4.set_ylabel("time (s)")
#ax4.set_title("Running time for algorithm")

distrib = torch.load("testheat1b/mvn%i"%100)
samples = distrib.sample(5000).numpy()
samples[:,0] = sigmoid(samples[:,0])
samples[:,1] = sigmoid(samples[:,1],b=0.4)
samples[:,2] = exp(samples[:,2])
samples[:,3] = exp(samples[:,3])

for i in range(4):
    print("Data %i"%i)
    mean = samples[:,i].mean()
    print(pymc3.stats.hpd(samples[:,i],0.3))

#names = [r"$x_0$",r"$t_s$",r"$q_0$",r"$\rho$"]
#datadict = dict([(names[i],samples[:,i]) for i in range(4)])
#dataframe = pd.DataFrame(datadict)
#
#lims = [(-0.2,1.2),
#        (-0.2,0.5),
#        (-0.1,21.0),
#        (-0.1,2.1)]
    
#g = sns.PairGrid(dataframe)
#def set_lims_pairgrid(g,lims):
#    shape = g.axes.shape
#    for i in range(shape[0]):
#        for j in range(shape[1]):
#            if i == j:
#                g.axes[i,j].set_xlim(lims[i])
#            elif i > j:
#                g.axes[i,j].set_xlim(lims[j])
#                g.axes[i,j].set_ylim(lims[i])
##            else:
##                g.axes[i,j].set_xlim(lims[i])
##                g.axes[i,j].set_ylim(lims[j])
#g.map_diag(sns.kdeplot)
#g.map_lower(sns.kdeplot, n_levels=10);
#set_lims_pairgrid(g,lims)
#for i, j in zip(*np.triu_indices_from(g.axes, 1)):
#    g.axes[i, j].set_visible(False)
#g.savefig("../../tex/figs/sourceproblemhistogramsvb")