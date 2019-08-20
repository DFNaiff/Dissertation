# -*- coding: utf-8 -*-

# =============================================================================
# Here we will be testing with mixtures of student-t, 1d
# =============================================================================
import sys
sys.path.insert(0,"../../src")
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

def gskl(mu1,mu2,cov1,cov2):
    d = len(mu1)
    term1 = -2*d
    term2 = np.trace(np.linalg.inv(cov1)@cov2 + np.linalg.inv(cov2)@cov1)
    dmu = (mu1-mu2).reshape(1,-1)
    term3 = dmu@(np.linalg.inv(cov1)+np.linalg.inv(cov2))@dmu.reshape(-1,1)
    return 0.5*(term1+term2+term3)

gskl_list = []
master_name = "ex5b"
fig1,ax1 = plt.subplots()
fig2,ax2 = plt.subplots()
for dim in [3,6,10]:
    folder_name = "%s_d%i"%(master_name,dim)
    distribdata = np.load("%s/distribdata.npz"%folder_name,allow_pickle=True)
    data = np.load("%s/tracking.npz"%folder_name,allow_pickle=True)
    means = data["means"]
    covs = data["covs"]
    elbo = data["elbo"]
    steps = data["steps"]
    ax1.plot(steps,np.log10(means),marker='.',linestyle=' ',label="D=%i"%dim)
    ax2.plot(steps,np.log10(covs/np.linalg.norm(data["true_cov"])),label="D=%i"%dim,marker='x',linestyle=' ')
    distrib = torch.load("%s/vbdistrib"%folder_name)
    mu1 = distrib.mean().numpy()
    cov1 = distrib.cov().numpy()
    mu2 = data["true_mean"]
    cov2 = data["true_cov"]
    gskl_list.append(gskl(mu1,mu2,cov1,cov2))
ax1.set_xlabel("iteration")
ax1.set_ylabel(r"$\log_{10}||\mu - \mu_0||_2$")
ax1.set_ylim(bottom=-2.5,top=1.0)
ax1.legend()
ax1.set_xlabel("iteration")
ax2.set_ylabel(r"$\log_{10}||\Sigma - \Sigma_0||_F/||\Sigma_0||_F$")
ax2.set_ylim(bottom=-2.0,top=1.0)
ax2.legend()
fig1.savefig("/home/danilo/Danilo/Dissertação/Tex/figs/%s_mean"%master_name)
fig2.savefig("/home/danilo/Danilo/Dissertação/Tex/figs/%s_cov"%master_name)
print(gskl_list)
#
#
#distribdata = np.load("%s/distribdata.npz"%folder_name,allow_pickle=True)
#data = np.load("%s/tracking.npz"%folder_name,allow_pickle=True)
#means = data["means"]
#covs = data["covs"]
#
#elbo = data["elbo"]
#steps = data["steps"]
#fig1,ax1 = plt.subplots()
#color = 'tab:blue'
#ax1.plot(steps,np.log10(means),color=color,marker='.',linestyle=' ')
#ax1.set_xlabel("iteration")
#ax1.set_ylabel(r"$\log_{10}||\mu - \mu_0||_2$",color=color)
#ax1.set_ylim(bottom=-2.5,top=1.0)
#ax1.tick_params(axis='y', labelcolor=color)
#ax2 = ax1.twinx()
#color = 'tab:red'
#ax2.plot(steps,np.log10(covs/np.linalg.norm(data["true_cov"])),color=color,marker='x',linestyle=' ')
#ax2.set_ylabel(r"$\log_{10}||\Sigma - \Sigma_0||_F/||\Sigma_0||_F$",color=color)
#ax2.set_ylim(bottom=-4.0,top=1.0)
#ax2.tick_params(axis='y', labelcolor=color)
#fig1.savefig("/home/danilo/Danilo/Dissertação/Tex/figs/%s_meancov"%folder_name)
##%%
#def gskl(mu1,mu2,cov1,cov2):
#    d = len(mu1)
#    term1 = -2*d
#    term2 = np.trace(np.linalg.inv(cov1)@cov2 + np.linalg.inv(cov2)@cov1)
#    dmu = (mu1-mu2).reshape(1,-1)
#    term3 = dmu@(np.linalg.inv(cov1)+np.linalg.inv(cov2))@dmu.reshape(-1,1)
#    return 0.25*(term1+term2+term3)
#distrib = torch.load("%s/vbdistrib"%folder_name)
#mu1 = distrib.mean().numpy()
#cov1 = distrib.cov().numpy()
#mu2 = data["true_mean"]
#cov2 = data["true_cov"]
#print(gskl(mu1,mu2,cov1,cov2))