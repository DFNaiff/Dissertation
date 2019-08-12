# -*- coding: utf-8 -*-

import torch
import numpy as np

from .variational_boosting_bmc import VariationalBoosting
from . import utils


def kl_vb_bmc(vb,nsamples):
    """
        Deprecated. Use vb.kullback_proposal_bmc
    """
    return vb.kullback_proposal_bmc(nsamples)
    

def everything_optimizer(vb,nsamples,**kwargs):
    """
        Deprecated. Use vb.update_full
    """
    lr = kwargs.get("lr",0.1)
    maxiter = kwargs.get("maxiter",500)
    verbose = kwargs.get("verbose",1)
    vb.update_full(nsamples,lr,maxiter,verbose)
    return

class Exp3Bandit(object):
    def __init__(self,options,eta,gamma,nu):
        self.options_list = options
        self.num_options = len(options)
        self.rewards = np.zeros(self.num_options)
        self.eta = eta
        self.gamma = gamma
        self.nu = nu
        self.current_nominee_index = 0
        self.current_prob = 1.0
        
    def select_nominee(self):
        hedge_probs_ = np.exp(self.eta*self.rewards)
        hedge_probs = hedge_probs_/sum(hedge_probs_)
        if np.random.rand() < self.gamma:
            nominee_index = np.random.choice(range(self.num_options))
        else:
            nominee_index = np.random.choice(range(self.num_options),p=hedge_probs)
        self.current_prob = (1-self.gamma)*hedge_probs[nominee_index] + \
                             self.gamma/self.num_options
        self.current_nominee_index = nominee_index
        return self.options_list[nominee_index]
        
    def update_rewards(self,reward):
        new_rewards = np.zeros(self.num_options)
        new_rewards[self.current_nominee_index] = reward/self.current_prob
        self.rewards = (1 - self.nu)*self.rewards + new_rewards