# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as spla

#%%
def band_diag(N,k,t):
    return t*np.diag(np.ones(N-abs(k)),k=k)

def matrix_bandification(M,bands):
    bands = np.sort(bands)[::-1]
    Mb = np.zeros((len(bands),M.shape[0]))
    for i,b in enumerate(bands):
        if b > 0:
            Mb[i,b:] = np.diag(M,k=b)
        elif b == 0:
            Mb[i,:] = np.diag(M)
        else:
            Mb[i,:b] = np.diag(M,k=b)
    return Mb

class SourceProblem(object):
    """
        Solves the differential equation
        du/dt = d2u/dx^2 + f(x,t), on B; du/dn = 0, dB; u=0,t=0
    """
    def __init__(self,L,alpha,source_function):
        self.alpha = alpha
        self.L = L
        self.source_function = source_function
    
    def discretize(self,Nl,dt):
        self.Nl = Nl
        self.dl = self.L/self.Nl
        self.dt = dt
    
    def initialize(self):
        self.x = np.linspace(0,1,self.num_nodes).reshape(-1,1)
        self.u = np.zeros_like(self.x)
        self.t = 0
        
    def make_mass_matrix(self):
        dl = self.dl
        dt = self.dt
        alpha = self.alpha
        M = np.zeros((self.num_nodes,self.num_nodes))
        M += band_diag(self.num_nodes,0,(1 + alpha*2*dt/(dl**2)))
        M += band_diag(self.num_nodes,1,-dt*alpha/(dl**2))
        M += band_diag(self.num_nodes,-1,-dt*alpha/(dl**2))
        M[0,:2] = [1.0,-1.0]
        M[-1,-2:] = [-1.0,1.0]
        self.M = M
        self.Mb = matrix_bandification(self.M,[-1,0,1])
        self.l_and_u = (1,1)
    
    def make_force_matrix(self):
        f1 = self.source_function(self.x,self.t)
        f = self.dt*f1 + self.u
        f[0] = 0.0
        f[-1] = 0.0
        return f
    
    def step(self):
        f = self.make_force_matrix()
        self.u = spla.solve_banded((1,1),self.Mb,f)
#        self.u = spla.solve(self.M,f)
        self.t += self.dt
        
    @property
    def num_nodes(self):
        return self.Nl+1