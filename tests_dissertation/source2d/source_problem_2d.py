# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla
import scipy.sparse as spsparse
from scipy.sparse.linalg import spsolve

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

class SourceProblemCN(object):
    """
        Solves the differential equation
        du/dt = d2u/dx^2 + f(x,t), on B; du/dn = 0, dB; u=0,t=0
    """
    def __init__(self,L,source_function):
        self.L = L
        self.source_function = source_function
    
    def discretize(self,Nl,dt):
        self.Nl = Nl
        self.dl = self.L/Nl
        self.dt = dt

    def initialize(self):
        self.x = np.linspace(0,1,self.Nlp).reshape(-1,1)
        self.y = np.linspace(0,1,self.Nlp).reshape(-1,1)
        self.X,self.Y = np.meshgrid(self.x,self.y)
        self.u = np.zeros(self.num_nodes)
        self.t = 0

    def make_mass_matrix(self):
        dl = self.dl
        dt = self.dt
        Q = np.zeros((self.num_nodes,self.num_nodes))
        Q += band_diag(self.num_nodes,0,(-4/(dl**2)))
        Q += band_diag(self.num_nodes,1,1/(dl**2))
        Q += band_diag(self.num_nodes,-1,1/(dl**2))
        Q += band_diag(self.num_nodes,self.Nlp,1/(dl**2))
        Q += band_diag(self.num_nodes,-self.Nlp,1/(dl**2))
        M = np.eye(self.num_nodes) - 0.5*dt*Q
        F = np.eye(self.num_nodes) + 0.5*dt*Q
        #Boundaries
        M[self.indm_down,:] = 0.0
        M[self.indm_down,self.indm_down] = 1.0
        M[self.indm_down,self.indm_down+self.Nlp] = -1.0
        M[self.indm_up,:] = 0.0
        M[self.indm_up,self.indm_up] = 1.0
        M[self.indm_up,self.indm_up-self.Nlp] = -1.0
        M[self.indm_left,:] = 0.0
        M[self.indm_left,self.indm_left] = 1.0
        M[self.indm_left,self.indm_left+1] = -1.0
        M[self.indm_right,:] = 0.0
        M[self.indm_right,self.indm_right] = 1.0
        M[self.indm_right,self.indm_right-1] = -1.0
        #Corners
        M[0,:] = 0.0;M[0,0]=1.0;
        M[0,1]=-0.5;M[0,self.Nlp]=-0.5
        M[self.Nl,:] = 0.0;M[self.Nl,self.Nl] = 1.0;
        M[self.Nl,self.Nl-1] = -0.5;M[self.Nl,self.Nl+self.Nlp]=-0.5
        M[self.Nl*self.Nlp,:] = 0.0;M[self.Nl*self.Nlp,self.Nl*self.Nlp] = 1.0
        M[self.Nl*self.Nlp,self.Nl*self.Nlp+1]=-0.5;M[self.Nl*self.Nlp,self.Nl*self.Nlp-self.Nlp]=-0.5
        M[self.Nlp**2-1,:] = 0.0;M[self.Nlp**2-1,self.Nlp**2-1] = 1.0
        M[self.Nlp**2-1,self.Nlp**2-1-1] = -0.5;M[self.Nlp**2-1,self.Nlp**2-1-self.Nlp] = -0.5
        Ms = spsparse.dia_matrix(M)
        Fs = spsparse.dia_matrix(F)
        self.Ms = Ms
        self.Fs = Fs
#        self.M = M
    
    def make_force_matrix(self):
        f1_ = 0.5*(self.source_function(self.X,self.Y,self.t) + \
                   self.source_function(self.X,self.Y,self.t+self.dt))
        f1 = f1_.flatten(order='C')
        f = self.dt*f1 + self.Fs.dot(self.u)
        f[self.indm_down] = 0.0
        f[self.indm_up] = 0.0
        f[self.indm_left] = 0.0
        f[self.indm_right] = 0.0
        f[0] = 0.0
        f[self.Nl] = 0.0
        f[self.Nl*self.Nlp] = 0.0
        f[self.Nlp**2-1] = 0.0
        return f
    
    def step(self):
        f = self.make_force_matrix()
        self.u = spsolve(self.Ms,f)
#        self.u = spla.solve(self.M,f)
        self.t += self.dt

    def inds_to_indm(self,i,j):
        return i + self.Nl*j
    
    def indm_to_inds(self,indm):
        j = indm%self.Nl
        i = indm - j*self.Nl
        return i,j
    
    @property
    def uxy(self):
        return self.u.reshape(self.Nlp,self.Nlp)
    
    @property
    def indm_down(self):
        return np.arange(1,self.Nlp-1)
    
    @property
    def indm_up(self):
        return np.arange(1,self.Nlp-1) + self.Nlp*self.Nl
    
    @property
    def indm_left(self):
        return np.arange(1,self.Nlp-1)*self.Nlp
    
    @property
    def indm_right(self):
        return np.arange(1,self.Nlp-1)*self.Nlp + self.Nlp-1
    
    @property
    def num_nodes(self):
        return (self.Nlp)**2
    
    @property
    def Nlp(self):
        return self.Nl+1

class SourceProblemIE(object):
    """
        Solves the differential equation
        du/dt = d2u/dx^2 + f(x,t), on B; du/dn = 0, dB; u=0,t=0
    """
    def __init__(self,L,source_function):
        self.L = L
        self.source_function = source_function
    
    def discretize(self,Nl,dt):
        self.Nl = Nl
        self.dl = self.L/Nl
        self.dt = dt

    def initialize(self):
        self.x = np.linspace(0,1,self.Nlp).reshape(-1,1)
        self.y = np.linspace(0,1,self.Nlp).reshape(-1,1)
        self.X,self.Y = np.meshgrid(self.x,self.y)
        self.u = np.zeros(self.num_nodes)
        self.t = 0

    def make_mass_matrix(self):
        dl = self.dl
        dt = self.dt
        M = np.zeros((self.num_nodes,self.num_nodes))
        M += band_diag(self.num_nodes,0,(1 + 4*dt/(dl**2)))
        M += band_diag(self.num_nodes,1,-dt/(dl**2))
        M += band_diag(self.num_nodes,-1,-dt/(dl**2))
        M += band_diag(self.num_nodes,self.Nlp,-dt/(dl**2))
        M += band_diag(self.num_nodes,-self.Nlp,-dt/(dl**2))
        #Boundaries
        M[self.indm_down,:] = 0.0
        M[self.indm_down,self.indm_down] = 1.0
        M[self.indm_down,self.indm_down+self.Nlp] = -1.0
        M[self.indm_up,:] = 0.0
        M[self.indm_up,self.indm_up] = 1.0
        M[self.indm_up,self.indm_up-self.Nlp] = -1.0
        M[self.indm_left,:] = 0.0
        M[self.indm_left,self.indm_left] = 1.0
        M[self.indm_left,self.indm_left+1] = -1.0
        M[self.indm_right,:] = 0.0
        M[self.indm_right,self.indm_right] = 1.0
        M[self.indm_right,self.indm_right-1] = -1.0
        #Corners
        M[0,:] = 0.0;M[0,0]=1.0;
        M[0,1]=-0.5;M[0,self.Nlp]=-0.5
        M[self.Nl,:] = 0.0;M[self.Nl,self.Nl] = 1.0;
        M[self.Nl,self.Nl-1] = -0.5;M[self.Nl,self.Nl+self.Nlp]=-0.5
        M[self.Nl*self.Nlp,:] = 0.0;M[self.Nl*self.Nlp,self.Nl*self.Nlp] = 1.0
        M[self.Nl*self.Nlp,self.Nl*self.Nlp+1]=-0.5;M[self.Nl*self.Nlp,self.Nl*self.Nlp-self.Nlp]=-0.5
        M[self.Nlp**2-1,:] = 0.0;M[self.Nlp**2-1,self.Nlp**2-1] = 1.0
        M[self.Nlp**2-1,self.Nlp**2-1-1] = -0.5;M[self.Nlp**2-1,self.Nlp**2-1-self.Nlp] = -0.5
        Ms = spsparse.dia_matrix(M)
        self.Ms = Ms
        self.M = M
    
    def make_force_matrix(self):
        f1_ = self.source_function(self.X,self.Y,self.t)
        f1 = f1_.flatten(order='C')
        f = self.dt*f1 + self.u
        f[self.indm_down] = 0.0
        f[self.indm_up] = 0.0
        f[self.indm_left] = 0.0
        f[self.indm_right] = 0.0
        f[0] = 0.0
        f[self.Nl] = 0.0
        f[self.Nl*self.Nlp] = 0.0
        f[self.Nlp**2-1] = 0.0
        return f
    
    def step(self):
        f = self.make_force_matrix()
        self.u = spsolve(self.Ms,f)
#        self.u = spla.solve(self.M,f)
        self.t += self.dt

    def inds_to_indm(self,i,j):
        return i + self.Nl*j
    
    def indm_to_inds(self,indm):
        j = indm%self.Nl
        i = indm - j*self.Nl
        return i,j
    
    @property
    def uxy(self):
        return self.u.reshape(self.Nlp,self.Nlp)
    
    @property
    def indm_down(self):
        return np.arange(1,self.Nlp-1)
    
    @property
    def indm_up(self):
        return np.arange(1,self.Nlp-1) + self.Nlp*self.Nl
    
    @property
    def indm_left(self):
        return np.arange(1,self.Nlp-1)*self.Nlp
    
    @property
    def indm_right(self):
        return np.arange(1,self.Nlp-1)*self.Nlp + self.Nlp-1
    
    @property
    def num_nodes(self):
        return (self.Nlp)**2
    
    @property
    def Nlp(self):
        return self.Nl+1  
#    @property
#    def M(self):
#        raise NotImplementedError
        

if __name__ == "__main__":
    source = SourceProblem(1.0,1.0,lambda x,t : 0.0)
    source.discretize(10,10)
    source.make_mass_matrix()
#    plt.imshow(source.M)