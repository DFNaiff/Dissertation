# -*- coding: utf-8 -*-
import functools

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

from source_problem_1d import SourceProblem,matrix_bandification
#%%
np.random.seed(100)
def source(x,t,x0,rho,q0,Ts):
    if t >= Ts:
        return np.zeros_like(x)
    else:
        return q0*np.exp(-(x-x0)**2/(2*rho**2))

def compute_log_likelihood(x0=0.3,Ts=0.3,q0=6.366,rho=0.05,
                             alpha=1e-2,beta=1e-2):
    data = np.load("source_1d_measurements.npz")
    dt = 0.001
    source_fn = functools.partial(source,x0=x0,rho=rho,q0=q0,Ts=Ts)
    model = SourceProblem(1.0,1.0,source_fn)
    model.discretize(100,dt)
    model.initialize()
    model.make_mass_matrix()
    
    ulist = []
    tlist = []
    ulist.append(model.u.copy().flatten())
    tlist.append(0.0)
    T = 0.4
    Nt = int(np.ceil(T/dt))
    
    measurement_times = [0.075,0.15,0.225,0.3,0.4]
    left_measurements = []
    right_measurements = []
    for i in range(Nt):
        model.step()
        ulist.append(model.u.copy().flatten())
        tlist.append(model.t)
        if np.round(model.t,decimals=5) in measurement_times:
            left_measurements.append(float(model.u[0]))
            right_measurements.append(float(model.u[-1]))
    ll = 0.0
    sigma = beta/alpha
    nu = 2*alpha
    logt = lambda x,mu,sigma,nu : -0.5*(nu+1)*np.log(1+1/nu*((x-mu)/sigma)**2)
    for i,t in enumerate(data["times"]):
        ll += logt(left_measurements[i],data["left"][i],sigma,nu)
        ll += logt(right_measurements[i],data["right"][i],sigma,nu)
    return ll

if __name__ == "__main__":
    def sigmoid(x):
        return 1.0/(1.0+np.exp(-x))
    def dsigmoid(x):
        return np.exp(x)/((1+np.exp(x))**2)
    x = np.linspace(-10.0,10.0,101)
    l = []
    for xx in x:
#        l.append(compute_log_likelihood_2(xx))
        l.append(compute_log_likelihood(sigmoid(xx)) + np.log(dsigmoid(xx)))
    plt.plot(x,l)
#    print(compute_log_likelihood(0.23))
#    x = np.linspace(0.01,0.5,24)
#    t = np.linspace(1.0,11.0,24)
#    X,T = np.meshgrid(x,t)
#    ll = np.zeros_like(X)
#    for i,xx in enumerate(x):
#        for j,tt in enumerate(t):
#            ll[j,i] = compute_log_likelihood_2(xx,tt)
#        print(i)
#    from mpl_toolkits.mplot3d import Axes3D
#    #%%
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_surface(X,T,np.exp(ll))