# -*- coding: utf-8 -*-
import functools

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

from source_problem_1d import SourceProblem,matrix_bandification
#%%
np.random.seed(100)
if __name__ == "__main__":
    def source(x,t,x0,rho,q0,Ts):
        if t >= Ts:
            return np.zeros_like(x)
        else:
            return np.sum(q0*np.exp(-(x-x0)**2/(2*rho**2)),
                          axis=-1,keepdims=True)
    rho = 0.05
    Ts = 0.3
    x0 = np.array([0.1,0.3,0.4,0.6,0.7,0.9])
    q0 = np.array([3.0,5.0,7.0,])
    dt = 0.001
    source_fn = functools.partial(source,x0=x0,rho=rho,q0=q0,Ts=Ts)
    model = SourceProblem(1.0,1.0,source_fn)
    model.discretize(100,dt)
    model.initialize()
    model.make_mass_matrix()
    
    plt.plot(model.x,source_fn(model.x,0.0))
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
            left_measurements.append(model.u[0]+0.1*np.random.randn())
            right_measurements.append(model.u[-1]+0.1*np.random.randn())
    left_measurements = np.array(left_measurements)
    right_measurements = np.array(right_measurements)
    measurement_times = np.array(measurement_times)

#    np.savez("source_1d_measurements",
#             times=measurement_times,left=left_measurements.flatten(),
#             right=right_measurements.flatten())
    uarray = np.array(ulist)
    tarray = np.array(tlist)
    X,T = np.meshgrid(model.x,tarray)
    fig,ax = plt.subplots()
    plot = ax.pcolor(X,T,uarray)
    ax.set_xlabel("X")
    ax.set_ylabel("T")
    print(left_measurements)
    print(right_measurements)