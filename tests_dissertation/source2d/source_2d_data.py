# -*- coding: utf-8 -*-
import functools

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt

from source_problem_2d import SourceProblemCN,SourceProblemIE,matrix_bandification
#%%
np.random.seed(100)
def source(x,y,t,x0,y0,rho,q0,Ts):
    x_,y_ = x.reshape(-1,1),y.reshape(-1,1)
    res_ = q0*np.exp(-((x_-x0)**2 + (y_-y0)**2)/(2*rho**2))*(t<Ts)
    res = res_.sum(axis=-1)
    return res
#        return (x+y)*0+10.0
plotting = True

rho = 0.05
Ts = 0.3
x0 = 0.09
y0 = 0.23
q0 = 6.366197723675814

noise = 0.1

dt = 0.005
T = 0.4
Nl = 100

measurement_times = np.array([0.075,0.15,0.225,0.3,0.4])
#measurement_times = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.39])
measurement_grid_indexes = np.array((0,Nl//2,Nl))
indx_,indy_ = np.meshgrid(measurement_grid_indexes,
                          measurement_grid_indexes)
measurement_indexes = [indx_.flatten(),indy_.flatten()]
measurements = []
measurement_times_ind = []

source_fn = functools.partial(source,x0=x0,y0=y0,rho=rho,q0=q0,Ts=Ts)
model = SourceProblemIE(1.0,source_fn)
model.discretize(Nl,dt)
model.initialize()
model.make_mass_matrix()

Nt = int(np.ceil(T/dt))
print(T)
print(Nt)

for i in range(Nt):
    model.step()
    if np.any(np.abs(model.t-measurement_times)<1e-5):
        measurement_times_ind.append(i)
        true_values = model.uxy[measurement_indexes[0],measurement_indexes[1]]
        measurements.append(true_values+noise*np.random.randn(*true_values.shape))
        print(model.t)
        print(true_values)
    if (i+1)%2 == 0 and plotting:
        fig,ax = plt.subplots()
        s = model.source_function(model.X,model.Y,model.t).reshape(model.Nlp,model.Nlp)
        s = model.uxy
        plot = ax.pcolor(model.X,model.Y,s)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("%.3f"%model.t)
measurements = np.array(measurements)
np.savez("source_2d_data",times=measurement_times,indst=measurement_times_ind,
         indsx = measurement_indexes,measurements=measurements,T=T,dt=dt,Nl=Nl)