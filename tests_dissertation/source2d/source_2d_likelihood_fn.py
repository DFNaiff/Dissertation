# -*- coding: utf-8 -*-
import functools

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from source_problem_2d import SourceProblemCN,SourceProblemIE,matrix_bandification
#%%
def compute_log_likelihood(x0,y0,rho,q0,Ts):
    data = np.load("source_2d_data.npz")
    def source(x,y,t,x0,y0,rho,q0,Ts):
        x_,y_ = x.reshape(-1,1),y.reshape(-1,1)
        res_ = q0*np.exp(-((x_-x0)**2 + (y_-y0)**2)/(2*rho**2))*(t<Ts)
        res = res_.sum(axis=-1)
        return res
    
    dt = data["dt"]
    T = data["T"]
    Nl = 20
    Nt = int(np.ceil(T/dt))
    
    source_fn = functools.partial(source,x0=x0,y0=y0,rho=rho,q0=q0,Ts=Ts)
    model = SourceProblemIE(1.0,source_fn)
    model.discretize(Nl,dt)
    model.initialize()
    model.make_mass_matrix()
    
    measurement_grid_indexes = np.array((0,Nl//2,Nl))
    indx_,indy_ = np.meshgrid(measurement_grid_indexes,
                              measurement_grid_indexes)
    measurement_indexes = [indx_.flatten(),indy_.flatten()]
    measurement_times_ind = data["indst"]
    true_values_array = []
    for i in range(Nt):
        model.step()
        if i in measurement_times_ind:
            true_values = model.uxy[measurement_indexes[0],measurement_indexes[1]]
            true_values_array.append(true_values)
        
    true_values = np.array(true_values)
    measurements = data["measurements"]
    alpha=0.01
    beta=0.01
    sigma = beta/alpha
    nu = 2*alpha
    logt = lambda x,mu,sigma,nu : -0.5*(nu+1)*np.log(1+1/nu*((x-mu)/sigma)**2)
    ll = np.sum(logt(true_values,measurements,sigma,nu))
#    sigma = 0.1
#    logmvn = lambda x,mu,sigma : -0.5*(x-mu)**2/(sigma**2)
#    ll = np.sum(logmvn(true_values,measurements,sigma))
    print('ok')
    return ll

if __name__ == "__main__":
    def sigmoid(x,a=0,b=1):
        return (b-a)*1.0/(1.0+np.exp(-x)) + a
    def dsigmoid(x,a=0,b=1):
        return (b-a)*np.exp(x)/((1+np.exp(x))**2)
    def exp(x):
        return np.exp(x)
    def unwarped_logjoint_np(x0,y0,Ts,q0,rho):
        rho = 0.05
        Ts = 0.3
        q0 = 6.366197723675814
        ll = compute_log_likelihood(x0,y0,rho,q0,Ts)
    #    ll += -np.log(1+(q0/10.0)**2)
    #    ll += -np.log(1+(rho/0.1)**2)
        return ll

    def logjoint_np_u(x):
    #    x0,y0,Ts,q0,rho = x[0],x[1],x[2],x[3],x[4]
        x0,y0,Ts,q0,rho = x[0],x[1],0.0,0.0,0.0
        return unwarped_logjoint_np(x0,y0,rho,q0,Ts)

    def logjoint_np(x):
    #    x0,y0,Ts,q0,rho = x[0],x[1],x[2],x[3],x[4]
        x0,y0,Ts,q0,rho = x[0],x[1],0.0,0.0,0.0
        ll = unwarped_logjoint_np(sigmoid(x0),sigmoid(y0),
                                  sigmoid(Ts,b=0.4),
                                  exp(q0),exp(rho)) + \
             np.log(dsigmoid(x0)) + np.log(dsigmoid(y0))
    #         np.log(dsigmoid(Ts,b=0.4)) + \
    #         np.log(dexp(q0)) + np.log(dexp(rho))
        return ll
    
    Nplot = 11
#    x,y = np.linspace(-10,10.0,Nplot),np.linspace(-10,10.0,Nplot)
    x,y = np.linspace(0,1,Nplot),np.linspace(0,1,Nplot)
    X,Y = np.meshgrid(x,y)
    XY_ = np.stack([X,Y],axis=-1).reshape(-1,2)
    Ztrue_ = np.array([logjoint_np_u(xy) for xy in XY_]).reshape(-1,1)
    Ztrue = Ztrue_.reshape(*X.shape)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot_surface(X,Y,np.exp(Ztrue+16))
    plt.show()