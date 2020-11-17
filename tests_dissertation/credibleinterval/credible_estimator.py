# -*- coding: utf-8 -*-

import numpy as np
"""
    Stochastic bisection algorithm, 
    as described in
    "Probabilistic bisection converges almost 
     as quickly as stochastic approximation,
     Peter I. Frazier, Shane G. Henderson, Rolf Waeber"
"""

def credible_ball_estimator(sampler,p,x0,N=1000,delta_step=1.0,maxiter_warming=100,
                            maxiter=1000,tol=1e-2,maxdrift=1000,gamma=0.9,
                            verbose=0):
    """
        sampler(N) : returns (N,D) samples
        p : probability that we want to P(||X - x0|| < r) equals to. Must be less than 0.99
        x0 : x0 in the above descriptio
        N : number of samples per step
        delta_step: step increase in warming
        maxiter_warming: maximum iterations in warming
        gamma : gamma factor for drift test, as described in the article
        maxiter : maximum number of iterations of algorithm
        tol : tolerance (NOT IMPLEMENTED YET)
        maxdrift : maximum number of iterations for each drift test
        verbose : frequency of printings of x_m
    """
    #Pre warming
    k = delta_step
    warm_steps = 1
    print("Beginning warming...")
    while True:
        if warm_steps > maxiter_warming:
            raise ValueError("Too many warm steps. Increase delta step")
        p_test = np.mean((sampler(N)**2).sum(axis=-1) < k**2)
        print(p_test)
        print(p)
        if p_test > p + (1-p)/3.0:
            break
        else:
            k += delta_step
            warm_steps += 1
    print("k = %f"%k)
    def f(rtilde): #r = k*rtilde
        return p - np.mean((sampler(N)**2).sum(axis=-1) < (k*rtilde)**2)
    print(f(0.0))
    print(f(1.0))
    print("Beginning calculation...")
    rtilde = stochastic_bisection(f,gamma=gamma,
                                  maxiter=maxiter,
                                  maxdrift=maxdrift,
                                  tol=tol,
                                  verbose=verbose)
    r = k*rtilde
    return r,k

def stochastic_bisection(measure,gamma=0.9,maxiter=100,maxdrift=500,tol=1e-3,
                         verbose=0):
    """
        measure : function that takes a scalar as value and returns
                  a noisy measurement of some 1d function f:[0,1] -> R
        gamma : gamma factor for drift test, as described in the article
        maxiter : maximum number of iterations of algorithm
        maxdrift : maximum number of iterations for each drift test
        verbose : frequency of printings of x_m
        tol : tolerance (NOT IMPLEMENTED YET)
    """
    pc = 1.0 - gamma/2
    p0 = pc-1e-2
    points = [0.0,1.0]
    values = [0.0,1.0]
    x_m = 0.5
    x_r0 = x_m
    running_alpha = 0.1
    if verbose == 0:
        verbose = maxiter+1
    for n in range(maxiter):
        sign_func = lambda : np.sign(measure(x_m))
        z_m = _drift_test(sign_func,gamma,maxdrift)
        if z_m == -1:
            p_update = p0
        elif z_m == 1:
            p_update = 1-p0
        else:
            continue
        points,values = _update_cdf(x_m,p_update,points,values)
        x_m = _get_median(points,values)
        x_r = x_r0 + running_alpha*(x_m-x_r0)
        if n >= 10 and np.abs(x_r-x_r0) <= tol:
            break
        else:
            x_r0 = x_r
        if (n+1)%verbose == 0:
            print(x_r,x_m)
    print("Finished")
    return x_r

def _drift_test(sign_func,gamma,maxiter):
    s0 = 0.0
    m = 1
    while True:
        k = np.sqrt(2*m*np.log(m+1)-np.log(gamma))
        s0 += sign_func()
        if s0 >= k:
            return 1
        elif s0 <= -k:
            return -1
        elif m >= maxiter:
            return 0
        m += 1

def _location_ordered_list(x,L):
    #L : [p0,p1,p2,...,pN], where it's ordered
    #returns : i, where i is the index where p[i-1] <= x < p[i]
    #          if empty list, return 0. 
    #          if x < p[0], return 0. x >= p[-1], return len(p)
    N = len(L)
    if N == 0:
        return 0
    if N == 1:
        return 1 if x >= L[0] else 0
    if x >= L[-1]:
        return N
    elif x < L[0]:
        return 0
    else:
        i = (N-1)//2
        if L[i+1] <= x:
            return i+1 + _location_ordered_list(x,L[i+1:])
        else:
            if L[i] <= x:
                return i+1
            else:
                return _location_ordered_list(x,L[:i+1])

def _update_cdf(x,p,points,values):
    #x \in (0,1). F_n(x) = 1/2
    #p \in (0,1)
    #points : [x0=0,x1,x2,...,xN=1]
    #values : [y0=0,y1,y2,...,yN=1]
    #Those represent the CDF of an uniform by parts. Let the PDF of if be f_N
    #returns: CDF of the distribution with density f_{N+1}(y) = p*f_N(y) if y < x else (1-p)*f_N(y)
    q = 1-p
    p_ = 2*p
    q_ = 2*q
    ind = _location_ordered_list(x,points)
    points_low = points[:ind]
    points_high = points[ind:]
    values_low = values[:ind]
    values_high = values[ind:]
    values_low = list(p_*np.array(values_low))
    values_high = list((p_-q_)/2+q_*np.array(values_high))
    points_new = points_low + [x] + points_high
    values_new = values_low + [p_/2] + values_high
    return points_new,values_new

def _get_median(points,values):
    #get the median of a CDF defined by points and values
    #points : [x0=0,x1,x2,...,xN=1]
    #values : [y0=0,y1,y2,...,yN=1]
    y_median = 0.5
    i = _location_ordered_list(y_median,values)
    xlow,xhigh,ylow,yhigh = points[i-1],points[i],values[i-1],values[i]
    x_median = ((xhigh-xlow)*y_median-(ylow*xhigh-yhigh*xlow))/(yhigh-ylow)
    return x_median