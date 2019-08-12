# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def f(x,mu,sigma2):
    return 1/np.sqrt(2*np.pi*sigma2)*np.exp(-0.5*(x-mu)**2/sigma2)

mu = 4.0
sigma21 = 0.05
sigma22 = 1.0
sigma23 = 100.0

xplot = np.linspace(0,10,11)
labels = [r"$M_1$",r"$M_2$",r"$M_3$"]
for i,sigma2 in enumerate([sigma21,sigma22,sigma23]):
    yplot = f(xplot,mu,sigma2)
    yplot = yplot/sum(yplot)
    plt.plot(xplot,f(xplot,mu,sigma2),'o',label=labels[i],alpha=0.8)
plt.xticks([3],[r"$\mathcal{D}$"])
plt.xlabel(r"$\mathcal{D}'$")
plt.legend()
plt.savefig("/home/danilo/Danilo/Dissertação/Tex/figs/occamfig")