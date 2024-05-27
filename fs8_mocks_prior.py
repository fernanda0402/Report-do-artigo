#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:41:44 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from gapp import gp
from scipy.optimize import curve_fit
from scipy.special import gamma
from numpy import exp

plt.rcParams['text.usetex'] = True

import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, 
    h=0.6727, sigma8=0.8120, n_s=0.9649)

# baixando os dados
data_f8z = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/fsig8_bold_data.dat')

zf8 = data_f8z[:, 0]
f8z = data_f8z[:, 1]

sig_f8z = data_f8z[:, 2]



# DEFININDO INVGAMMA
def invgamma(x, a, b):
    x = x[1]
    p = b**a/gamma(a) * x**(-1 - a) * exp(-b/x)

    return p

#plt.errorbar(zf8, f8z, sig_f8z, fmt='s', color='blue')

# plt.scatter(zf8, sig_f8z)

# def E(x, a, b):
    
#     return (a*x) + b

# popt, pcov = curve_fit(E, zf8, sig_f8z)

# plt.plot(zf8, E(zf8, *popt), color='red')


N = len(zf8)

HM = []
for i in range(500):

    z = np.linspace(min(zf8), max(zf8), N)
    
    a = 1. / (1. + z)
    
    f8 = ccl.growth_rate(cosmo, a) * 0.8120 * ccl.growth_factor(cosmo, a)
    
    ef8 = np.random.normal(0.07+(0.02*z), 0.001)
    
    f8 = np.random.normal(f8, ef8)
    
    # plt.tick_params(labelsize=14,color='red')
    # plt.xlabel('$z$', fontsize=16)
    # plt.ylabel('$f\sigma_{8}(z)$', fontsize=16)
    # plt.errorbar(z, f8, ef8, fmt='s', color='black')

    zi = np.linspace(0, 1.0, 200)
    
    f8i = ccl.growth_rate(cosmo, 1/(1+zi))*0.8120*ccl.growth_factor(cosmo,1/(1+zi))
    
    # plt.plot(zi, f8i, color='red')

    ####################### gaussian process

    # def T(x, a1, a2, a3):
        
    #     return (a1*x*x) + (a2*x) + a3
        
    
    # popt, pcov = curve_fit(T, z, f8, sigma=ef8)
    
    # a1,a2,a3 = popt
    
    # # plt.plot(zi, T(zi,a1,a2,a3)+0.5)
    
    # def G(x, a1, a2, a3):
        
    #     return 0.5 + (a1*x*x) + (a2*x) + a3
    
    
    
    # nomeando
    x_gapp = z
    y_gapp = f8
    e = ef8
    
    # xmin, xmax and nstar are interpreted as two-dimensional vectors
    xmin = 0
    xmax = 1.0
    nstar = 200
    
    # initial values of the hyperparameters of the squared-exponential covariance function
    #initheta = [2.0, 2.0]
    
    # initialization of the Gaussian Process

    g = gp.GaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar), prior=invgamma, priorargs=(4, 1.5),
                        grad='False')
    
    # training of the hyperparameters and reconstruction of the function
    (rec, theta) = g.gp()
    
    xi = rec[:, 0]
    
    y_pred = rec[:, 1]
    sigma  = rec[:, 2]
    
    y_pred_95_less = y_pred - 1.9600*sigma
    y_pred_95_plus = y_pred + 1.9600*sigma
    
    j = (y_pred-f8i)
    HM.append(j)
    
    # plt.plot(xi, y_pred, color='blue')
    # plt.fill_between(xi,y_pred-sigma,y_pred+sigma, alpha=0.5, 
    #                   color='blue')
    # plt.fill_between(xi,y_pred-1.96*sigma,y_pred+1.96*sigma, alpha=0.3, 
    #                   color='blue')
    # plt.show()
    

HM = np.array(HM)

dm  = []
edm = [] 
for i in range(len(xi)):
    
    dm.append(np.mean(HM[:, i]))
    edm.append(np.std(HM[:, i]))

dm  = np.array(dm)
edm = np.array(edm)

plt.xlim(0, 1.0)
plt.ylim(-0.2, +0.2)
#plt.title('$\mu(x)=Ax$', fontsize=16)
plt.tick_params(labelsize=14,color='red')
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$f\sigma_{8}(z)$-$f\sigma_{8}^{fid}(z)$', fontsize=16)
plt.hlines(0, min(zi), max(zi), ls='dashed', color='black')
plt.plot(xi, dm, color='green', lw=1) 
plt.fill_between(xi, dm-edm, dm+edm, alpha=0.5, color='forestgreen')
plt.fill_between(xi, dm-1.96*edm, dm+1.96*edm, alpha=0.5, color='lightgreen')
#plt.legend(loc='best', fontsize=12)    
plt.savefig('diff_mu_0_f8z_prior.pdf', format='pdf', bbox_inches='tight')    



