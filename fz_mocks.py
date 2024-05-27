#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:50:27 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from gapp import gp
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = True

import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, 
    h=0.6727, sigma8=0.8120, n_s=0.9649)

# baixando os dados
data_fz = np.genfromtxt('/home/usuario/Documentos/Dados/fz_data.csv', delimiter=', ')

zf = data_fz[:, 0]
fz = data_fz[:, 1]

sig_fz = data_fz[:, 2]

#plt.errorbar(zf, fz, sig_fz, fmt='s', color='blue')

# plt.scatter(z, sig_f)

# def E(x, a, b):
    
#     return (a*x) + b

# from scipy.optimize import curve_fit

# popt, pcov = curve_fit(E, z, sig_f)

# plt.plot(z, E(z, *popt), color='red')

N = len(zf)

HM = []
for i in range(500):

    z = np.linspace(min(zf), max(zf), N)
    
    a = 1. / (1. + z)
    
    f = ccl.growth_rate(cosmo, a)
    
    ef = np.random.normal(0.14+(0.08*z), 0.02)
    
    f = np.random.normal(f, ef)
    
    # plt.tick_params(labelsize=14,color='red')
    # plt.xlabel('$z$', fontsize=16)
    # plt.ylabel('$f(z)$', fontsize=16)
    # plt.errorbar(z, f, ef, fmt='s', color='black')

    zi = np.linspace(0, 1, 200)
    
    fi = ccl.growth_rate(cosmo, 1/(1+zi)) 
    
    # plt.plot(zi, fi, color='red')

    ####################### gaussian process

    # def T(x, a1, a2, a3):
        
    #     return (a1*x*x) + (a2*x) + a3
        
    
    # popt, pcov = curve_fit(T, z, f, sigma=ef)
    
    # a1,a2,a3 = popt
    
    # # plt.plot(zi, T(zi,a1,a2,a3)+0.5)
    
    # def G(x, a1, a2, a3):
        
    #     return 0.5 + (a1*x*x) + (a2*x) + a3
    
    # nomeando
    x_gapp = z
    y_gapp = f
    e = ef
    
    # xmin, xmax and nstar are interpreted as two-dimensional vectors
    xmin = 0
    xmax = 1.0
    nstar = 200
    
    # initial values of the hyperparameters of the squared-exponential covariance function
    #initheta = [2.0, 2.0]
    
    # initialization of the Gaussian Process

    g = gp.GaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar),
                            mu=None)
    
    # training of the hyperparameters and reconstruction of the function
    (rec, theta) = g.gp()
    
    xi = rec[:, 0]
    
    y_pred = rec[:, 1]
    sigma  = rec[:, 2]
    
    y_pred_95_less = y_pred - 1.9600*sigma
    y_pred_95_plus = y_pred + 1.9600*sigma
    
    j = (y_pred-fi)
    
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
plt.ylim(-0.5, +0.5)
#plt.title('$\mu(x)=Ax$', fontsize=16)
plt.tick_params(labelsize=14,color='red')
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$f(z)$-$f^{fid}(z)$', fontsize=16)
plt.hlines(0, min(zi), max(zi), ls='dashed', color='black')
plt.plot(xi, dm, color='green', lw=1) 
plt.fill_between(xi, dm-edm, dm+edm, alpha=0.5, color='forestgreen')
plt.fill_between(xi, dm-1.96*edm, dm+1.96*edm, alpha=0.5, color='lightgreen')
#plt.legend(loc='best', fontsize=12)    
plt.savefig('diff_mu_0_fz.pdf', format='pdf', bbox_inches='tight')    



