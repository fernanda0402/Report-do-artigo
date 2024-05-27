#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:40:53 2024

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
data_Hz = np.genfromtxt('/home/usuario/Documentos/Dados/CC_Hz_data (cópia).csv', delimiter=', ')

zH = data_Hz[:, 0]
Hz = data_Hz[:, 1]

sig_Hz = data_Hz[:, 2]

# plt.errorbar(zH, Hz, sig_Hz, fmt='o')

# plt.scatter(zH, sig_Hz)

# def E(x, a, b):
    
#     return (a*x) + b

# popt, pcov = curve_fit(E, zH, sig_Hz)

# plt.plot(zH, E(zH, *popt), color='red')


N = len(zH)

HM = []
for i in range(500):

    z = np.linspace(min(zH), max(zH), N)
    
    a = 1. / (1. + z)
    
    H = 67.27 * ccl.background.h_over_h0(cosmo, a)
    
    eH = np.random.normal(14.28+(10.34*z), 0.1)
    
    H = np.random.normal(H, eH)
    
    # plt.tick_params(labelsize=14,color='red')
    # plt.xlabel('$z$', fontsize=16)
    # plt.ylabel('$H(z)$', fontsize=16)
    # plt.errorbar(z, H, eH, fmt='s', color='black')

    zi = np.linspace(0, 1.0, 200)
    
    Hi = 67.27*ccl.background.h_over_h0(cosmo, 1/(1+zi)) 
    
    # plt.plot(zi, Hi, color='red')

    ####################### gaussian process

    # def T(x, a1, a2, a3):
        
    #     return (a1*x*x) + (a2*x) + a3
        
    
    # popt, pcov = curve_fit(T, z, H, sigma=eH)
    
    # a1,a2,a3 = popt
    
    # # plt.plot(zi, T(zi,a1,a2,a3)+0.5)
    
    # def G(x, a1, a2, a3):
        
    #     return 10 + (a1*x*x) + (a2*x) + a3
    
    # nomeando
    x_gapp = z
    y_gapp = H
    e = eH
    
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
    
    HM.append(y_pred-Hi)
    
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
plt.ylim(-50, +50)
#plt.title('$\mu(x)=Ax$', fontsize=16)
plt.tick_params(labelsize=14,color='red')
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$H(z)$-$H^{fid}(z)$', fontsize=16)
plt.hlines(0, min(zi), max(zi), ls='dashed', color='black')
plt.plot(xi, dm, color='green', lw=1) 
plt.fill_between(xi, dm-edm, dm+edm, alpha=.5, color='forestgreen')
plt.fill_between(xi, dm-1.96*edm, dm+1.96*edm, alpha=.5, color='lightgreen')
#plt.legend(loc='best', fontsize=12)    
plt.savefig('diff_mu_0_Hz.pdf', format='pdf', bbox_inches='tight')    



