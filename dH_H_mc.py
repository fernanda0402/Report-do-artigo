#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:30:47 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from gapp import gp, dgp
import time

ti = time.time()

plt.rcParams['text.usetex'] = True


# baixando os dados
data_Hz = np.genfromtxt('/home/usuario/Documentos/Dados/CC_Hz_data (cópia).csv', delimiter=', ')

z = data_Hz[:, 0]
H = data_Hz[:, 1]

sig_H = data_Hz[:, 2]



##################### PROCESSO GAUSSIANO GAPP ###########################

# nomeando
x_gapp = z
y_gapp = H
e = sig_H

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0.0
xmax = 1.0
nstar = 200

# initial values of the hyperparameters of the squared-exponential covariance function
initheta = [2.0, 2.0]

# initialization of the Gaussian Process
g = gp.GaussianProcess(x_gapp, y_gapp, e,  cXstar=(xmin, xmax, nstar))

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.gp(theta=initheta)

xi = rec[:, 0]

y_pred = rec[:, 1]
sigma  = rec[:, 2]

y_pred_95_less = y_pred - 1.9600*sigma
y_pred_95_plus = y_pred + 1.9600*sigma

###############################################################################

# nomeando
x_gapp = z
y_gapp = H
e = sig_H

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0.0
xmax = 1.0
nstar = 200

# initial values of the hyperparameters of the squared-exponential covariance function
initheta = [2.0, 2.0]

# initialization of the Gaussian Process
g = dgp.DGaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar))

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.dgp(theta=initheta)

xi = rec[:, 0]

dy_pred = rec[:, 1]
dsigma  = rec[:, 2]

dy_pred_95_less = dy_pred - 1.9600*dsigma
dy_pred_95_plus = dy_pred + 1.9600*dsigma


dH_H = dy_pred / y_pred

plt.plot(xi, dH_H, color='green', label='GP Prediction')

dH_mc = []
for i in range(10000):
                         
    Hi = np.random.normal(y_pred, scale=sigma)
    
    dHi = np.random.normal(dy_pred, scale=dsigma)
    
    dH_mc.append(dHi/Hi)

dH_mc = np.array(dH_mc)   

sigma_dH = []
for i in range(len(xi)):
    
    sigma_dH.append(np.std(dH_mc[:, i]))

sigma_dH = np.array(sigma_dH)    

plt.fill_between(xi, dH_H - sigma_dH, dH_H + sigma_dH, alpha=.5, fc='forestgreen', ec='None')
plt.fill_between(xi, dH_H - 1.96*sigma_dH, dH_H + 1.96*sigma_dH, alpha=.5, fc='lightgreen', ec='None')


G = xi, dH_H, sigma_dH

#np.savetxt('dh_recon.dat', np.transpose(G), delimiter='\t')


#LCDM

import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, 
    h=0.6727, sigma8=0.8120, n_s=0.9649)

zlcdm = np.linspace(0, 1, 1000)
a = 1. / (1. + zlcdm)

Hz = ccl.background.h_over_h0(cosmo, a)

dHz = np.gradient(Hz, zlcdm)


plt.plot(zlcdm, dHz/Hz, color='red', label='$\Lambda$CDM')


# legenda, label e título
plt.ylim(0, 1)
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\mathcal{H}(z)$', fontsize=16)
plt.legend(loc='best')
#plt.savefig('dh_h_mc.pdf', format='pdf', bbox_inches='tight')
plt.show()


# TIMER

tf = time.time()

tempo = tf - ti

print(tempo)



################################ E(z) ###########################

Ez = y_pred / y_pred[0]
eEz = sigma / y_pred[0]

G = xi, Ez, eEz

#np.savetxt('E_recon.dat', np.transpose(G), delimiter='\t')






