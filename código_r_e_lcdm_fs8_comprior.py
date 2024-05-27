#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:04:01 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
import pyccl as ccl
plt.rcParams['text.usetex'] = True
from gapp import gp
from scipy.special import gamma
from numpy import exp


# primeira derivada

from gapp import dgp


data_fs8 = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/fsig8_bold_data.dat')

z = data_fs8[:,0]
fs8 = data_fs8[:,1]
sig_fs8 = data_fs8[:,2]


# DEFININDO INVGAMMA
def invgamma(x, a, b):
    x = x[1]
    p = b**a/gamma(a) * x**(-1 - a) * exp(-b/x)

    return p



# nomeando
x_gapp = z
y_gapp = fs8
e = sig_fs8

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0
xmax = 1.0
nstar = 1000

# initial values of the hyperparameters of the squared-exponential covariance function
initheta = [0.5, 0.3]

# initialization of the Gaussian Process
#g = gp.GaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar))
g = gp.GaussianProcess(x_gapp,y_gapp,e,cXstar=(xmin, xmax, nstar),
                        prior=invgamma, priorargs=(4, 1.5),
                        grad='False')

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.gp(theta=initheta)

xi = rec[:, 0]

y_pred = rec[:, 1]
sigma  = rec[:, 2]

y_pred_95_less = y_pred - 1.9600*sigma
y_pred_95_plus = y_pred + 1.9600*sigma


# salvando os dados reconstruídos

F = xi, y_pred, sigma
#np.savetxt('fs8_recon_gapp.csv', np.transpose(F), delimiter=', ')



# MODELO LCDM


cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 1.0, 1000)

a = 1. / (1. + zlcdm)

fs8_lcdm = ccl.growth_rate(cosmo, a)*0.812*ccl.growth_factor(cosmo, a)



# comparando

dFs8 = y_pred - fs8_lcdm
sig_dFs8 = sigma


# Plot the function, the prediction and the 95% confidence interval
plt.figure()
plt.ylim(-0.3,0.3)
plt.xlim(0,1.0)
plt.tick_params(labelsize=14, color='purple')
plt.plot(xi, dFs8, color = 'green', linestyle="--")
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([dFs8 - 1.00 * sigma,
                        (dFs8 + 1.00 * sigma)[::-1]]),
         alpha=.5, color = 'forestgreen', ec='None')
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([dFs8 - 1.9600 * sigma,
                        (dFs8 + 1.9600 * sigma)[::-1]]),
         alpha=.5, color = 'lightgreen', ec='None')

# legenda, label e título
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$f\sigma_8(z) - f\sigma_8 ^{fid}(z)$', fontsize=15)
#plt.legend(loc='best')
#plt.savefig('dFs8_recon_comprior_gapp.pdf', format='pdf', bbox_inches='tight')
plt.show()