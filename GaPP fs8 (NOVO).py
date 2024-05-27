#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:54:42 2023

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

# BAIXANDO O ARQUIVO DE DL

data = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/fsig8_bold_data.dat')

z = data[:, 0]
fs8 = data[:, 1]
sig_fs8 = data[:, 2]




# DEFININDO INVGAMMA
def invgamma(x, a, b):
    x = x[1]
    p = b**a/gamma(a) * x**(-1 - a) * exp(-b/x)

    return p

# nomeando
x_gapp = z[z<2]
y_gapp = fs8[z<2]
e = sig_fs8[z<2]

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0
xmax = 1.0
nstar = 200

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



# Plot the function, the prediction and the 95% confidence interval 
plt.figure()
plt.tick_params(labelsize=14, color='purple')
plt.errorbar(x_gapp, y_gapp, e, fmt='o', markersize=5, color='purple', label='Data')
plt.plot(xi, y_pred, color = 'green', label='Prediction', linestyle="-")
plt.plot(zlcdm, fs8_lcdm, label='$\Lambda$CDM', color='red')
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, color = 'lightgreen', ec='None')
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.00 * sigma,
                        (y_pred + 1.00 * sigma)[::-1]]),
         alpha=.5, color = 'forestgreen', ec='None')

# legenda, label e título
plt.xlim(0,1.0)
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$f\sigma_8(z)$', fontsize=15)
plt.legend(loc='best')
#plt.savefig('fs8_recon_gapp.pdf', format='pdf', bbox_inches='tight')
plt.show()
















