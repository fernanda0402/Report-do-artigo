#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:12:46 2023

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


# BAIXANDO O ARQUIVO 

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
x_gapp = z
y_gapp = fs8
e = sig_fs8

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0
xmax = 1.0
nstar = 200

# initial values of the hyperparameters of the squared-exponential covariance function
initheta = [0.5, 0.3]


# initialization of the Gaussian Process
#g = dgp.DGaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar))
g = dgp.DGaussianProcess(x_gapp,y_gapp,e,cXstar=(xmin, xmax, nstar),
                        prior=invgamma, priorargs=(4, 1.5),
                        grad='False')

# training of the hyperparameters and reconstruction of the function
(drec, theta) = g.dgp(thetatrain=initheta)

# the second and third derivatives use g.d2gp() and g.d3gp()

xi     = drec[:, 0]
y_pred = drec[:, 1]
sigma  = drec[:, 2]

y_pred_95_less = y_pred - 1.9600*sigma
y_pred_95_plus = y_pred + 1.9600*sigma


# salvando os dados reconstruídos

dF = xi, y_pred, sigma
#np.savetxt('dfs8_recon_bold_gapp.csv', np.transpose(dF), delimiter=', ')



# MODELO LCDM
cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 1.0, 1000)

a = 1. / (1. + zlcdm)

fs8_lcdm = ccl.growth_rate(cosmo, a)*0.812*ccl.growth_factor(cosmo, a)

dfs8_lcdm = np.gradient(fs8_lcdm, zlcdm)



# Plot the function, the prediction and the 95% confidence interval 
plt.figure()
plt.tick_params(labelsize=14,color='red')
plt.plot(xi, y_pred, color='green', label='Prediction', linestyle="-")
plt.plot(zlcdm, dfs8_lcdm, color='red', label='$\Lambda$CDM')
plt.fill_between(xi, y_pred - 1.9600 * sigma, y_pred + 1.9600 * sigma, alpha=.5, fc='lightgreen')
plt.fill_between(xi, y_pred - 1.00 * sigma, y_pred + 1.00 * sigma, alpha=.5, color = 'forestgreen')

# legenda, label e título
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$df\sigma_8/dz$', fontsize=16)
plt.legend(loc='best')
#plt.savefig('dfs8_recon_gapp_bold.pdf', format='pdf', bbox_inches='tight')
plt.show()

