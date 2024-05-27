#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:45:05 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from gapp import gp, dgp
import time
from scipy.special import gamma
from numpy import exp
from gapp import dgp, covariance

ti = time.time()

plt.rcParams['text.usetex'] = True


# baixando os dados
data = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/fsig8_bold_data.dat')

z = data[:, 0]
fs8 = data[:, 1]
sig_fs8 = data[:, 2]


# DEFININDO INVGAMMA
def invgamma(x, a, b):
    x = x[1]
    p = b**a/gamma(a) * x**(-1 - a) * exp(-b/x)

    return p


##################### PROCESSO GAUSSIANO GAPP ###########################

# nomeando
x_gapp = z
y_gapp = fs8
e = sig_fs8

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0.0
xmax = 1.0
nstar = 200

# initial values of the hyperparameters of the squared-exponential covariance function
initheta = [0.2, 1]

# initialization of the Gaussian Process
#g = gp.GaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar))
g = gp.GaussianProcess(x_gapp,y_gapp,e, cXstar=(xmin, xmax, nstar),
                        prior=invgamma, priorargs=(4, 1.5),
                        grad='False', covfunction=covariance.Matern52)

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.gp(theta=initheta)

xi = rec[:, 0]

y_pred = rec[:, 1]
sigma  = rec[:, 2]

y_pred_95_less = y_pred - 1.9600*sigma
y_pred_95_plus = y_pred + 1.9600*sigma


plt.plot(xi, y_pred, color='green', label='GP Prediction')
plt.fill_between(xi, y_pred - sigma , y_pred + sigma , alpha=.5, fc='forestgreen', ec='None')
plt.fill_between(xi, y_pred - 1.96*sigma , y_pred + 1.96*sigma , alpha=.5, fc='lightgreen', ec='None')


# legenda, label e título
#plt.ylim(-0.3, 0.7)
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$f\sigma_8(z)$', fontsize=16)
plt.legend(loc='best')
plt.show()





