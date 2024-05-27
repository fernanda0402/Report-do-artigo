#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:44:59 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from gapp import gp, dgp
import time
from scipy.special import gamma
from numpy import exp

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


############################## DERIVADA DE FS8 #################################################

# nomeando
x_gapp = z
y_gapp = fs8
e = sig_fs8

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0.0
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
(rec, theta) = g.dgp(theta=initheta)

xi = rec[:, 0]

dy_pred = rec[:, 1]
dsigma  = rec[:, 2]

dy_pred_95_less = dy_pred - 1.9600*dsigma
dy_pred_95_plus = dy_pred + 1.9600*dsigma



dfs8_fs8 = dy_pred / y_pred

plt.plot(xi, dfs8_fs8, color='green', label='GP Prediction')


dfs8_mc = []
for i in range(10000):
                         
    fs8i = np.random.normal(y_pred, scale=sigma)
    
    dfs8i = np.random.normal(dy_pred, scale=dsigma)
    
    dfs8_mc.append(dfs8i/fs8i)

dfs8_mc = np.array(dfs8_mc)   

sigma_dfs8 = []
for i in range(len(xi)):
    
    sigma_dfs8.append(np.std(dfs8_mc[:, i]))

sigma_dfs8 = np.array(sigma_dfs8)    

plt.fill_between(xi, dfs8_fs8 - sigma_dfs8, dfs8_fs8 + sigma_dfs8, alpha=.5, fc='forestgreen', ec='None')
plt.fill_between(xi, dfs8_fs8 - 1.96*sigma_dfs8, dfs8_fs8 + 1.96*sigma_dfs8, alpha=.5, fc='lightgreen', ec='None')


G = xi, dfs8_fs8, sigma_dfs8

#np.savetxt('dfs8_recon_novo.dat', np.transpose(G), delimiter='\t')




#LCDM

import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, 
    h=0.6727, sigma8=0.8120, n_s=0.9649)

z = np.linspace(0, 1, 1000)
a = 1. / (1. + z)

fs8_lcdm = ccl.growth_rate(cosmo, a)*0.812*ccl.growth_factor(cosmo, a)

dfs8_lcdm = np.gradient(fs8_lcdm, z)


plt.plot(z, dfs8_lcdm/fs8_lcdm, color='red', label='$\Lambda$CDM')

# legenda, label e título
#plt.ylim(-0.3, 0.7)
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\mathcal{S}(z)$', fontsize=16)
plt.legend(loc='best')
#plt.savefig('dfs8_fs8_mc.pdf', format='pdf', bbox_inches='tight')
plt.show()


# TIMER

tf = time.time()

tempo = tf - ti

print(tempo)


