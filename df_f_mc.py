#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:35:26 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from gapp import gp, dgp
import time

ti = time.time()

plt.rcParams['text.usetex'] = True

# baixando os dados de f
fz = np.genfromtxt('/home/usuario/Documentos/Dados/fz_data.csv', delimiter=', ')

z = fz[:, 0]
f_z = fz[:, 1]

ef = fz[:, 2]

##################### PROCESSO GAUSSIANO GAPP ###########################

# nomeando
x_gapp = z
y_gapp = f_z
e = ef

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0.0
xmax = 1.0
nstar = 200

# initial values of the hyperparameters of the squared-exponential covariance function
initheta = [2.0, 2.0]

# initialization of the Gaussian Process
g = gp.GaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar))

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
y_gapp = f_z
e = ef

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




df_f = dy_pred / y_pred

plt.plot(xi, df_f, color='green', label='GP Prediction')




df_mc = []
for i in range(10000):
                         
    fi = np.random.normal(y_pred, scale=sigma)
    
    dfi = np.random.normal(dy_pred, scale=dsigma)
    
    df_mc.append(dfi/fi)

df_mc = np.array(df_mc)   

sigma_df = []
for i in range(len(xi)):
    
    sigma_df.append(np.std(df_mc[:, i]))

sigma_df = np.array(sigma_df)    

plt.fill_between(xi, df_f - sigma_df, df_f + sigma_df, alpha=.5, fc='forestgreen', ec='None')
plt.fill_between(xi, df_f - 1.96*sigma_df, df_f + 1.96*sigma_df, alpha=.5, fc='lightgreen', ec='None')



G = xi, df_f, sigma_df

#np.savetxt('df_recon.dat', np.transpose(G), delimiter='\t')


#LCDM

import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, 
    h=0.6727, sigma8=0.8120, n_s=0.9649)

z = np.linspace(0, 1, 1000)
a = 1. / (1. + z)

flcdm = ccl.background.growth_rate(cosmo, a)

df_lcdm = np.gradient(flcdm, z)


plt.plot(z, df_lcdm/flcdm, color='red', label='$\Lambda$CDM')

# legenda, label e título
plt.ylim(-0.5, 1.5)
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\mathcal{F}(z)$', fontsize=16)
plt.legend(loc='best')
#plt.savefig('df_f_mc.pdf', format='pdf', bbox_inches='tight')
plt.show()


# TIMER

tf = time.time()

tempo = tf - ti

print(tempo)


