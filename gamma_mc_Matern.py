#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:21:35 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
from gapp import gp, dgp
import time
import pyccl as ccl

ti = time.time()

plt.rcParams['text.usetex'] = True


############################################## H(z) ###################################################

# baixando os dados
data_Hz = np.genfromtxt('/home/usuario/Documentos/Códigos/Report do artigo/dh_recon_Matern.dat', delimiter='\t')

z = data_Hz[:, 0]
dH_H = data_Hz[:, 1]

sigma_dH = data_Hz[:, 2]






# definindo Om'/Om
dOm_Om = (3/(1+z)) - (2*dH_H)
eOm = np.sqrt( (2*sigma_dH)**2 )

plt.plot(z, dOm_Om, color='green', label='GP Prediction')
plt.fill_between(z, dOm_Om - eOm , dOm_Om + eOm , alpha=.5, fc='forestgreen', ec='None')
plt.fill_between(z, dOm_Om - 1.96*eOm , dOm_Om + 1.96*eOm , alpha=.5, fc='lightgreen', ec='None')






# MODELO LCDM

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649,
    matter_power_spectrum='linear')


zlcdm = np.linspace(0.001, 1.0, 1000)

a = 1. / (1. + zlcdm)

Om_lcdm = ccl.background.omega_x(cosmo, a, 'matter')

dO_lcdm = np.gradient(Om_lcdm, zlcdm) / Om_lcdm

plt.plot(zlcdm, dO_lcdm, color='red', label='$\Lambda$CDM')


# legenda, label e título
plt.ylim(0, 2.7)
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\mathcal{O}(z)$', fontsize=16)
plt.legend(loc='best')
#plt.savefig('dO_O_mc.pdf', format='pdf', bbox_inches='tight')
plt.show()





#################################################### f(z) #######################################

# baixando os dados de f
fz = np.genfromtxt('/home/usuario/Documentos/Códigos/Report do artigo/df_recon_Matern.dat', delimiter='\t')

z = fz[:, 0]
df_f = fz[:, 1]

ef = fz[:, 2]




############################ DEFININDO GAMMA #####################


gamma_rec = df_f / dOm_Om



gamma_mc = []
for i in range(10000):
                         
    fi = np.random.normal(df_f, ef)
    Oi = np.random.normal(dOm_Om, eOm)
    
    gamma_mc.append(fi / Oi)

gamma_mc = np.array(gamma_mc)   


sigma_g = []
for i in range(len(z)):
    
    gi = gamma_mc[:, i]
    sigma_g.append(np.std(gi[(gi>0.)&(gi<1)]))

sigma_g = np.array(sigma_g)    


G = z, gamma_rec, sigma_g

np.savetxt('gamma_mc_novo_Matern.dat', np.transpose(G), delimiter='\t')



# plote
fig, ax = plt.subplots()
plt.plot(z, gamma_rec, color='green', label='GP Prediction')
plt.axhline(y=0.55, color='red', linestyle='-', linewidth=1, label='$\Lambda$CDM')

plt.fill_between(z, gamma_rec - sigma_g, gamma_rec + sigma_g, alpha=.5, fc='forestgreen', ec='None')
plt.fill_between(z, gamma_rec - 1.96*sigma_g, gamma_rec + 1.96*sigma_g, alpha=.5, fc='lightgreen', ec='None')



# legenda, label e título
plt.ylim(-2,2)
plt.xlim(0,1.0)
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\gamma$', fontsize=16)
plt.legend(loc='upper left')
#plt.savefig('gamma_mc_Matern.pdf', format='pdf', bbox_inches='tight')
plt.show()


print(gamma_rec[0])
print(sigma_g[0])

