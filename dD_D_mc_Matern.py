#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:27:02 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True

Hz_data = np.genfromtxt('/home/usuario/Documentos/Códigos/Report do artigo/dh_recon_Matern.dat', delimiter='\t')

z = Hz_data[:, 0]
H = Hz_data[:, 1]

eH = Hz_data[:, 2]

Ez_data = np.genfromtxt('/home/usuario/Documentos/Códigos/Report do artigo/E_recon_Matern.dat', delimiter='\t')

E = Ez_data[:, 1]

eE = Ez_data[:, 2]

######################################################

E3 = E ** 3

eE3 = np.sqrt((3*E*E*eE)**2)

from scipy.integrate import cumtrapz

Z = (1+z) / E3

I = cumtrapz(Z, x=z, initial=0)

D = H - Z*(1/(1-I))

plt.xlim(0, 1)
plt.ylim(-0.8, 0)
plt.plot(z, D, color='green', label='GP Prediction')


# MODELO LCDM

import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, 
    h=0.6727, sigma8=0.8120, n_s=0.9649)

zi = np.linspace(0, 1, 1000)
ai = 1. / (1. + zi)

Dz = ccl.growth_factor(cosmo, ai)

dDz = np.gradient(Dz, zi)

D_LCDM = dDz / Dz

plt.plot(zi, D_LCDM, color='red', label='$\Lambda$CDM')

############################ propagação do erro

# erro do I

I_mc = []
for i in range(10000):
    
    E3i = np.random.normal(E3, eE3)
    
    Zi = (1+z) / E3i
    
    Ii = cumtrapz(Zi, x=z, initial=0)
    
    I_mc.append(Ii)
        
I_mc = np.array(I_mc)

sigma_I = []
for i in range(len(z)):
    
    Ii = I_mc[:, i]
    sigma_I.append(np.std(Ii))

sigma_I = np.array(sigma_I)  


D_mc = []
for i in range(10000):
    
    E3i = np.random.normal(E3, eE3)
    
    Zi = (1+z) / E3i
    
    Ii = np.random.normal(I, sigma_I)
    
    dHi = np.random.normal(H, eH)
    
    Di = dHi - (Zi/(1-Ii))
     
    D_mc.append(Di)
        
D_mc = np.array(D_mc)

sigma_D = []
for i in range(len(z)):
    
    Di = D_mc[:, i]
    sigma_D.append(np.std(Di))

sigma_D = np.array(sigma_D) 

plt.fill_between(z, D-sigma_D, D+sigma_D, alpha=0.5, fc='forestgreen', ec='None') 
plt.fill_between(z, D-1.96*sigma_D, D+1.96*sigma_D, alpha=0.5,  fc='lightgreen', ec='None') 


# legenda, label e título
plt.ylim(-1, 0.1)
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\mathcal{D}(z)$', fontsize=16)
plt.legend(loc='lower right')
#plt.savefig('dD_D_mc.pdf', format='pdf', bbox_inches='tight')
plt.show()



G = z, D, sigma_D

#np.savetxt('dD_recon_Matern.dat', np.transpose(G), delimiter='\t')
