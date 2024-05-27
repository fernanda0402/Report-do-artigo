#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:22:11 2024

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



############################################### D(z) #######################################3

Dz = np.genfromtxt('/home/usuario/Documentos/Códigos/Report do artigo/dD_recon_Matern.dat', delimiter='\t')

z = Dz[:, 0]
dD_D = Dz[:, 1]

eD = Dz[:, 2]


#################################################### fs8(z) #######################################

# baixando os dados de fs8
fz = np.genfromtxt('/home/usuario/Documentos/Códigos/Report do artigo/dfs8_recon_novo_Matern.dat', delimiter='\t')

z = fz[:, 0]
dfs8_fs8 = fz[:, 1]

efs8 = fz[:, 2]


############################ DEFININDO GAMMA #####################


gamma_rec = (dfs8_fs8 - dD_D) / dOm_Om



gamma_mc = []
for i in range(10000):
                         
    fi = np.random.normal(dfs8_fs8, efs8)
    Oi = np.random.normal(dOm_Om, eOm)
    Di = np.random.normal(dD_D, eD)
    
    gamma_mc.append( (fi - Di) / Oi)

gamma_mc = np.array(gamma_mc)   


sigma_g = []
for i in range(len(z)):
    
    gi = gamma_mc[:, i]
    sigma_g.append(np.std(gi[(gi>0.)&(gi<1)]))

sigma_g = np.array(sigma_g)    



G = z, gamma_rec, sigma_g

#np.savetxt('gamma_fs8_mc_novo_Matern.dat', np.transpose(G), delimiter='\t')


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
plt.legend(loc='upper right')
#plt.savefig('gamma_fs8_mc_new_Matern.pdf', format='pdf', bbox_inches='tight')
plt.show()


print(gamma_rec[0])
print(sigma_g[0])