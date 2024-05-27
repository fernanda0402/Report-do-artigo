#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:27:20 2024

@author: usuario
"""

import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
plt.rcParams['text.usetex'] = True



data1 = np.genfromtxt('/home/usuario/Documentos/Códigos/Report do artigo/gamma_mc_novo.dat', delimiter='\t')

z = data1[:, 0]
gamma_RBF = data1[:, 1]
sig_g1 = data1[:, 2]


data2 = np.genfromtxt('/home/usuario/Documentos/Códigos/Report do artigo/gamma_mc_novo_Matern.dat', delimiter='\t')

gamma_M = data2[:, 1]
sig_g2 = data2[:, 2]


dg = gamma_RBF / gamma_M - 1

sigma_dg = np.sqrt( (sig_g1 / gamma_M)**2 + (gamma_RBF*sig_g2 / (gamma_M)**2)**2 )


# plote
fig, ax = plt.subplots()
plt.ylim(-6,6)
plt.xlim(0,1.0)
plt.tick_params(labelsize=14, color='red')
plt.plot(z, dg, color='darkgreen', linestyle="--")
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([dg - 1.000 * sigma_dg,
                        (dg + 1.000 * sigma_dg)[::-1]]),
         alpha=.5, fc='forestgreen', ec='None')
plt.fill(np.concatenate([z, z[::-1]]),
         np.concatenate([dg - 1.9600 * sigma_dg,
                        (dg + 1.9600 * sigma_dg)[::-1]]),
         alpha=.5, fc='lightgreen', ec='None')
plt.axhline(y=0.0, color='red', linestyle='-', linewidth=1)

# legenda, label e título
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$\gamma^{RBF}/\gamma^{M}$ - 1', fontsize=16)
#plt.savefig('gamma_CC_kernels.pdf', format='pdf', bbox_inches='tight')
plt.show()