#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np

def Glycoscillator(species,t):
    # constants    
    J_0 = 2.5
    k_1 = 100
    k_2 = 6
    k_3 = 16
    k_4 = 100
    k_5 = 1.28
    k_6 = 12
    k = 1.8
    kappa = 13
    q = 4
    K_1 = 0.52
    psi = 0.1
    N = 1
    A = 4
    # 7 States
    S_1 = species[0]
    S_2 = species[1]
    S_3 = species[2]
    S_4 = species[3]
    S_5 = species[4]
    S_6 = species[5]
    S_7 = species[6]
    # ODEs for each state
    S_1_dot = J_0 - (k_1 * S_1 * S_6)/(1 + (S_6/K_1)**q)
    S_2_dot = 2 * (k_1 * S_1 * S_6)/(1 + (S_6/K_1)**q) - k_2 * S_2 * (N - S_5) - k_6 * S_2 * S_5
    S_3_dot = k_2 * S_2 * (N - S_5) - k_3 * S_3 * (A - S_6)
    S_4_dot = k_3 * S_3 * (A - S_6) - k_4 * S_4 * S_5 - kappa * (S_4 - S_7)
    S_5_dot = k_2 * S_2 * (N - S_5) - k_4 * S_4 * S_5 - k_6 * S_2 * S_5
    S_6_dot = -2 * (k_1 * S_1 * S_6)/(1 + (S_6/K_1)**q) + 2 * k_3 * S_3 * (A - S_6) - k_5 * S_6
    S_7_dot = psi * kappa * (S_4 - S_7) - kappa * S_7
    
    speciesdot = [S_1_dot, S_2_dot, S_3_dot, S_4_dot, S_5_dot, S_6_dot, S_7_dot]
    
    return speciesdot

def Repressilator(species,t):
    #constants
    kdm = 1
    kdp = 1
    alpha = 100
    alpha_0 = 0
    beta = 1
    n = 2
    #Usually 6 but can be of higher number of states
    len2 = len(species)//2
    speciesdot = []
    #mRNA
    for i in np.arange(len2):
        if i == 0:
            speciesdot.append(-kdm*species[i] + alpha/(1 + species[2*len2-1]**n) + alpha_0)
        else:
            speciesdot.append(-kdm*species[i] + alpha/(1 + species[i+len2-1]**n) + alpha_0)
    #Proteins
    for j in np.arange(len2,2*len2):
        speciesdot.append(beta*(species[j-len2] - kdp*species[j]))
    
    return speciesdot

