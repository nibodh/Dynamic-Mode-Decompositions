#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from itertools import combinations_with_replacement as choose 

def Monomials(data, polyorder):
    n, m, l = data.shape
    psi_data = data
    if polyorder > 1:
        psi_data = np.append(psi_data,np.ones((1,m,l)),axis = 0) #Adding 0 order polynomials and 1s
        for i in np.arange(2,polyorder + 1):  #taking combinations possible
            for j in choose(np.arange(n),i):  #iterating through eeach combination
                prod = 1
                for k in j: #iterating through states for combination
                    prod = prod*data[k,:,:] #multiplying each state for that combination
                psi_data = np.append(psi_data,prod[np.newaxis,:,:],axis = 0)
    return psi_data