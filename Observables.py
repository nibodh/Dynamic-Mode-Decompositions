#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from itertools import combinations_with_replacement as choose 

def Monomials(data, polyorder):
    n, m, l = data.shape
    psi_data = data
    if polyorder > 1:
        psi_data = np.append(psi_data,np.ones((1,m,l)),axis = 0)
        for i in np.arange(2,polyorder + 1):
            for j in choose(np.arange(n),i):
                prod = 1
                for k in j:
                    prod = prod*data[k,:,:]
                psi_data = np.append(psi_data,prod[np.newaxis,:,:],axis = 0)
    return psi_data