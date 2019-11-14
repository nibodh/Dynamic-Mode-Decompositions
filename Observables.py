#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from itertools import combinations_with_replacement #equivalent to (n+k-1)_c_(k-1)

def Monomials(data, polyorder):
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
        
    psi_data = data
    if polyorder > 1:
        psi_data = np.append(psi_data,np.ones((1,m,l)),axis = 0) #Adding 0 order polynomials and 1s
        for i in np.arange(2,polyorder + 1):  #taking all possible powers from 2 to polyorder
            for j in combinations_with_replacement(np.arange(n),i):  #at each power, there are (p+n-1)_c_(n-1) solutions to the number of distinct monomials of order i. j iterates through each of them
                prod = 1
                for k in j: #k iterates through the tuple that j is to increase the order of each state in the monomial
                    prod = prod*data[k,:,:] #multiplying each state for that combination
                psi_data = np.append(psi_data,prod[np.newaxis,:,:],axis = 0)
    return psi_data



def Hermite(data, polyorder):
    from numpy.polynomial.hermite_e import hermeval    #importing hermeval from hermite class
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
        
    psi_data = data
    if polyorder > 1:
        psi_data = np.append(psi_data,np.ones((1,m,l)),axis = 0)
        
#         def herm(x,n): #recursion routine to calculate hermite poly
#             if n == 0:
#                 return np.ones(x.shape)
#             elif n ==1:
#                 return x
#             else:
#                 return x*herm(x,n-1) - (n-1)*herm(x,n-2)
                    
        for i in np.arange(2,polyorder + 1):
            c = np.zeros(i+1)  # vector of coefficients of size i+1
            c[i] = 1  # 1 at index n+1 for extracting H^i
            psi_data = np.append(psi_data,hermeval(data,c),axis = 0)  #appending H^i to psi
#             psi_data = np.append(psi_data,herm(data,i),axis = 0)
    return psi_data



def Legendre(data, polyorder):
    from numpy.polynomial.legendre import legval    #importing legval from legendre class
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
        
    psi_data = data
    if polyorder > 1:
        psi_data = np.append(psi_data,np.ones((1,m,l)),axis = 0)
        
#         def legend(x,n): #recursion routine to calculate legendre poly
#             if n == 0:
#                 return np.ones(x.shape)
#             elif n ==1:
#                 return x
#             else:
#                 return (1/n)*((2*n-1)*x*legend(x,n-1) - (n-1)*legend(x,n-2))
                    
        for i in np.arange(2,polyorder + 1):
            c = np.zeros(i+1)    # vector of coefficients of size i+1
            c[i] = 1     # 1 at index n+1 for extracting L^i
            psi_data = np.append(psi_data,legval(data,c),axis = 0)    #appending L^i to psi
#             psi_data = np.append(psi_data,legend(data,i),axis = 0)
    return psi_data