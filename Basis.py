#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from itertools import combinations_with_replacement #equivalent to (n+k-1)_c_(k-1)

def Monomials(data, basis_order):
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
        
    psi_data = data
    if basis_order > 1:
        psi_data = np.append(psi_data,np.ones((1,m,l)),axis = 0) #Adding 0 order polynomials and 1s
        for i in np.arange(2, basis_order + 1):  #taking all possible powers from 2 to polyorder
            for j in combinations_with_replacement(np.arange(n),i):  #at each power, there are (p+n-1)_c_(n-1) solutions to the number of distinct monomials of order i. j iterates through each of them
                prod = 1
                for k in j: #k iterates through the tuple that j is to increase the order of each state in the monomial
                    prod = prod*data[k,:,:] #multiplying each state for that combination
                psi_data = np.append(psi_data,prod[np.newaxis,:,:],axis = 0)
    return psi_data



def Hermite(data, basis_order):
    from numpy.polynomial.hermite_e import hermeval    #importing hermeval from hermite class
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
        
    psi_data = data
    if basis_order > 1:
        psi_data = np.append(psi_data,np.ones((1,m,l)),axis = 0)                
        for i in np.arange(2, basis_order + 1):
            c = np.zeros(i+1)  # vector of coefficients of size i+1
            c[i] = 1  # 1 at index n+1 for extracting H^i
            psi_data = np.append(psi_data,hermeval(data,c),axis = 0)  #appending H^i to psi
#             psi_data = np.append(psi_data,herm(data,i),axis = 0)
    return psi_data



def Legendre(data, basis_order):
    from numpy.polynomial.legendre import legval    #importing legval from legendre class
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
        
    psi_data = data
    if basis_order > 1:
        psi_data = np.append(psi_data,np.ones((1,m,l)),axis = 0)
        for i in np.arange(2, basis_order + 1):
            c = np.zeros(i+1)    # vector of coefficients of size i+1
            c[i] = 1     # 1 at index n+1 for extracting L^i
            psi_data = np.append(psi_data,legval(data,c),axis = 0)    #appending L^i to psi
#             psi_data = np.append(psi_data,legend(data,i),axis = 0)
    return psi_data

def ThinPlate(data, basis_order):
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
    
    x_max = 2.2
    x_min = -2.2
    
    basis_order = np.int(np.ceil(basis_order**(1./n))) #taking the nth root and assigning that as basis order across each dimension
    
    psi_data = np.ones((1,m,l)) #initializing constant function
    
    for i in np.arange(n): #In each dimension
        centers = np.linspace(x_min,x_max,basis_order) #RBF centers along each dimension
        for j in np.arange(basis_order):
            r = np.absolute(data[i,:,:] - centers[j]) #computing value of RBFs per dimension with 
            psi_x = np.log(r) * np.square(r)
            psi_x[np.isnan(psi_x)] = 0 #checks all nan to 0
            psi_data = np.append(psi_data, psi_x[np.newaxis,:,:], axis = 0)
            
    return psi_data
    
def RBF(data, basis_order):
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
    
    epsilon = 5
    
    x_max = 2.2
    x_min = -2.2
    
    basis_order = np.int(np.ceil(basis_order**(1./n))) #taking the nth root and assigning that as basis order across each dimension
    
    psi_data = np.ones((1,m,l)) #initializing constant function
    
    for i in np.arange(n): #In each dimension
        centers = np.linspace(x_min,x_max,basis_order) #RBF centers along each dimension
        for j in np.arange(basis_order):
            r = np.absolute(data[i,:,:] - centers[j]) #computing value of RBFs per dimension with 
            psi_x = np.exp(-1 * (epsilon**2) *np.square(r))
            psi_data = np.append(psi_data, psi_x[np.newaxis,:,:], axis = 0)
            
    return psi_data
    
def Polyharmonic(data, basis_order):
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
    
    k = 2
    
    x_max = 2.2
    x_min = -2.2
    
    basis_order = np.int(np.ceil(basis_order**(1./n))) #taking the nth root and assigning that as basis order across each dimension
    
    psi_data = np.ones((1,m,l)) #initializing constant function
    
    for i in np.arange(n): #In each dimension
        centers = np.linspace(x_min,x_max,basis_order) #RBF centers along each dimension
        for j in np.arange(basis_order):
            r = np.absolute(data[i,:,:] - centers[j]) #computing value of RBFs per dimension with 
            psi_x = np.log(np.power(r,r)) * np.power(r,k-1)
            psi_x[np.isnan(psi_x)] = 0 #checks all nan to 0
            psi_data = np.append(psi_data, psi_x[np.newaxis,:,:], axis = 0)
            
    return psi_data

def Wendland(data, basis_order):
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
    
    x_max = 2.2
    x_min = -2.2
    
    basis_order = np.int(np.ceil(basis_order**(1./n))) #taking the nth root and assigning that as basis order across each dimension
    
    psi_data = np.ones((1,m,l)) #initializing constant function
    
    for i in np.arange(n): #In each dimension
        centers = np.linspace(x_min,x_max,basis_order)
        for j in np.arange(basis_order):
            r = np.absolute(data[i,:,:] - centers[j])
            r[r > 1] = 0
            psi_x = np.power(1-r,4) * (1 + 4*r)
            psi_data = np.append(psi_data, psi_x[np.newaxis,:,:], axis = 0)
            
    return psi_data