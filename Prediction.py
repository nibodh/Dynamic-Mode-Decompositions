#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def one_step(data, K_t, C_t, basis_type, basis_order):  #K_t comes in as TRANSPOSE of K
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
    nl = n
    
    import Basis #importing basis module for eDMD
    transformation = getattr(Basis, basis_type) #desire function for basis
    Psi = transformation(data,basis_order) #lifted space
    nl = Psi.shape[0] #dimension of lifted space
    
    Psi_p = Psi[:,:-1,:] #present
    data_f = data[:,1:,:] #forwarded
    
    Psi_est = np.transpose(np.matmul(np.transpose(np.repeat(K_t[:,:,np.newaxis],l,axis = 2),(2, 0, 1)), np.transpose(Psi_p,(2, 0, 1))),(1, 2, 0))
    Y_est = C_t @ np.reshape(Psi_est,(nl,(m-1)*l))
    Y = np.reshape(data_f,(n,(m-1)*l))
    MSE_error = np.linalg.norm(Y_est - Y, 'fro')/np.linalg.norm(Y,'fro')
    
    return Y_est, MSE_error

def N_step(data, K_t, C_t, basis_type, basis_order):  #K_t comes in as TRANSPOSE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
    nl = n
    
    import Basis #importing basis module for eDMD
    transformation = getattr(Basis, basis_type) #desire function for basis
    Psi = transformation(data,basis_order) #lifted space
    nl = Psi.shape[0] #dimension of lifted space
    
    Psi_p = Psi[:,:-1,:] #present
    data_f = data[:,1:,:] #forwarded
    
    Psi_est = np.empty([nl,m-1,l]) #initializing empty array b/c stupid python can't do it by itself
    Psi_init = Psi_p[:,0,:] #intial conditions
    for i in np.arange(m-1):  #each time stamp to take a power of the dynamics matrix to predict trajectory at that time
        Psi_est[:,i,:] = np.squeeze(np.transpose(np.matmul(np.transpose(np.repeat(np.linalg.matrix_power(K_t,(i+1))[:,:,np.newaxis],l,axis = 2),(2, 0, 1)), np.transpose(Psi_init[:,:,np.newaxis],(1,0,2))),(1,2,0)),axis = 1)
    Y_est = C_t @  np.reshape(Psi_est,(nl,(m-1)*l))
    Y = np.reshape(data_f,(n,(m-1)*l))
    MSE_error = np.linalg.norm(Y_est - Y, 'fro')/np.linalg.norm(Y,'fro')
    
    return np.reshape(Y_est,(n,m-1,l)), MSE_error

