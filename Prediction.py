#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def one_step(K,data):  #basically applying K
    
    if data.ndim == 2:
        n, m = data.shape
        l = 1
    else:
        n, m, l = data.shape
    
    data_p = data[:,:-1,:] #present
    data_f = data[:,1:,:] #forwarded
    
    data_est = np.transpose(np.matmul(np.transpose(np.repeat(K[:,:,np.newaxis],l,axis = 2),(2, 0, 1)), np.transpose(data_p,(2, 0, 1))),(1, 2, 0))
    #substitute for the above stupid python expression is broken down below
    
#     K_tile  = np.repeat(K[:,:,np.newaxis],l,axis = 2)
#     K_transpose = np.transpose(K_tile,(2, 0, 1))
#     data_p_transpose = np.transpose(data_p,(2, 0, 1))
#     K_times_X = np.matmul(K_transpose, data_p_transpose)
#     data_est = np.transpose(K_times_X,(1, 2, 0))
    
    Y_est = np.reshape(data_est,(n,(m-1)*l))
    Y = np.reshape(data_f,(n,(m-1)*l))
    MSE_error = np.linalg.norm(Y_est - Y, 'fro')/np.linalg.norm(Y,'fro')
    
    return data_est, MSE_error

def N_step(K,data):  #basically applying powers of K
    if data.ndim == 2:
        n, m = data.shape
        l = 1
    else:
        n, m, l = data.shape
    
    data_p = data[:,:-1,:] #present
    data_f = data[:,1:,:] #forwarded
    
    data_est = np.empty([n,m-1,l]) #initializing empty array b/c stupid python can't do it by itself
    data_init = data_p[:,0,:] #intial conditions
    for i in np.arange(m-1):  #each time stamp to take a power of the dynamics matrix to predict trajectory at that time
        data_est[:,i,:] = np.squeeze(np.transpose(np.matmul(np.transpose(np.repeat(np.linalg.matrix_power(K,(i+1))[:,:,np.newaxis],l,axis = 2),(2, 0, 1)), np.transpose(data_init[:,:,np.newaxis],(1,0,2))),(1,2,0)),axis = 1)
        #substitute for the above stupid python expression is broken down below
        
#     for i in np.arange(0,m-1):
#         K_power = np.linalg.matrix_power(K,(i+1))
#         K_tile  = np.repeat(K_power[:,:,np.newaxis],l,axis = 2)
#         K_transpose = np.transpose(K_tile,(2, 0, 1))
#         data_init = data_p[:,0,:]
#         data_p_transpose = np.transpose(data_init[:,:,np.newaxis],(1,0,2))
#         K_times_X = np.matmul(K_transpose, data_p_transpose)
#         siz = K_times_X.shape
#         data_est[:,i,:] = np.squeeze(np.transpose(K_times_X,(1,2,0)),axis = 1)
    
    Y_est = np.reshape(data_est,(n,(m-1)*l))
    Y = np.reshape(data_f,(n,(m-1)*l))
    MSE_error = np.linalg.norm(Y_est - Y, 'fro')/np.linalg.norm(Y,'fro')
    
    return data_est, MSE_error

