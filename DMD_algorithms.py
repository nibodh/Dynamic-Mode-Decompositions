#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def DMD(data,polytype,polyorder):
    n, m, l = data.shape

    X = np.reshape(data[:,:-1,:],(n,(m-1)*l)) # reshaping data_p to bring all trajectories into one dimension
    Y = np.reshape(data[:,1:,:],(n,(m-1)*l)) # reshaping data_f to bring all trajectories into one dimension
    K = Y @ np.linalg.pinv(X)  #the 2-norm minimized over the trajectory is the frobenius norm
    
    return K

def eDMD(data,polytype,polyorder):
    n, m, l = data.shape
    
    import Observables #importing observables module for eDMD
    transformation = getattr(Observables, polytype) #desire function for observable
    psi_data = transformation(data,polyorder) #lifted space
    
    nl = psi_data.shape[0] #dimension of lifted space
    
    psi_X = np.reshape(psi_data[:,:-1,:],(nl,(m-1)*l))
    psi_Y = np.reshape(psi_data[:,1:,:],(nl,(m-1)*l))
    
#     def mat_summ(X,Y): #Refer Williams: Extedning DMD, 2015
#         n, m = X.shape
#         Z = np.zeros([n,n])
#         for i in np.arange(m):
#             Z = Z + Y[:,i]@np.transpose(X[:,i])
#         Z = Z
#         return Z
#     
#     G = mat_summ(psi_X,psi_X)
#     A = mat_summ(psi_X,psi_Y)
    
    K = psi_Y @ np.linalg.pinv(psi_X) #the 2-norm minimized over the trajectory is the frobenius norm
    
    return K

# def HankelDMD(data):
#     n, m, l = data.shape
    
#     m = X.shape[1] #num of time points
#     n = X.shape[0] #num of states
#     K = Y * np.linalg.pinv(X)
    
#     return K

