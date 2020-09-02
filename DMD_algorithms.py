#!/usr/bin/env python
# coding: utf-8

# In[ ]:

##### Matrix inversions using pseudo inverse, singular value based to be added

import numpy as np

def DMD(data,basis_type,basis_order): #polytype and polyorder unused
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape

    X = np.transpose(np.reshape(data[:,:-1,:],(n,(m-1)*l)),(1,0)) # reshaping data_p to bring all trajectories into one dimension
    Y = np.transpose(np.reshape(data[:,1:,:],(n,(m-1)*l)),(1,0)) # reshaping data_f to bring all trajectories into one dimension
    
    G = np.transpose(X) @ X
    A = np.transpose(X) @ Y
    
    K = np.linalg.pinv(G) @ A  #the 2-norm minimized over the trajectory is the frobenius norm
    
    return G, A, K

def eDMD(data,basis_type,basis_order): # Data comes in with state-space notation
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape
    
    import Basis #importing Basis module for eDMD
    transformation = getattr(Basis, basis_type) #desire function for basis
    psi_data = transformation(data,basis_order) #lifted space
    
    nl = psi_data.shape[0] #dimension of lifted space
    
    psi_X = np.transpose(np.reshape(psi_data[:,:-1,:],(nl,(m-1)*l)),(1,0)) #transposing data for notation
    psi_Y = np.transpose(np.reshape(psi_data[:,1:,:],(nl,(m-1)*l)),(1,0))
    
    G = np.transpose(psi_X) @ psi_X
    A = np.transpose(psi_X) @ psi_Y
    B = np.transpose(psi_X) @ np.transpose(np.reshape(data[:,:-1,:],(n,(m-1)*l)),(1,0))
    
    K = np.linalg.pinv(G) @ A #the 2-norm minimized over the trajectory is the frobenius norm
    C = np.linalg.pinv(G) @ B
    
    return G, A, K, C

def HankelDMD(data,basis_type,basis_order): #basis_type and polyorder unused
    
    if data.ndim == 2:
        data = data[:,:,np.newaxis]
    n, m, l = data.shape   #num of time points  #num of states #num of trajectories
    
    #Construct Hankel delay matrices per basis
    
    #Below is a temporary estimte of k ############# This will be used as the intial guess to look at the number of 0 singular values so that the k may be increased if none and decreased if many 
    import Basis #importing Basis module for eDMD
    transformation = getattr(Basis, basis_type) #desire function for basis
    psi_data = transformation(data,basis_order) #lifted space
    
    k = psi_data.shape[0] #guessed value for dimension of K-invariant subspace, should create an optimization routine
    p = (m-1)//k #based on size of column, max number of columns is obtained
    discard = (m-1) - k*p #number of last time points ignored ignored to build a Hankel data matrix
    
    #Contructing Hankel matrices:
    
    #separating data into present and future
    #reshaping each onbservables of trajectory into Hankel matrix per state per trajectory and transposing
    H_X = np.transpose(np.reshape(data[:,:m-1-discard,:],(n,k,p,l)),(1,2,0,3)) #storing X data for reshaping
    H_Y = np.transpose(np.reshape(data[:,1:m-discard,:],(n,k,p,l)),(1,2,0,3))  #storing Y data for reshaping
    #scaling according to norm of the last column of each of the Hankel matrix
    alpha = np.linalg.norm(H_X[:,-1,:,:],axis = 0)/np.linalg.norm(H_X[:,-1,0,0]) #calculating norm of last column of each trajectory's H_X and scaling them all with that of the first H_X
    
    #multiplying with scaling factors
    for i in np.arange(0,n):
        for j in np.arange(0,l):
            H_X[:,:,i,j] = H_X[:,:,i,j]*alpha[i,j]
            H_Y[:,:,i,j] = H_Y[:,:,i,j]*alpha[i,j]
    
    X = np.reshape(H_X,(k,p*n*l)) #unfolding Hankel matrices into dimension 1
    Y = np.reshape(H_Y,(k,p*n*l)) #unfolding Hankel matrices into dimension 1
    K = Y @ np.linalg.pinv(X)
    
    return K

