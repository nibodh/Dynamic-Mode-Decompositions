#!/usr/bin/env python
# coding: utf-8

# In[ ]:

##### Matrix inversions using pseudo inverse, singular value based to be added

import numpy as np

def DMD(data,polytype,polyorder): #polytype and polyorder unused
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

def HankelDMD(data,polytype,polyorder): #polytype and polyorder unused
    n, m, l = data.shape   #num of time points  #num of states #num of trajectories
    
    #Construct Hankel delay matrices per observable
    k = 20 #guessed value for dimension of K-invariant subspace
    p = (m-1)//k #based on size of column, number of columns is obtained
    
    #Contructing Hankel matrices:
    
    #separating data into present and future
    #reshaping each onbservables of trajectory into Hankel matrix per state per trajectory and transposing
    H_X = np.transpose(np.reshape(data[:,:-1,:],(n,k,p,l)),(1,2,0,3)) #storing X data for reshaping
    H_Y = np.transpose(np.reshape(data[:,1:,:],(n,k,p,l)),(1,2,0,3))  #storing Y data for reshaping
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

