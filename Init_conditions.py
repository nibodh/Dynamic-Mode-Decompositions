#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def Glycoscillator():
    IC_inside = [ 0.5325, 0.4767, 0.0618, 0.3173, 0.2075, 1.5311, 0.0572] #This IC grows onto the limit cycle, away from an unstable fp
    IC_limit = [1.21440084, 0.27349879, 0.04031784, 0.05089061, 0.14671042, 1.88564678, 0.00505542] #IC on the attractor
    IC_outside = [ 0.4820, 1.9893, 0.0644, 0.3065, 0.1984, 2.6602, 0.0539]  #IC decays and settles on the attractor
    IC_1 = [ 0.875, 1.175, 0.12, 0.225, 0.19, 1.405, 0.075] #Artbitrarily selected IC
    
    IC_test = [3.5, 2, 1, 1.5, 2.3, 1.8, 1.4] #More such test trajectories can be entered below and tested for using num_test in main code

    IC = np.transpose([IC_inside, IC_limit, IC_outside, IC_1, IC_test])
    return IC

def Repressilator():
    
#     limits = 120
    
#     nx = np.int(np.ceil(np.sqrt(num_train)))
#     ny = nx
#     x = np.linspace(0,limits,nx)
#     for i in np.arange(nx):
#         for j in np.arange(nx):
#             for k in np.arange(ny):
#                 for l in np.arange(nx):
#             if i == 0 and j == 0:
#                 IC_train = np.array([x[0], y[0]])
#                 IC_train = IC_train[:,np.newaxis]
#             else:
#                 pt = np.array([x[i], y[j]])
#                 IC_train = np.append(IC_train, pt[:,np.newaxis], axis = 1)

    IC_inside = [3.79156715, 5.65839438, 4.45129228, 3.87443063, 5.07860964, 4.89017854] #This IC grows onto the limit cycle, away from an unstable fp
    IC_limit = [ 0.48708115, 4.65643912, 60.66488088, 2.75453539,  1.88284155, 71.56193699] #IC on the attractor
    IC_outside = [200, 0, -50, 120, 0, -10]  #IC decays and settles on the attractor
    IC_1 = [40, 60, 70, 50, 55, 65]  #Artbitrarily selected IC

    IC_test = [80, 40, 20, 25, 50, 60]  #More such test trajectories can be entered below and tested for using num_test in main code

    IC = np.transpose([IC_inside, IC_limit, IC_outside, IC_1, IC_test])
    return IC

def Duffing(num_train,num_test):

    limits = 2
    
    nx = np.int(np.ceil(np.sqrt(num_train)))
    ny = nx
    x = np.linspace(-limits,limits,nx)
    y = np.linspace(-limits,limits,ny)

    for j in np.arange(ny):
        for i in np.arange(nx):
            if i == 0 and j == 0:
                IC_train = np.array([x[0], y[0]])
                IC_train = IC_train[:,np.newaxis]
            else:
                pt = np.array([x[i], y[j]])
                IC_train = np.append(IC_train, pt[:,np.newaxis], axis = 1)
#     IC_train = limits*np.random.rand(2,num_train)
    IC_test = limits*np.random.rand(2,num_test) # - 4)
#     for j in (0,ny-1):
#         for i in (0,nx-1):
#             pt = np.array([x[i], y[j]])
#             IC_test = np.append(IC_test,  pt[:,np.newaxis], axis = 1)
              
    return np.append(IC_train, IC_test, axis = 1)