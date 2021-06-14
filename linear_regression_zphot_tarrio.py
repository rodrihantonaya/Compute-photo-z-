#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:12:50 2020

@author: ptarrio

---------------------------------------------
DESCRIPTION:
---------------------------------------------
Linear regression to calculate z_phot for all the elements in the test set


---------------------------------------------
CALLING SEQUENCE:
---------------------------------------------
To run this script from the terminal, go to its location and write:
  >  python linear_regression_zphot_tarrio, features_training, features_test, z_spec, Nneigh, z_phot_err, flag_extrapolation

"""

import numpy as np
    
def linear_regression_zphot_tarrio(features_training, features_test, z_spec, Nneigh):

    Ntrain = len(z_spec)
    Ntest = np.size(features_test,0)

    flag_extrapolation = np.zeros(Ntest, dtype='float')
    z_phot = np.zeros(Ntest, dtype='float')
    z_phot_err = np.zeros(Ntest, dtype='float')

    for i in range(Ntest):
        # Search for the 100 closest training points
        dist_to_training = np.sum((np.outer(np.ones(Ntrain, dtype='float'),features_test[i,:]) - features_training)**2,1)
        #neighbours = np.argpartition(dist_to_training, Nneigh)[0:Nneigh]
        print(i)
        # FAST SORT: we reduce the array size before to the sort
        index_short_dist = (dist_to_training < 0.02)
        ss = np.count_nonzero(index_short_dist)
        if ss < Nneigh:
            index_short_dist = (dist_to_training < 0.05)
            rr = np.count_nonzero(index_short_dist)
            if rr < Nneigh:
                index_short_dist = (dist_to_training < 0.1)
                qq = np.count_nonzero(index_short_dist)
                if qq < Nneigh:
                    index_short_dist = (dist_to_training < 0.5)
                    nn = np.count_nonzero(index_short_dist)
                    if nn < Nneigh:
                        index_short_dist = (dist_to_training < 1.)
                        mm = np.count_nonzero(index_short_dist)
                        if mm < Nneigh:
                            index_short_dist = (dist_to_training < 10.)
                            oo = np.count_nonzero(index_short_dist)
                            if oo < Nneigh:
                                index_short_dist = (dist_to_training < 210.)
                                pp = np.count_nonzero(index_short_dist)
                                if pp < Nneigh:
                                    index_short_dist = np.ones(len(dist_to_training)) 
        aux = dist_to_training[index_short_dist]
        index_aux = np.where(index_short_dist)[0]
        if len(aux)==Nneigh:
            neighbours = index_aux
        else:
            index_aux_sorted = np.argpartition(aux, Nneigh)[0:Nneigh]
            neighbours = index_aux[index_aux_sorted]

        
        # Linear regression to estimate z_phot
        xx = features_training[neighbours,:]
        yy = z_spec[neighbours]
        H = np.concatenate((xx,np.ones(Nneigh, dtype='float')[:, None]),axis=1)
        print(H)
        res = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(H),H)),np.transpose(H)),yy)
        z_phot[i] = np.sum(np.dot(res,np.append(features_test[i,:], 1)))
        z_phot_neigh = np.dot(H,res)
        z_phot_err[i] = np.sqrt(np.sum((yy-z_phot_neigh)**2)/Nneigh)
        if np.any(np.logical_or((features_test[i,:] < np.amin(xx,axis=0)),(features_test[i,:] > np.amax(xx,axis=0)))):
            flag_extrapolation[i] = 1
        
        # Refinement (see Beck et al 2016, section 2.1, penultimate paragraph):
        neigh_index = ( (yy-z_phot_neigh) <= 3*z_phot_err[i]) # Neighbours not outliers
        ng = np.count_nonzero(neigh_index)
        if ng < Nneigh and ng > 0:
            xx = xx[neigh_index,:]
            yy = yy[neigh_index]
            H = np.concatenate((xx,np.ones(ng, dtype='float')[:, None]),axis=1)
            res = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(H),H)),np.transpose(H)),yy)
            z_phot[i] = np.sum(np.dot(res,np.append(features_test[i,:], 1)))
            z_phot_neigh = np.dot(H,res)
            z_phot_err[i] = np.sqrt(np.sum((yy-z_phot_neigh)**2)/ng)
            if np.any(np.logical_or((features_test[i,:] < np.amin(xx,axis=0)),(features_test[i,:] > np.amax(xx,axis=0)))):
                flag_extrapolation[i] = 1
    
    return z_phot, z_phot_err, flag_extrapolation


