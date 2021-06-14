#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:27:44 2020

@author: ptarrio


  ; ***********************************************************************************************************************
  ; This procedure calculates the photometric redshift of a set of galaxies using
  ; the training set and the photometric redshift estimation method described in
  ; Tarrío and Zarattini 2020.
  ;
  ; *****************
  ; INPUT PARAMETERS:
  ; *****************
  ; - dir_name_cat_train: Full path and name of the training set (FITS file downloaded with this code)
  ; - dir_name_cat_test:  Full path and name of the set of galaxies for which we will calculate z_phot
  ;                       This set should be in FITS format. 
  ;                       It should contain a catalogue with any number of galaxies. 
  ;                       For each galaxy, it should provide the 9 features to be used for the computation 
  ;                       of the photometric redshift. Two options are admitted:
  ;                       - For aperture colours: R_KRON,GR_APER,RI_APER,IZ_APER,ZY_APER, w1mpro, w2mpro, w3mpro, w4mpro
  ;                       - For Kron colours:     R_KRON,GR_KRON,RI_KRON,IZ_KRON,ZY_KRON, w1mpro, w2mpro, w3mpro, w4mpro 
  ;                       If a feature is not available, it should be set to -999
  ;                       The user may want to use prepare_test_galaxies.pro to create this file
  ; - dir_name_cat_out:   Full path and name of the output catalogue (where results will be saved)
  ;
  ; *****************
  ; KEYWORDS:
  ; *****************
  ; - feat_type:  Set of features to be used to compute the photometric redshift. Two options are available:
  ;               - 'aper': The 5 features to be used are the g-r, r-i, i-z, and z-y aperture colours and the r-band Kron magnitude 
  ;               - 'kron': The 5 features to be used are the g-r, r-i, i-z, and z-y Kron colours and the r-band Kron magnitude
  ;               The default value is 'aper'.
  ;               See Tarrío and Zarattini 2020 for details on how these colours are defined.
  ;
  ; - train_type: Defines the subset of the full training set that will be used to compute the photometric redshift when a
  ;               feature is missing from a galaxy. Two options are available:
  ;               - 'T9': Subset of the full training set that has the 9 features available
  ;               - 'T8': Subset of the full training set that has the same 8 features as the galaxy
  ;               The default value is 'T5'.
  ;               See Tarrío and Zarattini 2020 for details more details on these definitions.
  ;
  ; - std_type:   Defines how the standardization of the features is done. The standardization of the features
  ;               is done by subtracting the mean and dividing by the standard deviation of the features in a
  ;               given subset of the training set. This keyword defines the subset to be used for calculating
  ;               the mean and standard deviation to be applied in the standardization. Two options are available:
  ;               - 'T9': Subset of the full training set that has the 9 features available
  ;               - 'T1': For each feature, we take the subset of the full training set where that feature is available
  ;               The default value is 'T9'.
  ;               The standardization applies to training set galaxies as well as to the test galaxies.
  ;
  ; - use_9feat:  To compute (1) or not (0) the photometric redshift of galaxies that have the 9 features available.
  ;               The default value is 1
  ;
  ; - use_8feat:  To compute (1) or not (0) the photometric redshift of galaxies that have only 8 features available.
  ;               The default value is 1
  ;               
  ; - Nneigh:     Number of neighbours that will be used in the local linear regression (k in Tarrío and Zarattini 2020)
  ;               The default value is 100
  ;
  ; *****************
  ; OUTPUT:
  ; *****************
  ; The output catalogue contains the same number of galaxies as the input catalogue, in the same order.
  ; For each galaxy, the following fields are provided:
  ; - z_phot: photometric redshift
  ; - z_phot_err: error in the photometric redshift
  ; - n_feat: number of features that were used to calculate the photometric redshift
  ; - flag_extrapolate: flag to indicate if the photometric redshift is calculated via an interpolation (0)
  ;                     or an extrapolation (1) of the training features. Extrapolation occurs when the
  ;                     test galaxy lies outside the bounding box of its k nearest neighbours (in the feature space).
  ;                     See Tarrío and Zarattini 2020 for details more details on this flag.
  ;
  ; *****************
  ; RESTRICTIONS:
  ; *****************
  ; Not fully tested yet. Report any bugs to Paula Tarrío
  ; p.tarrio@oan.es
  ;
  ; *****************
  ; REVISION HISTORY
  ; *****************
  ; Written, Paula Tarrío, July 20 2020
  ;
  ; ***********************************************************************************************************************



"""

import numpy as np
from astropy.io import fits
from linear_regression_zphot_tarrio import linear_regression_zphot_tarrio


def compute_photo_z_pan_wise_tarrio(dir_name_cat_train, dir_name_cat_test, dir_name_cat_out, feat_type='aper', train_type='T9', std_type='T9', use_9feat=1, use_8feat=1, Nneigh=100):

    # ----------------------------------------------------------
    # Checking keywords
    # ----------------------------------------------------------
    badinput = 0
    if (feat_type != 'aper' and feat_type != 'kron'):
        print('Error: The selected feat_type is not correct: It should be kron or aper (default).')
        badinput = 1

    if (train_type != 'T8' and train_type != 'T9'):
        print('Error: The selected train_type is not correct: It should be T4 or T5 (default).')
        badinput = 1

    if (std_type != 'T1' and std_type != 'T9'):
        print('Error: The selected std_type is not correct: It should be T1 or T5 (default).')
        badinput = 1

    if (use_9feat != 1 and use_9feat != 0):
        print('Error: The selected use_5feat is not correct: It should be 0 or 1 (default).')
        badinput = 1

    if (use_8feat != 1 and use_8feat != 0):
        print('Error: The selected use_4feat is not correct: It should be 0 or 1 (default).')
        badinput = 1

    if (Nneigh < 1):
        print('Error: The selected Nneigh is not correct: It should be at least 1 (100 recommended)')
        badinput = 1
    # ----------------------------------------------------------

    if badinput == 0:
        
        # Open training set
        cat = fits.getdata(dir_name_cat_train, 1) 
        Ntot = len(cat)
        
        # Select the features of the training set:
        if feat_type == 'aper':
            features_training_notnorm = np.concatenate((cat['gr_aper'][:, None],cat['ri_aper'][:, None],cat['iz_aper'][:, None],cat['zy_aper'][:, None],cat['r_kron'][:, None], cat['w1mpro'][:, None], cat['w2mpro'][:, None], cat['w3mpro'][:, None], cat['w4mpro'][:, None]),axis=1)
        if feat_type == 'kron':
            features_training_notnorm = np.concatenate((cat['gr_kron'][:, None],cat['ri_kron'][:, None],cat['iz_kron'][:, None],cat['zy_kron'][:, None],cat['r_kron'][:, None], cat['w1mpro'][:, None], cat['w2mpro'][:, None], cat['w3mpro'][:, None], cat['w4mpro'][:, None]),axis=1)
        
        # Subsets of the training set
        index_T9 = np.logical_and.reduce((features_training_notnorm[:,0] > -100 , features_training_notnorm[:,1] > -100 , features_training_notnorm[:,2] > -100 , features_training_notnorm[:,3] > -100 , features_training_notnorm[:,4] > -100, features_training_notnorm[:,5] > -100, features_training_notnorm[:,6] > -100, features_training_notnorm[:,7] > -100, features_training_notnorm[:,8] > -100),axis=0)
        index_T8_0 = np.logical_and.reduce((features_training_notnorm[:,0] <= -100 , features_training_notnorm[:,1] > -100 , features_training_notnorm[:,2] > -100 , features_training_notnorm[:,3] > -100 , features_training_notnorm[:,4] > -100, features_training_notnorm[:,5] > -100, features_training_notnorm[:,6] > -100, features_training_notnorm[:,7] > -100, features_training_notnorm[:,8] > -100),axis=0)
        index_T8_1 = np.logical_and.reduce((features_training_notnorm[:,0] > -100 , features_training_notnorm[:,1] <= -100 , features_training_notnorm[:,2] > -100 , features_training_notnorm[:,3] > -100 , features_training_notnorm[:,4] > -100, features_training_notnorm[:,5] > -100, features_training_notnorm[:,6] > -100, features_training_notnorm[:,7] > -100, features_training_notnorm[:,8] > -100),axis=0)
        index_T8_2 = np.logical_and.reduce((features_training_notnorm[:,0] > -100 , features_training_notnorm[:,1] > -100 , features_training_notnorm[:,2] <= -100 , features_training_notnorm[:,3] > -100 , features_training_notnorm[:,4] > -100, features_training_notnorm[:,5] > -100, features_training_notnorm[:,6] > -100, features_training_notnorm[:,7] > -100, features_training_notnorm[:,8] > -100),axis=0)
        index_T8_3 = np.logical_and.reduce((features_training_notnorm[:,0] > -100 , features_training_notnorm[:,1] > -100 , features_training_notnorm[:,2] > -100 , features_training_notnorm[:,3] <= -100 , features_training_notnorm[:,4] > -100, features_training_notnorm[:,5] > -100, features_training_notnorm[:,6] > -100, features_training_notnorm[:,7] > -100, features_training_notnorm[:,8] > -100),axis=0)
        index_T8_4 = np.logical_and.reduce((features_training_notnorm[:,0] > -100 , features_training_notnorm[:,1] > -100 , features_training_notnorm[:,2] > -100 , features_training_notnorm[:,3] > -100 , features_training_notnorm[:,4] <= -100, features_training_notnorm[:,5] > -100, features_training_notnorm[:,6] > -100, features_training_notnorm[:,7] > -100, features_training_notnorm[:,8] > -100),axis=0)
        index_T8_5 = np.logical_and.reduce((features_training_notnorm[:,0] > -100 , features_training_notnorm[:,1] > -100 , features_training_notnorm[:,2] > -100 , features_training_notnorm[:,3] > -100 , features_training_notnorm[:,4] > -100, features_training_notnorm[:,5] <= -100, features_training_notnorm[:,6] > -100, features_training_notnorm[:,7] > -100, features_training_notnorm[:,8] > -100),axis=0)
        index_T8_6 = np.logical_and.reduce((features_training_notnorm[:,0] > -100 , features_training_notnorm[:,1] > -100 , features_training_notnorm[:,2] > -100 , features_training_notnorm[:,3] > -100 , features_training_notnorm[:,4] > -100, features_training_notnorm[:,5] > -100, features_training_notnorm[:,6] <= -100, features_training_notnorm[:,7] > -100, features_training_notnorm[:,8] > -100),axis=0)
        index_T8_7 = np.logical_and.reduce((features_training_notnorm[:,0] > -100 , features_training_notnorm[:,1] > -100 , features_training_notnorm[:,2] > -100 , features_training_notnorm[:,3] > -100 , features_training_notnorm[:,4] > -100, features_training_notnorm[:,5] > -100, features_training_notnorm[:,6] > -100, features_training_notnorm[:,7] <= -100, features_training_notnorm[:,8] > -100),axis=0)
        index_T8_8 = np.logical_and.reduce((features_training_notnorm[:,0] > -100 , features_training_notnorm[:,1] > -100 , features_training_notnorm[:,2] > -100 , features_training_notnorm[:,3] > -100 , features_training_notnorm[:,4] > -100, features_training_notnorm[:,5] > -100, features_training_notnorm[:,6] > -100, features_training_notnorm[:,7] > -100, features_training_notnorm[:,8] <= -100),axis=0)
        
        # Calculate the mean and standard deviation that will be used to standardize the features
        if std_type == 'T9':
            features_training_notnorm_formean = features_training_notnorm[index_T9]
            mu_features = features_training_notnorm_formean.mean(axis=0)
            sigma_features = features_training_notnorm_formean.std(axis=0)

        if std_type == 'T1':
            mu_gr = np.mean(features_training_notnorm[(features_training_notnorm[:,0] > -100), 0])
            mu_ri = np.mean(features_training_notnorm[(features_training_notnorm[:,1] > -100), 1])
            mu_iz = np.mean(features_training_notnorm[(features_training_notnorm[:,2] > -100), 2])
            mu_zy = np.mean(features_training_notnorm[(features_training_notnorm[:,3] > -100), 3])
            mu_r  = np.mean(features_training_notnorm[(features_training_notnorm[:,4] > -100), 4])
            mu_w1 = np.mean(features_training_notnorm[(features_training_notnorm[:,5] > -100), 5])
            mu_w2 = np.mean(features_training_notnorm[(features_training_notnorm[:,6] > -100), 6])
            mu_w3 = np.mean(features_training_notnorm[(features_training_notnorm[:,7] > -100), 7])
            mu_w4 = np.mean(features_training_notnorm[(features_training_notnorm[:,8] > -100), 8])
            sigma_gr = np.std(features_training_notnorm[(features_training_notnorm[:,0] > -100), 0])
            sigma_ri = np.std(features_training_notnorm[(features_training_notnorm[:,1] > -100), 1])
            sigma_iz = np.std(features_training_notnorm[(features_training_notnorm[:,2] > -100), 2])
            sigma_zy = np.std(features_training_notnorm[(features_training_notnorm[:,3] > -100), 3])
            sigma_r  = np.std(features_training_notnorm[(features_training_notnorm[:,4] > -100), 4])
            sigma_w1 = np.std(features_training_notnorm[(features_training_notnorm[:,5] > -100), 5])
            sigma_w2 = np.std(features_training_notnorm[(features_training_notnorm[:,6] > -100), 6])
            sigma_w3 = np.std(features_training_notnorm[(features_training_notnorm[:,7] > -100), 7])
            sigma_w4 = np.std(features_training_notnorm[(features_training_notnorm[:,8] > -100), 8])
            mu_features = [mu_gr,mu_ri,mu_iz,mu_zy,mu_r, mu_w1, mu_w2, mu_w3, mu_w4]
            sigma_features = [sigma_gr,sigma_ri,sigma_iz,sigma_zy,sigma_r, sigma_w1, sigma_w2, sigma_w3, sigma_w4]

        
        # We standardize the features of the training set:
        features_training = np.divide((features_training_notnorm - np.array([mu_features,]*Ntot)) , np.array([sigma_features,]*Ntot))

          
        # Open test set
        cat_test = fits.getdata(dir_name_cat_test, 1)
        Ntest = len(cat_test)
        
        if Ntest > 0:
        
            # Select features for the test galaxies
            if feat_type == 'aper':
                features_test_notnorm = np.concatenate((cat_test['gr_aper'][:, None],cat_test['ri_aper'][:, None],cat_test['iz_aper'][:, None],cat_test['zy_aper'][:, None],cat_test['r_kron'][:, None], cat_test['w1mpro'][:, None], cat_test['w2mpro'][:, None], cat_test['w3mpro'][:, None], cat_test['w4mpro'][:, None]),axis=1)
            if feat_type == 'kron':
                features_test_notnorm = np.concatenate((cat_test['gr_kron'][:, None],cat_test['ri_kron'][:, None],cat_test['iz_kron'][:, None],cat_test['zy_kron'][:, None],cat_test['r_kron'][:, None], cat_test['w1mpro'][:, None], cat_test['w2mpro'][:, None], cat_test['w3mpro'][:, None], cat_test['w4mpro'][:, None]),axis=1)
            # We normalize the features of the test set:
            features_test = np.divide((features_test_notnorm - np.array([mu_features,]*Ntest)) , np.array([sigma_features,]*Ntest))

            # Subsets of the test set
            good_9feat = np.logical_and.reduce((features_test[:,0] > -100 , features_test[:,1] > -100 , features_test[:,2] > -100 , features_test[:,3] > -100 , features_test[:,4] > -100, features_test[:,5] > -100 , features_test[:,6] > -100 , features_test[:,7] > -100 , features_test[:,8] > -100 ),axis=0)
            good_0 = np.logical_and.reduce((features_test[:,0] <= -100 , features_test[:,1] > -100 , features_test[:,2] > -100 , features_test[:,3] > -100 , features_test[:,4] > -100, features_test[:,5] > -100 , features_test[:,6] > -100 , features_test[:,7] > -100 , features_test[:,8] > -100 ),axis=0)
            good_1 = np.logical_and.reduce((features_test[:,0] > -100 , features_test[:,1] <= -100 , features_test[:,2] > -100 , features_test[:,3] > -100 , features_test[:,4] > -100, features_test[:,5] > -100 , features_test[:,6] > -100 , features_test[:,7] > -100 , features_test[:,8] > -100 ),axis=0)
            good_2 = np.logical_and.reduce((features_test[:,0] > -100 , features_test[:,1] > -100 , features_test[:,2] <= -100 , features_test[:,3] > -100 , features_test[:,4] > -100, features_test[:,5] > -100 , features_test[:,6] > -100 , features_test[:,7] > -100 , features_test[:,8] > -100 ),axis=0)
            good_3 = np.logical_and.reduce((features_test[:,0] > -100 , features_test[:,1] > -100 , features_test[:,2] > -100 , features_test[:,3] <= -100 , features_test[:,4] > -100, features_test[:,5] > -100 , features_test[:,6] > -100 , features_test[:,7] > -100 , features_test[:,8] > -100 ),axis=0)
            good_4 = np.logical_and.reduce((features_test[:,0] > -100 , features_test[:,1] > -100 , features_test[:,2] > -100 , features_test[:,3] > -100 , features_test[:,4] <= -100, features_test[:,5] > -100 , features_test[:,6] > -100 , features_test[:,7] > -100 , features_test[:,8] > -100 ),axis=0)
            good_5 = np.logical_and.reduce((features_test[:,0] > -100 , features_test[:,1] > -100 , features_test[:,2] > -100 , features_test[:,3] > -100 , features_test[:,4] > -100, features_test[:,5] <= -100 , features_test[:,6] > -100 , features_test[:,7] > -100 , features_test[:,8] > -100 ),axis=0)
            good_6 = np.logical_and.reduce((features_test[:,0] > -100 , features_test[:,1] > -100 , features_test[:,2] > -100 , features_test[:,3] > -100 , features_test[:,4] > -100, features_test[:,5] > -100 , features_test[:,6] <= -100 , features_test[:,7] > -100 , features_test[:,8] > -100 ),axis=0)
            good_7 = np.logical_and.reduce((features_test[:,0] > -100 , features_test[:,1] > -100 , features_test[:,2] > -100 , features_test[:,3] > -100 , features_test[:,4] > -100, features_test[:,5] > -100 , features_test[:,6] > -100 , features_test[:,7] <= -100 , features_test[:,8] > -100 ),axis=0)
            good_8 = np.logical_and.reduce((features_test[:,0] > -100 , features_test[:,1] > -100 , features_test[:,2] > -100 , features_test[:,3] > -100 , features_test[:,4] > -100, features_test[:,5] > -100 , features_test[:,6] > -100 , features_test[:,7] > -100 , features_test[:,8] <= -100 ),axis=0)
            n_9 = np.sum(good_9feat)
            n_8_0 = np.sum(good_0)
            n_8_1 = np.sum(good_1)
            n_8_2 = np.sum(good_2)
            n_8_3 = np.sum(good_3)
            n_8_4 = np.sum(good_4)
            n_8_5 = np.sum(good_5)
            n_8_6 = np.sum(good_6)
            n_8_7 = np.sum(good_7)
            n_8_8 = np.sum(good_8)

            # Linear regression to calculate z_phot for the galaxies in the test set that have the 5 features available
            if use_9feat == 1:
                index_training = index_T9
                if n_9 > 0:
                    aux=1
                    result = linear_regression_zphot_tarrio(features_training[index_training,:], features_test[good_9feat,:], cat['z_spec'][index_training], Nneigh)
                    z_phot_9feat = result[0]
                    z_phot_err_9feat  = result[1]
                    flag_extrapolate_9feat = result[2]

        
            # Linear regression to calculate z_phot for the galaxies in the test set that have only 4 features available
            if use_8feat == 1:
                if train_type == 'T9':
                    index_training = index_T9
        
                ftr = features_training[:,[1,2,3,4,5,6,7,8]]
                fte = features_test[:,[1,2,3,4,5,6,7,8]]
                if train_type == 'T8':
                    index_training = index_T8_0
                if n_8_0 > 0:
                    result = linear_regression_zphot_tarrio(ftr[index_training,:], fte[good_0,:], cat['z_spec'][index_training], Nneigh)
                    z_phot_0 = result[0]
                    z_phot_err_0  = result[1]
                    flag_extrapolate_0 = result[2]
        
                ftr = features_training[:,[0,2,3,4,5,6,7,8]]
                fte = features_test[:,[0,2,3,4,5,6,7,8]]
                if train_type == 'T8':
                    index_training = index_T8_1
                if n_8_1 > 0:
                    result = linear_regression_zphot_tarrio(ftr[index_training,:], fte[good_1,:], cat['z_spec'][index_training], Nneigh)
                    z_phot_1 = result[0]
                    z_phot_err_1  = result[1]
                    flag_extrapolate_1 = result[2]
                    
                ftr = features_training[:,[0,1,3,4,5,6,7,8]]
                fte = features_test[:,[0,1,3,4,5,6,7,8]]
                if train_type == 'T8':
                    index_training = index_T8_2
                if n_8_2 > 0:
                    result = linear_regression_zphot_tarrio(ftr[index_training,:], fte[good_2,:], cat['z_spec'][index_training], Nneigh)
                    z_phot_2 = result[0]
                    z_phot_err_2  = result[1]
                    flag_extrapolate_2 = result[2]
                    
                ftr = features_training[:,[0,1,2,4,5,6,7,8]]
                fte = features_test[:,[0,1,2,4,5,6,7,8]]
                if train_type == 'T8':
                    index_training = index_T8_3
                if n_8_3 > 0:
                    result = linear_regression_zphot_tarrio(ftr[index_training,:], fte[good_3,:], cat['z_spec'][index_training], Nneigh)
                    z_phot_3 = result[0]
                    z_phot_err_3  = result[1]
                    flag_extrapolate_3 = result[2]
                
                ftr = features_training[:,[0,1,2,3,5,6,7,8]]
                fte = features_test[:,[0,1,2,3,5,6,7,8]]
                if train_type == 'T8':
                    index_training = index_T8_4
                if n_8_4 > 0:
                    result = linear_regression_zphot_tarrio(ftr[index_training,:], fte[good_4,:], cat['z_spec'][index_training], Nneigh)
                    z_phot_4 = result[0]
                    z_phot_err_4  = result[1]
                    flag_extrapolate_4 = result[2]
                
                ftr = features_training[:,[0,1,2,3,4,6,7,8]]
                fte = features_test[:,[0,1,2,3,4,6,7,8]]
                if train_type == 'T8':
                    index_training = index_T8_5
                if n_8_5 > 0:
                    result = linear_regression_zphot_tarrio(ftr[index_training,:], fte[good_5,:], cat['z_spec'][index_training], Nneigh)
                    z_phot_5 = result[0]
                    z_phot_err_5  = result[1]
                    flag_extrapolate_5 = result[2]
                
                ftr = features_training[:,[0,1,2,3,4,5,7,8]]
                fte = features_test[:,[0,1,2,3,4,5,7,8]]
                if train_type == 'T8':
                    index_training = index_T8_6
                if n_8_6 > 0:
                    result = linear_regression_zphot_tarrio(ftr[index_training,:], fte[good_6,:], cat['z_spec'][index_training], Nneigh)
                    z_phot_6 = result[0]
                    z_phot_err_6  = result[1]
                    flag_extrapolate_6 = result[2]
                    
                ftr = features_training[:,[0,1,2,3,4,5,6,8]]
                fte = features_test[:,[0,1,2,3,4,5,6,8]]
                if train_type == 'T8':
                    index_training = index_T8_7
                if n_8_7 > 0:
                    result = linear_regression_zphot_tarrio(ftr[index_training,:], fte[good_7,:], cat['z_spec'][index_training], Nneigh)
                    z_phot_7 = result[0]
                    z_phot_err_7  = result[1]
                    flag_extrapolate_7 = result[2]
                    
                ftr = features_training[:,[0,1,2,3,4,5,6,7]]
                fte = features_test[:,[0,1,2,3,4,5,6,7]]
                if train_type == 'T8':
                    index_training = index_T8_8
                if n_8_8 > 0:
                    result = linear_regression_zphot_tarrio(ftr[index_training,:], fte[good_8,:], cat['z_spec'][index_training], Nneigh)
                    z_phot_8 = result[0]
                    z_phot_err_8  = result[1]
                    flag_extrapolate_8 = result[2]
           
        
        
            # Define output catalogue structure
            col1 = fits.Column(name='z_phot', format='D', array=-np.ones(len(cat_test)))
            col2 = fits.Column(name='z_phot_err', format='D', array=-np.ones(len(cat_test)))
            col3 = fits.Column(name='n_feat', format='I', array=np.zeros(len(cat_test)))
            col4 = fits.Column(name='flag_extrapolate', format='I', array=-np.ones(len(cat_test)))
            hdu = fits.BinTableHDU.from_columns([col1 , col2 , col3 , col4])
            cat_out = hdu.data

            # Save results in output catalogue structure
            if use_9feat == 1:
                cat_out['z_phot'][good_9feat] = z_phot_9feat
                cat_out['z_phot_err'][good_9feat] = z_phot_err_9feat
                cat_out['n_feat'][good_9feat] = 9
                cat_out['flag_extrapolate'][good_9feat] = flag_extrapolate_9feat
            if use_8feat == 1:
                if n_8_0 > 0:
                    cat_out['z_phot'][good_0] = z_phot_0
                    cat_out['z_phot_err'][good_0] = z_phot_err_0
                    cat_out['flag_extrapolate'][good_0] = flag_extrapolate_0
                    cat_out['n_feat'][good_0] = 8
            if use_8feat == 1:
                if n_8_1 > 0:
                    cat_out['z_phot'][good_1] = z_phot_1
                    cat_out['z_phot_err'][good_1] = z_phot_err_1
                    cat_out['flag_extrapolate'][good_1] = flag_extrapolate_1
                    cat_out['n_feat'][good_1] = 8
            if use_8feat == 1:
                if n_8_2 > 0:
                    cat_out['z_phot'][good_2] = z_phot_2
                    cat_out['z_phot_err'][good_2] = z_phot_err_2
                    cat_out['flag_extrapolate'][good_2] = flag_extrapolate_2
                    cat_out['n_feat'][good_2] = 8
            if use_8feat == 1:
                if n_8_3 > 0:
                    cat_out['z_phot'][good_3] = z_phot_3
                    cat_out['z_phot_err'][good_3] = z_phot_err_3
                    cat_out['flag_extrapolate'][good_3] = flag_extrapolate_3
                    cat_out['n_feat'][good_3] = 8
            if use_8feat == 1:
                if n_8_4 > 0:
                    cat_out['z_phot'][good_4] = z_phot_4
                    cat_out['z_phot_err'][good_4] = z_phot_err_4
                    cat_out['flag_extrapolate'][good_4] = flag_extrapolate_4
                    cat_out['n_feat'][good_4] = 8
            if use_8feat == 1:
                if n_8_5 > 0:
                    cat_out['z_phot'][good_5] = z_phot_5
                    cat_out['z_phot_err'][good_5] = z_phot_err_5
                    cat_out['flag_extrapolate'][good_5] = flag_extrapolate_5
                    cat_out['n_feat'][good_5] = 8
            if use_8feat == 1:
                if n_8_6 > 0:
                    cat_out['z_phot'][good_6] = z_phot_6
                    cat_out['z_phot_err'][good_6] = z_phot_err_6
                    cat_out['flag_extrapolate'][good_6] = flag_extrapolate_6
                    cat_out['n_feat'][good_6] = 8
            if use_8feat == 1:
                if n_8_7 > 0:
                    cat_out['z_phot'][good_7] = z_phot_7
                    cat_out['z_phot_err'][good_7] = z_phot_err_7
                    cat_out['flag_extrapolate'][good_7] = flag_extrapolate_7
                    cat_out['n_feat'][good_7] = 8
            if use_8feat == 1:
                if n_8_8 > 0:
                    cat_out['z_phot'][good_8] = z_phot_8
                    cat_out['z_phot_err'][good_8] = z_phot_err_8
                    cat_out['flag_extrapolate'][good_8] = flag_extrapolate_8
                    cat_out['n_feat'][good_8] = 8
                
        
            # Save output catalogue in FITS format
            hdu.writeto(dir_name_cat_out, overwrite=True)
            

