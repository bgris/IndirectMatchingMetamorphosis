#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:15:49 2018

@author: bgris
"""

import odl
import numpy as np
import os
import IndirectMatchingMetamorphosis.Metamorphosis as meta
import IndirectMatchingMetamorphosis.Optimizer as opt


##%%
namepath= 'barbara'
#namepath= 'gris'

## Data parameters
index_name_template = 0
index_name_ground_truth = 0

index_angle = 2
index_maxangle = 0
index_noise = 2

## The parameter for kernel function
#sigmalist = [0.3, 0.6, 1., 2., 3.0, 4., 5., 10.]
#name_sigma_list=['3e_1','6e_1', '1', '2', '3','4', '5', '10' ]
sigmalist = [2]
name_sigma_list=['2']

niter=200
epsVinit=0.02
epsZinit=0.002
## Give regularization parameter
#explist = [1, 3, 5, 7]
#lamblist = [1e-1, 1e-3, 1e-5,1e-7]
#name_lamb_list =['1e_' + str(ex) for ex in explist]
#taulist = [1e-1, 1e-3, 1e-5,1e-7]
#name_tau_list =['1e_' + str(ex) for ex in explist]
#
explistlamb = [5]
lamblist = [1e-5]
name_lamb_list =['1e_5']
explisttau = [5]
taulist = [1e-5]
name_tau_list =['1e_5']
# Give the number of time points
time_itvs = 20
nb_time_point_int=time_itvs




name_list_template = ['SheppLogan0']
name_list_ground_truth = ['SheppLogan8_deformed']
num_angles_list = [10, 50, 100, 20, 30]
maxiangle_list = ['pi', '0_25pi']
max_angle_list = [np.pi, 0.25*np.pi]
noise_level_list = [0.0, 0.05, 0.25]
noi_list = ['0', '0_05', '0_25']

name_val_template = name_list_template[index_name_template]
name_val = name_list_ground_truth[index_name_ground_truth]
num_angles = num_angles_list[index_angle]
maxiangle = maxiangle_list[index_maxangle]
max_angle = max_angle_list[index_maxangle]
noise_level = noise_level_list[index_noise]
noi = noi_list[index_noise]
min_angle = 0.0


for lamb, name_lamb in zip(lamblist, name_lamb_list):
    for tau, name_tau in zip(taulist, name_tau_list):
        for sigma, name_sigma in zip(sigmalist, name_sigma_list):
            print('lamb = {}, tau = {}, sigma = {}'.format(lamb, tau, sigma))
            epsV = epsVinit
            epsZ = epsZinit
            
            name_exp = name_val + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_'
            name_exp += maxiangle + '_noise_' + noi
            
            
            path_data = '/home/' + namepath + '/data/Metamorphosis/test8/'
            path_result_init = '/home/' + namepath + '/Results/Metamorphosis/test9/'
            #path_result_init = '/home/bgris/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/Results/test2/'
            path_result = path_result_init + name_exp + '__sigma_' + name_sigma + '__lamb_'
            path_result += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'
            
            
            
            #path_result_init_dropbox = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/Results_ODE/test8/'
            #path_result_dropbox = path_result_init_dropbox + name_exp + '__sigma_' + name_sigma + '__lamb_'
            #path_result_dropbox += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'
            
            
            
            # Discrete reconstruction space: discretized functions on the rectangle
            rec_space = odl.uniform_discr(
                min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
                dtype='float32', interp='linear')
            
            
            name_ground_truth = path_data + name_val
            ground_truth = rec_space.element(np.loadtxt(name_ground_truth))
            
            name_template = path_data + name_val_template
            template = rec_space.element(np.loadtxt(name_template))
            
            
            
            # Give kernel function
            def kernel(x):
                scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
                return np.exp(-sum(scaled))
            
            
            
            ## Create forward operator
            ## Create the uniformly distributed directions
            angle_partition = odl.uniform_partition(min_angle, max_angle, num_angles,
                                                nodes_on_bdry=[(True, True)])
            
            ## Create 2-D projection domain
            ## The length should be 1.5 times of that of the reconstruction space
            detector_partition = odl.uniform_partition(-24, 24, int(round(rec_space.shape[0]*np.sqrt(2))))
            
            ## Create 2-D parallel projection geometry
            geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
            
            ## Ray transform aka forward projection. We use ASTRA CUDA backend.
            forward_op = odl.tomo.RayTransform(rec_space, geometry, impl='astra_cpu')
            
            
            ## load data
            
            data_load = forward_op.range.element(np.loadtxt(path_data + name_exp))
            
            
            
            data=[data_load]
            #data=[proj_data]
            data_time_points=np.array([1])
            forward_operators=[forward_op]
            Norm=[odl.solvers.L2NormSquared(forward_op.range)]
            
            
            ##%% Define energy operator
            functional=meta.TemporalAttachmentMetamorphosisGeom(nb_time_point_int,
                                        lamb,tau,template ,data,
                                        data_time_points, forward_operators,Norm, kernel,
                                        domain=None)
            
            
            ##%%
            ##%% Gradient descent
            
            X_init=functional.domain.zero()
            X_final = opt.GradientDescent(niter, epsV, epsZ, functional, X_init)

            
            ##%% Compute estimated trajectory
            image_list_data=functional.ComputeMetamorphosis(X_final[0],X_final[1])
            
            
            image_list=functional.ComputeMetamorphosisListInt(X_final[0],X_final[1])
            
            deformation_evo=odl.deform.ShootTemplateFromVectorFields(X_final[0], template)
            
            zeta_transp=odl.deform.ShootSourceTermBackwardlist(X_final[0],X_final[1])
            
            image_evol=odl.deform.IntegrateTemplateEvol(template,zeta_transp,0,nb_time_point_int)
            ##%%
            mini=-1
            maxi=1
            #
            
            
            #
            ##%% save results
            
            os.mkdir(path_result)
            #os.mkdir(path_result_dropbox)
            
            
            for i in range(nb_time_point_int + 1):
                np.savetxt(path_result + 'Image_t_' + str(i), image_list[i])
                np.savetxt(path_result + 'TemplatePart_t_' + str(i), image_evol[i])
                np.savetxt(path_result + 'DeformationPart_t_' + str(i), deformation_evo[i])
            #
#
