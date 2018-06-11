#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:39:20 2018

@author: bgris

different forward operator for each time

"""

import odl
import numpy as np
import os
##%%
#namepath= 'barbara'
#namepath= 'bgris'
namepath= 'gris'

## Data parameters
index_name_template = 0
index_name_ground_truth = 0
nb_data_point = 10
indexes_name_ground_truth_timepoints = [i + 1 for i in range(nb_data_point)]
data_time_points=np.array([ (i+1)/10 for i in range(nb_data_point)])

index_angle = 0 
index_maxangle = 0
index_noise = 2



## The parameter for kernel function
sigma = 2.0
name_sigma=str(int(sigma))

niter = 100 
epsV=0.02
epsZ=0.0002
## Give regularization parameter
lamb = 1e-5
name_lamb='1e_' + str(-int(np.log(lamb)/np.log(10)))
tau = 1e-6
name_tau='1e_' + str(-int(np.log(tau)/np.log(10)))

# Give the number of time points
time_itvs = 10

nb_time_point_int=time_itvs
numtest = 16

nb_data_points = len(indexes_name_ground_truth_timepoints)

name_list_template = [ 'temporal__t_0']
name_list_ground_truth = [ 'temporal__t_']

num_angles_list = [10, 20, 30, 50, 100]
maxiangle_list = ['pi', '0_25pi']
max_angle_list = [np.pi, 0.25*np.pi]
noise_level_list = [0.0, 0.05, 0.25]
noi_list = ['0', '0_05', '0_25']
miniangle_list = ['0']
min_angle_list = [0]



name_val_template = name_list_template[index_name_template]
name_val = name_list_ground_truth[index_name_ground_truth]
num_angles = num_angles_list[index_angle]
maxiangle = maxiangle_list[index_maxangle]
max_angle = max_angle_list[index_maxangle]
noise_level = noise_level_list[index_noise]
noi = noi_list[index_noise]
min_angle = 0.0
miniangle = '0'

name_exp = name_val + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_' + maxiangle + '_noise_' + noi
name_list = [name_val + str(i) for i in range(nb_data_points)]


path_data = 'data/SheppLogan/'
path_result_init = '/home/' + namepath + '/Results/Metamorphosis/test' + str(numtest) + '/'
path_result = path_result_init + name_exp + '__sigma_' + name_sigma + '__lamb_'
path_result += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + 'datatimepoints' + str(len(data_time_points)) + '/'


# Discrete reconstruction space: discretized functions on the rectangle
rec_space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')


name_ground_truth = [path_data + name_val + str(indexes_name_ground_truth_timepoints[i])  for i in range(nb_data_points)]
ground_truth_list = [rec_space.element(np.loadtxt(name_ground_truth[i])) for i in range(nb_data_points)]

name_template = path_data + name_val_template
template = rec_space.element(np.loadtxt(name_template))



# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))



namedata =  path_data + name_val + 'num_angles_' + str(num_angles) + '_min_angle_' + miniangle + '_max_angle_'
namedata += maxiangle + 'randompartition'

list_angles_tot = np.loadtxt(namedata + 'angles')
detector_partition = odl.uniform_partition(-24, 24, int(round(rec_space.shape[0]*np.sqrt(2))))

#array_data = np.empty((num_angles * data_time_points, detector_partition.points().shape[0]))
list_data = []
list_forward_op = []
for i in range(len(data_time_points)):
    name_ground_truth = path_data + name_val + str(i+1)
    name = name_ground_truth + 'num_angles_' + str(num_angles) + '_min_angle_' + miniangle + '_max_angle_'
    name += maxiangle + '_noise_' + noi 
    name += 'randompartition'
#    array_data[i*num_angles : (i+1)*num_angles, :] = np.loadtxt(name)
    inter_angle = odl.set.domain.IntervalProd(list_angles_tot[i * num_angles], list_angles_tot[(i+1) * num_angles - 1])
    grid_tmp = odl.discr.grid.RectGrid(list_angles_tot[i*num_angles : (i+1)*num_angles])
    angle_partition = odl.discr.partition.RectPartition(inter_angle, grid_tmp)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
    list_forward_op.append(odl.tomo.RayTransform(rec_space, geometry, impl='astra_cpu'))
    list_data.append(list_forward_op[i].range.element(np.loadtxt(name)))


Norm_list=[odl.solvers.L2NormSquared(list_forward_op[i].range) for i in range(len(data_time_points))]

##%%
##%% Define energy operator
import IndirectMatchingMetamorphosis.Metamorphosis as meta
functional = meta.TemporalAttachmentMetamorphosisGeom(nb_time_point_int,
                            lamb,tau,template ,list_data,
                            data_time_points, list_forward_op, Norm_list, kernel,
                            domain=None)


##%%
##%% Gradient descent

X_init=functional.domain.zero()
import IndirectMatchingMetamorphosis.Optimizer as opt
X_final = opt.GradientDescent(niter, epsV, epsZ, functional, X_init)




##%% Compute estimated trajectory
image_list_data=functional.ComputeMetamorphosis(X_final[0],X_final[1])


image_list=functional.ComputeMetamorphosisListInt(X_final[0],X_final[1])

deformation_evo=meta.ShootTemplateFromVectorFields(X_final[0], template)

zeta_transp=meta.ShootSourceTermBackwardlist(X_final[0],X_final[1])

image_evol=meta.IntegrateTemplateEvol(template,zeta_transp,0,nb_time_point_int)

##%% save results
os.mkdir(path_result)


for i in range(nb_time_point_int + 1):
    np.savetxt(path_result + 'Image_t_' + str(i), image_list[i])
    np.savetxt(path_result + 'TemplatePart_t_' + str(i), image_evol[i])
    np.savetxt(path_result + 'DeformationPart_t_' + str(i), deformation_evo[i])
#
