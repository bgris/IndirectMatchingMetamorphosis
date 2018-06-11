#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:22:34 2017

@author: bgris
"""


import odl
import numpy as np
from matplotlib import pylab as plt



#%%
########%% Load template and ground truth for triangles

path='data/Triangles/'
path_save_data = '/home/barbara/data/Metamorphosis/test1/'


space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')


template = space.element(np.loadtxt(path + 'template_values__0__1'))
ground_truth = space.element(np.loadtxt(path + 'ground_truth_values__0_2__0_9'))




#%% Generate data or triangles
name_list = ['ground_truth_values__0_2__0_9', 'ground_truth_values__0__1']
num_angles_list = [10, 50, 100]
#num_angles_list = [6]
maxiangle_list = ['pi', '0_25pi']
max_angle_list = [np.pi, 0.25*np.pi]
noise_level_list = [0.0, 0.05, 0.25]
noi_list = ['0', '0_05', '0_25']
min_angle = 0.0


for name_val in name_list:
    for num_angles in num_angles_list:
        for maxiangle, max_angle in zip(maxiangle_list, max_angle_list):
                for noi, noise_level in zip(noi_list, noise_level_list):
                        print(name_val)
                        print(num_angles)
                        print(maxiangle)
                        print(max_angle)
                        print(noi)
                        print(noise_level)
                        ## Create the uniformly distributed directions
                        angle_partition = odl.uniform_partition(min_angle, max_angle, num_angles,
                                                            nodes_on_bdry=[(True, True)])

                        ## Create 2-D projection domain
                        ## The length should be 1.5 times of that of the reconstruction space
                        detector_partition = odl.uniform_partition(-24, 24, int(round(space.shape[0]*np.sqrt(2))))

                        ## Create 2-D parallel projection geometry
                        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

                        ## Ray transform aka forward projection. We use ASTRA CUDA backend.
                        forward_op = odl.tomo.RayTransform(space, geometry, impl='astra_cpu')

                        name_ground_truth = path_save_data + name_val
                        name = name_ground_truth + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_'
                        name += maxiangle + '_noise_' + noi

                        noise = noise_level * odl.phantom.noise.white_noise(forward_op.range)

                        data = forward_op(ground_truth) + noise

                        #data.show()

                        np.savetxt(name, data)


#
#%%

#%%
########%% Load template and ground truth for SheppLogans


import odl
import numpy as np
from matplotlib import pylab as plt
path='data/SheppLogan/'
path_save_data = '/home/barbara/data/Metamorphosis/test9/'


space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')


template = space.element(np.loadtxt(path + 'SheppLogan0'))
ground_truth = space.element(np.loadtxt(path + 'SheppLogan8_deformed'))




#%% Generate data
name_list = ['SheppLogan8_deformed']
num_angles_list = [10, 50, 100]
maxiangle_list = ['pi', '0_25pi']
max_angle_list = [np.pi, 0.25*np.pi]
noise_level_list = [0.0, 0.05, 0.25]
noi_list = ['0', '0_05', '0_25']
min_angle = 0.0


for name_val in name_list:
    for num_angles in num_angles_list:
        for maxiangle, max_angle in zip(maxiangle_list, max_angle_list):
                for noi, noise_level in zip(noi_list, noise_level_list):
                        print(name_val)
                        print(num_angles)
                        print(maxiangle)
                        print(max_angle)
                        print(noi)
                        print(noise_level)
                        ## Create the uniformly distributed directions
                        angle_partition = odl.uniform_partition(min_angle, max_angle, num_angles,
                                                            nodes_on_bdry=[(True, True)])

                        ## Create 2-D projection domain
                        ## The length should be 1.5 times of that of the reconstruction space
                        detector_partition = odl.uniform_partition(-24, 24, int(round(space.shape[0]*np.sqrt(2))))

                        ## Create 2-D parallel projection geometry
                        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

                        ## Ray transform aka forward projection. We use ASTRA CUDA backend.
                        forward_op = odl.tomo.RayTransform(space, geometry, impl='astra_cpu')

                        name_ground_truth = path_save_data + name_val
                        name = name_ground_truth + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_'
                        name += maxiangle + '_noise_' + noi

                        noise = noise_level * odl.phantom.noise.white_noise(forward_op.range)

                        data = forward_op(ground_truth) + noise

                        #data.show()

                        np.savetxt(name, data)


#


