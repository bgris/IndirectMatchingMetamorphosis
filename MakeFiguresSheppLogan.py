#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:58:42 2018

@author: barbara
"""


import odl
import numpy as np

##%% Create data from lddmm registration
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Image
#namepath = 'bgris'
namepath = 'barbara'



# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
    dtype='float32', interp='linear')

numtest = 9

## Data parameters
index_name_template = 0
index_name_ground_truth = 0

index_angle = 2
index_maxangle = 0
index_noise = 2



sigmalist = [0.3, 0.6, 1., 2., 3.0, 4., 5., 10.]
name_sigma_list=['3e_1','6e_1', '1', '2', '3', '4', '5', '10' ]

num_sigma = 4

## The parameter for kernel function
#sigma = 3.0
#name_sigma=str(int(sigma))
sigma = sigmalist[num_sigma]
name_sigma = name_sigma_list[num_sigma]




niter=200
epsV=0.02
epsZ=0.002
## Give regularization parameter
lamb = 1e-5
name_lamb='1e_' + str(-int(np.log(lamb)/np.log(10)))
tau = 1e-5
name_tau='1e_' + str(-int(np.log(tau)/np.log(10)))



explistlamb = [1, 3, 5, 7]
lamblist = [1e-1, 1e-3, 1e-5,1e-7]
name_lamb_list =['1e_' + str(ex) for ex in explistlamb]
explisttau = [1, 3, 5, 7]
taulist = [1e-1, 1e-3, 1e-5,1e-7]
name_tau_list =['1e_' + str(ex) for ex in explisttau]

numlam = 2
numtau = 2
lamb = lamblist[numlam]
name_lamb=name_lamb_list[numlam]
tau = taulist[numtau]
name_tau = name_tau_list[numtau]
#name_lamb = '1e-7'
#name_tau = '1e-7'


# Give the number of time points
time_itvs = 20
nb_time_point_int=time_itvs

typefig = '.pdf'


name_list_template = ['SheppLogan0']
name_list_ground_truth = ['SheppLogan8_deformed']
num_angles_list = [10, 50, 100]
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

name_exp = name_val + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_'
name_exp += maxiangle + '_noise_' + noi


#path_data = '/home/bgris/data/test' + str(numtest) + '/'
path_data = '/home/' + namepath + '/data/Metamorphosis/test' + str(8) + '/'
path_result_init = '/home/' + namepath + '/Results/Metamorphosis/test' + str(numtest) + '/'
path_result = path_result_init + name_exp + '__sigma_' + name_sigma + '__lamb_'
path_result += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'


path_figure = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/ReconstructionMetamorphosis/Paper/figures/'
name_figure = path_figure + 'test' + str(numtest) + name_exp + '__sigma_' + name_sigma + '__lamb_'
name_figure += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_'

#path_data = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/data/test' + str(numtest) + '/'

path_result_init = '/home/' + namepath + '/Results/Metamorphosis/test' + str(numtest) + '/'
#path_result_init = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/Results_ODE/Metamorphosis/test' + str(numtest) + '/'

path_result = path_result_init + name_exp + '__sigma_' + name_sigma + '__lamb_'
path_result += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'


path_figure = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/ReconstructionMetamorphosis/Paper/figures/'
name_figure = path_figure + 'test' + str(numtest) + name_exp + '__sigma_' + name_sigma + '__lamb_'
name_figure += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_'


##%%
mini=-0.7
maxi=1



name = 'Metamorphosis'
#name = 'Image'
#name = 'Template'
t = 20
image = space.element(np.loadtxt(path_result + name + '_t_' + str(t)))
fig = image.show(name, clim=[mini, maxi])

name = 'Metamorphosis'
name = 'Image'
#name = 'Template'
t = 20
image = space.element(np.loadtxt(path_result + name + '_t_' + str(t)))
fig = image.show(name, clim=[mini, maxi])

name = 'Metamorphosis'
name = 'Image'
name = 'Template'
t = 20
image = space.element(np.loadtxt(path_result + name + '_t_' + str(t)))
fig = image.show(name, clim=[mini, maxi])

#%%

name_list = ['Metamorphosis', 'Image', 'Template']
name_list_fig = ['Image', 'Template_part', 'Deformation_part']
for name, name_fig in zip(name_list, name_list_fig):
        #    for t in range(time_itvs + 1):
        t=time_itvs
        #image = np.rot90(space.element(np.loadtxt(path_result + name + '_t_' + str(t))))
        #plt.imshow(image, cmap=plt.get_cmap('bone'), vmin=mini, vmax=maxi)
        image = space.element(np.loadtxt(path_result + name + '_t_' + str(t)))
        fig = image.show(clim=[mini, maxi])
        plt.axis('off')
        fig.delaxes(fig.axes[1])
        plt.savefig(name_figure + name_fig  + str(t) + typefig, transparent = True, bbox_inches='tight',
        pad_inches = 0)
#

plt.close('all')


## Create forward operator
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

#rec_proj_data = forward_op(space.element(np.loadtxt(path_data + 'Metamorphosis' + '_t_' + str(time_itvs))))
data_load = forward_op.range.element(np.loadtxt(path_data + name_exp))
proj_template = forward_op(space.element(np.loadtxt(path_data + name_list_template[index_name_template] )))

indexdataplot = 10
plt.figure()
plt.plot(np.asarray(data_load)[indexdataplot], 'b', linewidth=0.5, label = 'Data')
#plt.plot(np.asarray(rec_proj_data)[indexdataplot], 'r', linewidth=0.5, label = 'Result')
plt.plot(np.asarray(proj_template)[indexdataplot], 'k', linewidth=0.5, label = 'Template data')
plt.axis([0, int(round(space.shape[0]*np.sqrt(2))), -4, 20]), plt.grid(True, linestyle='--')
plt.legend()
plt.savefig(name_figure + 'Data' + str(indexdataplot) + typefig, transparent = True, 
            bbox_inches= matplotlib.transforms.Bbox([[0,-0.5], [5.5, 5]]))


# figures for template
if False:
    image = space.element(space.element(np.loadtxt(path_data + name_list_template[index_name_template] )))
    #plt.imshow(image, cmap=plt.get_cmap('bone'), vmin=mini, vmax=maxi)
    fig = image.show(clim=[mini, maxi])
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    plt.savefig(path_figure + 'test' + str(numtest) + 'Template_withoutaxis' + typefig, transparent = True, bbox_inches='tight',
    pad_inches = 0)


# figures for ground truth
if False:
    numtruth = 0
    image = space.element(np.loadtxt(path_data + name_list_ground_truth[numtruth] ))
    fig = image.show(clim=[mini, maxi])
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    #plt.imshow(image, cmap=plt.get_cmap('bone'), vmin=mini, vmax=maxi)
    plt.savefig( path_figure + 'test' + str(numtest) + name_list_ground_truth[numtruth] + typefig, transparent = True, bbox_inches='tight',
    pad_inches = 0)





# figure for sinogram
if False:
    data_load = forward_op.range.element(np.loadtxt(path_data + name_exp))
    fig = data_load.show()
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    plt.savefig( path_figure + 'test' + str(numtest) + name_list_ground_truth[numtruth] + 'sinogram' + typefig, transparent = True, bbox_inches='tight',
    pad_inches = 0)
   
 



#%%

def snr_fun(signal, noise, impl):
    """Compute the signal-to-noise ratio.
    Parameters
    ----------
    signal : `array-like`
        Noiseless data.
    noise : `array-like`
        Noise.
    impl : {'general', 'dB'}
        Implementation method.
        'general' means SNR = variance(signal) / variance(noise),
        'dB' means SNR = 10 * log10 (variance(signal) / variance(noise)).
    Returns
    -------
    snr : `float`
        Value of signal-to-noise ratio.
        If the power of noise is zero, then the return is 'inf',
        otherwise, the computed value.
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal) / signal.size
        ave2 = np.sum(noise) / noise.size
        s_power = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        n_power = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))
        if impl == 'general':
            return s_power / n_power
        elif impl == 'dB':
            return 10.0 * np.log10(s_power / n_power)
        else:
            raise ValueError('unknown `impl` {}'.format(impl))
    else:
        return float('inf')

#
        

name_exp = name_val + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_'
name_exp += maxiangle + '_noise_0'

data_no_noise = forward_op.range.element(np.loadtxt(path_data + name_exp))






print('SNR = {}'.format(snr_fun(data_no_noise, data_load - data_no_noise, 'dB')))       


#%%  Pour TV
##%%
mini=-1
maxi=1
#lam = 1
data_matching = 'exact'
data_matching_list = ['exact', 'inexact']
for data_matching in data_matching_list :
    image = np.rot90(space.element(np.loadtxt(path_result + '_TV_' + data_matching + 'num_angles_' + str(num_angles) + '__lam_' + str(lam))))
    plt.axis('off')
    plt.imshow(image, cmap=plt.get_cmap('bone'), vmin=mini, vmax=maxi)
    plt.savefig(path_figure + 'test' + str(numtest) + 'TV' + data_matching+ 'num_angles_' + str(num_angles)  + '__lam_' + str(lam) + typefig, transparent = True, bbox_inches='tight',
        pad_inches = 0)
#

#%%  Pour FBP
##%%
mini=-1
maxi=1
image = np.rot90(space.element(np.loadtxt(path_result + '_FBP_' + 'num_angles_' + str(num_angles))))
plt.imshow(image, cmap=plt.get_cmap('bone'), vmin=mini, vmax=maxi)
plt.axis('off')
plt.savefig(path_figure + 'test' + str(numtest) + '_FBP_' +'num_angles_' + str(num_angles) + typefig, transparent = True, bbox_inches='tight',
pad_inches = 0)
#%% for robustness, figure by figure, varying sigma
#space = rec_space
namepath = 'barbara'
sigmalist = [0.3, 0.6, 1., 2., 3.0, 5., 10.]
name_sigma_list=['3e_1','6e_1', '1', '2', '3', '5', '10' ]


for sigma, name_sigma in zip(sigmalist, name_sigma_list):
    name_exp = name_val + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_'
    name_exp += maxiangle + '_noise_' + noi
            
    path_data = '/home/' + namepath + '/data/Metamorphosis/test8/'
    path_result_init = '/home/' + namepath + '/Results/Metamorphosis/test9/'
    #path_result_init = '/home/bgris/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/Results/test2/'
    path_result = path_result_init + name_exp + '__sigma_' + name_sigma + '__lamb_'
    path_result += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'
    path_drop = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/ReconstructionMetamorphosis/Paper/figures/'
    i = nb_time_point_int
    
    name_list = ['Metamorphosis', 'Image', 'Template']
    for name_im in name_list:
        image = space.element(space.element(np.loadtxt(path_result + name_im + '_t_' + str(i) )))
        #plt.imshow(image, cmap=plt.get_cmap('bone'), vmin=mini, vmax=maxi)
        fig = image.show(clim=[mini, maxi])
        plt.axis('off')
        fig.delaxes(fig.axes[1])
        plt.savefig(path_drop +  name_exp + 'various__sigma_' + name_sigma  + name_im + '_t_' + str(i) + 'plot' + typefig, transparent = True, bbox_inches='tight',
                    pad_inches = 0)

# 


  
#%% Example figures audition
niter = 200
mini=-1
maxi=1
path_result_init = '/home/' + namepath + '/Results/Metamorphosis/test9/'
name_list = ['Metamorphosis', 'Image', 'Template']
name_list = ['Image', 'Template']
titre_list = ['Composante template', 'Composante déformation' ]
#name_list = ['Metamorphosis']
#list_time = [0, 4, 6, 10, 14, 20]
list_time = [20]
for name, titre in zip(name_list, titre_list):
    for t in list_time:
        path_result = path_result_init + name_exp + '__sigma_' + name_sigma + '__lamb_'
        path_result += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'

        #image = np.rot90(space.element(np.loadtxt(path_result + name + '_t_' + str(t))))
        #plt.imshow(image, cmap=plt.get_cmap('bone'), vmin=mini, vmax=maxi)
        image = space.element(np.loadtxt(path_result + name + '_t_' + str(t)))
        fig = image.show(clim=[mini, maxi])
        plt.axis('off')
        #plt.title('t = ' + str(t / time_itvs), size=20)
        plt.title(titre, size=20)
        fig.delaxes(fig.axes[1])
        path_drop = '/home/' + namepath + '/Dropbox/Recherche/CNRS/Audition/figures/ex_meta/'
        plt.savefig(path_drop + 'SheppLogan' + name  + str(t) + typefig, transparent = True, bbox_inches='tight',
        pad_inches = 0, format='pdf')
#

plt.close('all')

# Pour ground truth
if False:
    numtruth = 0
    image = space.element(np.loadtxt(path_data + name_list_ground_truth[numtruth] ))
    fig = image.show(clim=[mini, maxi])
    plt.axis('off')
    plt.title('Image vraie', size=20)
    fig.delaxes(fig.axes[1])
    path_drop = '/home/' + namepath + '/Dropbox/Recherche/CNRS/Audition/figures/ex_meta/'
    plt.savefig(path_drop + 'ground_truth' + typefig, transparent = True, bbox_inches='tight',
    pad_inches = 0, format='pdf')


  
# figure for sinogram
if False:
    data_load = forward_op.range.element(np.loadtxt(path_data + name_exp))
    fig = data_load.show()
    plt.axis('off')
    plt.title('Sinogramme', size=20)
    fig.delaxes(fig.axes[1])
    path_drop = '/home/' + namepath + '/Dropbox/Recherche/CNRS/Audition/figures/ex_meta/'
    plt.savefig(path_drop + 'sinogram' + typefig, transparent = True, bbox_inches='tight',
    pad_inches = 0)
   
 
