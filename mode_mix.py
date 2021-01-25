import numpy.fft as fft
import tools_ps as tools
import map_cosmo
import power_spectrum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import scipy.optimize as opt
from scipy.stats import norm
import scipy
import src.MapObj
import src.tools
import src.Model
import src.Observable
from scipy import interpolate
# import mcmc_params
# import experiment_params as exp_params
import emcee
import corner
import datetime
from astropy import units as u
import os
import subprocess
import socket
import sys
import importlib
import h5py

#first I want to read a real rms map
real_map = '/mn/stornext/d16/cmbco/comap/protodir/maps/co2_map_signal.h5'
with h5py.File(real_map, mode="r") as my_file:
   real_rms = np.array(my_file['rms_coadd'][:]) #Dataset {4, 64, 120, 120}

#I want to take signal and x, y from the simulated signal map
signal_map = '/mn/stornext/d16/cmbco/comap/jowita/inference/notsmoothed_map.h5'
with h5py.File(signal_map, mode="r") as my_file2:      
   x = np.array(my_file2['x'][:])
   y = np.array(my_file2['y'][:])
   #signal_map = np.array(my_file2['map_beam'][:])


mcmc_params = importlib.import_module('mc_cube')

exp_params = importlib.import_module('exp_cube')

#signal_maps = []
ps_weights_arr = []
ps_noweights_arr = []
k_arr = []
for i in range(10):
   print (i, '. realization out of 10')
   src.tools.make_picklable((exp_params, mcmc_params))
   mcmc_params.observables = ('ps', 'vid')
   model, observables, _, map_obj = src.tools.set_up_mcmc(mcmc_params, exp_params)
   model_params = [-2.75, 0.05, 10.61, 12.3, 0.42]   # realistic model
   map_obj.map, map_obj.lum_func = model.generate_map(model_params)
   my_map = map_obj.map
   sh = my_map.shape
   my_map = my_map.reshape(sh[0], sh[1], 4, 64, 16).mean(4)
   my_map = my_map.transpose(2, 3, 0, 1)

   signal_map = my_map #<---------------this I would like to have different versions of
   #signal_maps.append(signal_map)

   #Create a map file with simulated signal map and real rms
   outname = 'sim_signal_real_rms.h5'
   f = h5py.File(outname, 'w') 
   f.create_dataset('x', data=x)
   f.create_dataset('y', data=y)
   f.create_dataset('map_coadd', data=signal_map)
   f.create_dataset('rms_coadd', data=real_rms)
   f.close()

   #Calculate PS with weights
   my_map = map_cosmo.MapCosmo('sim_signal_real_rms.h5')
   my_ps = power_spectrum.PowerSpectrum(my_map)
   ps_weights, k, nmodes = my_ps.calculate_ps(do_2d=False, weights=True)
   ps_weights_arr.append(ps_weights)
   #rms_ps_mean, rms_ps_std = my_ps.run_noise_sims(weights=True, n_sims=50)
   #my_ps.make_h5()
   my_map2 = map_cosmo.MapCosmo('sim_signal_real_rms.h5')
   my_ps2 = power_spectrum.PowerSpectrum(my_map2)
   ps_no_weights, k, nmodes = my_ps2.calculate_ps(do_2d=False, weights=False)
   ps_noweights_arr.append(ps_no_weights)
   k_arr.append(k)

mode_mixing_tf = ps_weights_arr/ps_noweights_arr
plt.figure()
for i in range(10):
   plt.plot(k_arr[i], mode_mixing_tf[i])
   plt.savefig('mode_mix.png')





