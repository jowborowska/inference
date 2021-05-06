import numpy.fft as fft
import tools_ps as tools
import map_cosmo
import map_cosmo2
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


no_of_realizations = 4
mcmc_params = importlib.import_module('mc_cube')

exp_params = importlib.import_module('exp_cube')

#signal_maps = []
ps_low_arr = [] #this will be smoothed
ps_high_arr = [] #this won't be smoothed
k_arr = []
ps_low_arr_1D = []
ps_high_arr_1D = []
k_arr_1D = []

for i in range(no_of_realizations):
   print (str(i+1) + '. realization out of ' + str(no_of_realizations))
   src.tools.make_picklable((exp_params, mcmc_params))
   mcmc_params.observables = ('ps', 'vid')
   model, observables, _, map_obj = src.tools.set_up_mcmc(mcmc_params, exp_params)
   model_params = [-2.75, 0.05, 10.61, 12.3, 0.42]   # realistic model
   #map_obj.map, map_obj.lum_func = model.generate_map(model_params)
   map_obj.map, map_obj.lum_func = src.tools.create_smoothed_map(model, model_params) #will be smoothed out in angular directions
   my_map1 = map_obj.map
   map_obj.map, map_obj.lum_func = model.generate_map(model_params) #not smoothed
   my_map2 = map_obj.map
   sh = my_map1.shape
   

   my_map_low_res = my_map1.reshape(sh[0], sh[1], 4, 64, 16).mean(4) 
   my_map_low_res = my_map_low_res.transpose(2, 3, 0, 1) #{4, 64, 120, 120}

   my_map_high_res = my_map2.reshape(sh[0], sh[1], 4, 64, 16).mean(4)
   my_map_high_res = my_map_high_res.transpose(2, 3, 0, 1) #{4, 64, 120, 120}
 
   signal_map_low_res = my_map_low_res
   signal_map_high_res = my_map_high_res
   #signal_maps.append(signal_map)
   rms_low_res = np.zeros_like(my_map_low_res) + 1.0 #Comment this one out to get pseudo spectra!!!
   #rms_low_res = real_rms
   rms_high_res = np.zeros_like(my_map_high_res) + 1.0
   #rms_high_res = real_rms
  

   #Create a map file
   outname = 'sim_signal_low_res.h5'
   f = h5py.File(outname, 'w') 
   f.create_dataset('x', data=map_obj.pix_bincents_x) #<-----------i changed these from data=x
   f.create_dataset('y', data=map_obj.pix_bincents_y) #<-----------i changed these from data=y
   f.create_dataset('map_coadd', data=signal_map_low_res)
   f.create_dataset('rms_coadd', data=rms_low_res)
   f.close()

   outname = 'sim_signal_high_res.h5'
   f = h5py.File(outname, 'w') 
   f.create_dataset('x', data=map_obj.pix_bincents_x) #<-----------i changed these from data=x
   f.create_dataset('y', data=map_obj.pix_bincents_y) #<-----------i changed these from data=y
   f.create_dataset('map_coadd', data=signal_map_high_res)
   f.create_dataset('rms_coadd', data=rms_high_res)
   f.close()

   #Calculate PS with weights
   my_map_low_res = map_cosmo.MapCosmo('sim_signal_low_res.h5')
   my_ps_low_res = power_spectrum.PowerSpectrum(my_map_low_res)
   ps_low, k, nmodes = my_ps_low_res.calculate_ps(do_2d=True, weights=True)
   ps_low_arr.append(ps_low)
   
   my_map_low_res_1D = map_cosmo.MapCosmo('sim_signal_low_res.h5')
   my_ps_low_res_1D = power_spectrum.PowerSpectrum(my_map_low_res_1D)
   ps_low_1D, k_1D, nmodes_1D = my_ps_low_res_1D.calculate_ps(do_2d=False, weights=True)
   ps_low_arr_1D.append(ps_low_1D)


   #rms_ps_mean, rms_ps_std = my_ps.run_noise_sims(weights=True, n_sims=50)
   #my_ps.make_h5()

   my_map_high_res = map_cosmo.MapCosmo('sim_signal_high_res.h5')
   my_ps_high_res = power_spectrum.PowerSpectrum(my_map_high_res)
   ps_high, k, nmodes = my_ps_high_res.calculate_ps(do_2d=True, weights=True)
   ps_high_arr.append(ps_high)
   k_arr.append(k)

   my_map_high_res_1D = map_cosmo.MapCosmo('sim_signal_high_res.h5')
   my_ps_high_res_1D = power_spectrum.PowerSpectrum(my_map_high_res_1D)
   ps_high_1D, k_1D, nmodes_1D = my_ps_high_res_1D.calculate_ps(do_2d=False, weights=True)
   ps_high_arr_1D.append(ps_high_1D)
   k_arr_1D.append(k_1D)



np.save('ps_smooth_newest.npy',np.array(ps_low_arr))
np.save('ps_original_newest.npy',np.array(ps_high_arr))
np.save('k_arr_newest.npy',np.array(k_arr))


np.save('ps_smooth_1D_newest.npy',np.array(ps_low_arr_1D))
np.save('ps_original_1D_newest.npy',np.array(ps_high_arr_1D))
np.save('k_arr_1D_newest.npy',np.array(k_arr_1D))

ps_low_arr= np.array(ps_low_arr)
ps_low_arr_1D = np.array(ps_low_arr_1D)

ps_low_arr= np.load('ps_smooth_newest.npy')
ps_high_arr = np.load('ps_original_newest.npy')
k_arr = np.load('k_arr_newest.npy')
beam_tf = np.zeros_like(ps_low_arr)

for i in range(no_of_realizations):
   beam_tf[i] = ps_low_arr[i]/ps_high_arr[i]
beam_tf = np.mean(beam_tf, axis=0)
np.save('beam_tf_newest.npy', beam_tf)

ps_low_arr_1D = np.load('ps_smooth_1D_newest.npy')
ps_high_arr = np.load('ps_original_1D_newest.npy')
k_arr = np.load('k_arr_1D_newest.npy')







