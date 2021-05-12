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

no_of_realizations = 10
mcmc_params = importlib.import_module('mc_cube')

exp_params = importlib.import_module('exp_cube')

#signal_maps = []
ps_low_arr = []

k_arr = []
ps_low_arr_1D = []

k_arr_1D = []

#looks similar to low res in pixel window, but now we set sigma_angular to be small, to have the smoothing only in z direction (in experiment_params FWHM=1 instead of 4)
for i in range(no_of_realizations):
   print (str(i+1) + '. realization out of ' + str(no_of_realizations))
   src.tools.make_picklable((exp_params, mcmc_params))
   mcmc_params.observables = ('ps', 'vid')
   model, observables, _, map_obj = src.tools.set_up_mcmc(mcmc_params, exp_params)
   model_params = [-2.75, 0.05, 10.61, 12.3, 0.42]   # realistic model
   #map_obj.map, map_obj.lum_func = model.generate_map(model_params)
   map_obj.map, map_obj.lum_func = src.tools.create_smoothed_map_3d(model, model_params) #will be smoothed out in all directions
   print ("Model map created.")
   my_map1 = map_obj.map
   sh = my_map1.shape
   

   my_map_low_res = my_map1.reshape(sh[0], sh[1], 4, 64, 16).mean(4) #high resolution frequency to low resolution freq. !this is what I need for pixel window
   my_map_low_res = my_map_low_res.transpose(2, 3, 0, 1) #{4, 64, 120, 120}


 
   signal_map_low_res = my_map_low_res
   
   #signal_maps.append(signal_map)
   #rms_low_res = np.zeros_like(my_map_low_res) + 1.0
   rms_low_res = real_rms
   #rms_high_res = np.zeros_like(my_map_high_res) + 1.0


   #Create a map file
   outname = 'sim_signal_low_res.h5'
   f = h5py.File(outname, 'w') 
   f.create_dataset('x', data=map_obj.pix_bincents_x) #<-----------i changed these from data=x
   f.create_dataset('y', data=map_obj.pix_bincents_y) #<-----------i changed these from data=y
   f.create_dataset('map_coadd', data=signal_map_low_res)
   f.create_dataset('rms_coadd', data=rms_low_res)
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

np.save('ps_theory_new_2D.npy',np.array(ps_low_arr))
np.save('k_arr_theory_2D.npy',np.array(k_arr))
np.save('ps_theory_new_1D.npy',np.array(ps_low_arr_1D))
np.save('k_arr_theory_1D.npy',np.array(k_arr_1D))

