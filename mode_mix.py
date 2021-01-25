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
'''
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
   print (i+1, '. realization out of 10')
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

np.save('ps_weights.npy',np.array(ps_weights_arr))
np.save('ps_no_weights.npy',np.array(ps_noweights_arr))
np.save('k_arr.npy',np.array(k_arr))

ps_weights_arr = np.array(ps_weights_arr)
ps_noweights_arr = np.array(ps_noweights_arr)
k_arr = np.array(k_arr)
mode_mixing_tf = np.zeros_like(ps_weights_arr)
'''
ps_weights_arr = np.load('ps_weights.npy')
ps_noweights_arr = np.load('ps_no_weights.npy')
k_arr = np.load('k_arr.npy')
mode_mixing_tf = np.zeros_like(ps_weights_arr)

plt.figure()
for i in range(10):
   mode_mixing_tf[i] = ps_weights_arr[i]/ps_noweights_arr[i]
   plt.plot(k_arr[i], mode_mixing_tf[i])
plt.xscale('log')
#plt.yscale('log')
plt.ylim(0.5,3.5)
plt.xlim(0.04,0.7)
plt.ylabel(r'$\mathrm{P^{weights}(k)/P(k)}$', fontsize=16)
plt.xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=16)
labnums = [0.05,0.1, 0.2, 0.5]
plt.xticks(ticks=labnums)
plt.tight_layout()
plt.savefig('mode_mix.png')

print (k_arr.shape)
mode_mixing_tf = ps_weights_arr/ps_noweights_arr
print (mode_mixing_tf.shape)
our_estimate = np.mean(mode_mixing_tf, axis=0)
print (our_estimate.shape)
error_bars = np.std(mode_mixing_tf, axis=0)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.errorbar(k_arr[0],our_estimate, error_bars, color='black', fmt='o')
ax1.set_ylabel(r'$\mathrm{P^{weights}(k)/P(k)}$', fontsize=14)
ax1.set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
labnums = [0.05,0.1, 0.2, 0.5]
ax1.set_xlim(0.046,0.68)
ax1.set_xscale('log')
ax1.set_ylim(0.8,1.5)
ax1.grid()
ax1.plot(k_arr[0], 0 * np.zeros_like(k_arr[0]) + 1, alpha=0.4, zorder=1, color='black')
ax1.set_xticks(labnums)
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig('mode_mix2.pdf')


