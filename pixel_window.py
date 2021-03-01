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


#I want to take signal and x, y from the simulated signal map
signal_map_file = '/mn/stornext/d16/cmbco/comap/jowita/inference/notsmoothed_map.h5'
with h5py.File(signal_map_file, mode="r") as my_file2:      
   x = np.array(my_file2['x'][:])
   y = np.array(my_file2['y'][:])
   #signal_map = np.array(my_file2['map_beam'][:])

no_of_realizations = 1
mcmc_params = importlib.import_module('mc_cube')

exp_params = importlib.import_module('exp_cube')

#signal_maps = []
ps_low_arr = []
ps_high_arr = []
k_arr = []
for i in range(no_of_realizations):
   print (str(i+1) + '. realization out of ' + str(no_of_realizations))
   src.tools.make_picklable((exp_params, mcmc_params))
   mcmc_params.observables = ('ps', 'vid')
   model, observables, _, map_obj = src.tools.set_up_mcmc(mcmc_params, exp_params)
   model_params = [-2.75, 0.05, 10.61, 12.3, 0.42]   # realistic model
   map_obj.map, map_obj.lum_func = model.generate_map(model_params)
   my_map1 = map_obj.map
   my_map2 = map_obj.map
   sh = my_map1.shape
   #this part added to check the normalization
   muK2K = 1e-6 #micro Kelvins to Kelvins
   rms_map = 10.*muK2K + 10.*np.random.uniform(0.0, 1.*muK2K, sh)
   my_map1 = rms_map
   my_map2 = rms_map

   #end of this part
   my_map_low_res = my_map1.reshape(sh[0], sh[1], 4, 64, 16).mean(4) #high resolution frequency to low resolution freq. !this is what I need for pixel window
   my_map_low_res = my_map_low_res.transpose(2, 3, 0, 1) #{4, 64, 120, 120}
   '''
   #this was moved to map_cosmo
   plt.figure()
   plt.imshow(my_map_low_res[2, :, :, 0].T, interpolation='none')
   plt.savefig('maplow.png')
   '''
   my_map_high_res = my_map2.reshape(sh[0], sh[1], 4, 64, 16)
   my_map_high_res = my_map_high_res.transpose(2, 3, 4, 0, 1) #{4, 64, 16, 120, 120}
   '''
   #this was moved to map_cosmo2
   plt.figure()
   plt.imshow(my_map_high_res[2, :, 0,:, 0].T, interpolation='none')
   plt.savefig('maphigh.png')
   '''
   signal_map_low_res = my_map_low_res
   signal_map_high_res = my_map_high_res
   #signal_maps.append(signal_map)
   rms_low_res = np.zeros_like(my_map_low_res) + 1.0
   rms_high_res = np.zeros_like(my_map_high_res) + 1.0

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
   #rms_ps_mean, rms_ps_std = my_ps.run_noise_sims(weights=True, n_sims=50)
   #my_ps.make_h5()
   my_map_high_res = map_cosmo2.MapCosmo('sim_signal_high_res.h5')
   my_ps_high_res = power_spectrum.PowerSpectrum(my_map_high_res)
   ps_high, k, nmodes = my_ps_high_res.calculate_ps(do_2d=True, weights=True)
   ps_high_arr.append(ps_high)
   k_arr.append(k)

def plot_ps(ps_2d, titlename, titlee, pw=False):
   fig, ax = plt.subplots(1,1)
   if pw == True: 
      img = ax.imshow(ps_2d, interpolation='none', origin='lower', extent=[0,1,0,1])
   else:
      img = ax.imshow(np.log10(ps_2d), interpolation='none', origin='lower', extent=[0,1,0,1])
   #plt.imshow(np.log10(nmodes), interpolation='none', origin='lower')
   cbar = fig.colorbar(img)
   if pw==False:
      cbar.set_label(r'$\log_{10}(\tilde{P}_{\parallel, \bot}(k))$ [$\mu$K${}^2$ (Mpc)${}^3$]')
   if pw==True:
      cbar.set_label(r'$\tilde{P}_{\parallel, \bot}(k)$ [$\mu$K${}^2$ (Mpc)${}^3$]')

   def log2lin(x, k_edges):
       loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
       logx = np.log10(x) - np.log10(k_edges[0])
       return logx / loglen


   # ax.set_xscale('log')
   minorticks = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
              0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
              0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
              0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
              2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
              20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
              200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0]

   majorticks = [1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02]
   majorlabels = ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$']

   xbins = my_ps_low_res.k_bin_edges_par

   ticklist_x = log2lin(minorticks, xbins)
   majorlist_x = log2lin(majorticks, xbins)

   ybins = my_ps_low_res.k_bin_edges_perp

   ticklist_y = log2lin(minorticks, ybins)
   majorlist_y = log2lin(majorticks, ybins)


   ax.set_xticks(ticklist_x, minor=True)
   ax.set_xticks(majorlist_x, minor=False)
   ax.set_xticklabels(majorlabels, minor=False)
   ax.set_yticks(ticklist_y, minor=True)
   ax.set_yticks(majorlist_y, minor=False)
   ax.set_yticklabels(majorlabels, minor=False)

   plt.xlabel(r'$k_{\parallel}$')
   plt.ylabel(r'$k_{\bot}$')
   plt.xlim(0, 1)
   plt.ylim(0, 1)
   #plt.savefig('ps_par_vs_perp_nmodes.png')
   plt.title(titlee, fontsize=12)
   plt.savefig(titlename)
   #plt.show()


np.save('ps_low_res.npy',np.array(ps_low_arr))
np.save('ps_high_res.npy',np.array(ps_high_arr))
np.save('k_arr.npy',np.array(k_arr))

ps_low_arr= np.array(ps_low_arr)
ps_high_arr = np.array(ps_high_arr)
k_arr = np.array(k_arr)
pixel_window = np.zeros_like(ps_low_arr)

ps_low_arr= np.load('ps_low_res.npy')
ps_high_arr = np.load('ps_high_res.npy')
k_arr = np.load('k_arr.npy')
pixel_window = np.zeros_like(ps_low_arr)
'''
for i in range(no_of_realizations):
   pixel_window[i] = ps_low_arr[i]/ps_high_arr[i]
np.save('pixel_window.npy', pixel_window)

plt.figure()
pw = plt.imshow(pixel_window[0])
plt.colorbar(pw)
plt.savefig('pw_new2.png')

plt.figure()
psl1 = plt.imshow(ps_low_arr[0])
plt.colorbar(psl1)
plt.savefig('psl1_new.png')

plt.figure()
psh1 = plt.imshow(ps_high_arr[0])
plt.colorbar(psh1)
plt.savefig('psh1_new.png')
'''
plot_ps(ps_low_arr[0], 'pslow_wn.png', 'Low freq resolution, white noise')
plot_ps(ps_high_arr[0], 'pshigh_wn.png', 'High freq resolution, white noise')
plot_ps(ps_low_arr[0]/ps_high_arr[0], 'pwindow_wn.png', 'Pixel window, white noise', pw=True)
'''
pixel_window = np.mean(pixel_window, axis=0)
ps_low_arr_mean = np.mean(ps_low_arr, axis=0)
ps_high_arr_mean = np.mean(ps_high_arr, axis=0)

plt.figure()
psl1 = plt.imshow(ps_low_arr[1])
plt.colorbar(psl1)
plt.savefig('psl1.png')

plt.figure()
psh1 = plt.imshow(ps_high_arr[1])
plt.colorbar(psh1)
plt.savefig('psh1.png')

plt.figure()
pw = plt.imshow(pixel_window)
plt.colorbar(pw)
plt.savefig('pw.png')


plt.figure()
ps_low = plt.imshow(ps_low_arr_mean)
plt.colorbar(ps_low)
plt.savefig('ps_low.png')

plt.figure()
ps_high = plt.imshow(ps_high_arr_mean)
plt.colorbar(ps_high)
plt.savefig('ps_high.png')
'''
