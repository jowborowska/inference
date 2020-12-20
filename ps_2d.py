import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import corner
import h5py
import sys

import tools_ps as tools
import map_cosmo
import power_spectrum

'''
try:
    mappath = sys.argv[1]
except IndexError:
    print('Missing filename!')
    print('Usage: python ps_2d.py mappath')
    sys.exit(1)
'''
single_sigma_smooth = []
single_ps_smooth = []
map_paths = ['smoothed_map.h5','smoothed_map2.h5','smoothed_map3.h5','smoothed_map4.h5','smoothed_map5.h5','smoothed_map6.h5','smoothed_map7.h5','smoothed_map8.h5','smoothed_map9.h5','smoothed_map10.h5']
for mappath in map_paths:
   my_map = map_cosmo.MapCosmo(mappath)
   my_ps = power_spectrum.PowerSpectrum(my_map)
   ps_2d, k, nmodes = my_ps.calculate_ps(do_2d=True)
   rms_ps_mean, rms_ps_std = my_ps.run_noise_sims(10)
   single_ps_smooth.append(ps_2d)
   single_sigma_smooth.append(rms_ps_std)
   my_ps.make_h5()

k_smooth = k
ps_smooth = ps_2d
single_sigma_notsmooth = []
single_ps_notsmooth = []

map_paths_notsmooth = ['notsmoothed_map.h5','notsmoothed_map2.h5','notsmoothed_map3.h5','notsmoothed_map4.h5','notsmoothed_map5.h5','notsmoothed_map6.h5','notsmoothed_map7.h5','notsmoothed_map8.h5','notsmoothed_map9.h5','notsmoothed_map10.h5']
for mappath in map_paths_notsmooth:
   my_map = map_cosmo.MapCosmo(mappath)
   my_ps = power_spectrum.PowerSpectrum(my_map)
   ps_2d, k, nmodes = my_ps.calculate_ps(do_2d=True)
   rms_ps_mean, rms_ps_std = my_ps.run_noise_sims(10)
   single_ps_notsmooth.append(ps_2d)
   single_sigma_notsmooth.append(rms_ps_std)
   my_ps.make_h5()

k_notsmooth = k 
ps_notsmooth = ps_2d
def coadd_ps(ps_list, sigma_list):
   N = len(ps_list)
   ps_mean = 0
   w_sum = 0
   for i in range(N):
      w = 1./ sigma_list[i]**2.
      w_sum += w
      ps_mean += w*ps_list[i]
   ps_mean = ps_mean/w_sum
   ps_error = w_sum**(-0.5)
   return np.array(ps_mean), np.array(ps_error)

def coadd_without_weights(ps_list, sigma_list):
   N = len(ps_list)
   ps_mean = 0
   for i in range(N):
      ps_mean += ps_list[i]
   ps_mean = ps_mean/N
   return ps_mean

ps_smooth, error_smooth = coadd_ps(single_ps_smooth,single_sigma_smooth)
ps_notsmooth, error_notsmooth = coadd_ps(single_ps_notsmooth,single_sigma_notsmooth)
smooth_mean = coadd_without_weights(single_ps_smooth,single_sigma_smooth)
notsmooth_mean = coadd_without_weights(single_ps_notsmooth,single_sigma_notsmooth)

np.save('ps_smooth_single.npy', ps_smooth)
np.save('ps_notsmooth_single.npy', ps_notsmooth)
np.save('smooth_mean.npy', smooth_mean)
np.save('notsmooth_mean.npy', notsmooth_mean)
np.save('ps_2d_smooth.npy', ps_smooth)      
np.save('ps_2d_notsmooth.npy', ps_notsmooth)   
np.save('k_smooth.npy', k_smooth)
np.save('k_notsmooth.npy', k_notsmooth)

