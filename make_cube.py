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



mcmc_params = importlib.import_module('mc_cube')

exp_params = importlib.import_module('exp_cube')

src.tools.make_picklable((exp_params, mcmc_params))
mcmc_params.observables = ('ps', 'vid')
model, observables, _, map_obj = src.tools.set_up_mcmc(
    mcmc_params, exp_params)

# ps_kbins = np.logspace(-1.5, 0.0, 21)

# ps_kbins = np.logspace(-1.5, 0.3, 21)

model_params = [-2.75, 0.05, 10.61, 12.3, 0.42]   # realistic model

smoothed = sys.argv[2]
if smoothed == 'no':
   print ('unmoothed map created')
   map_obj.map, map_obj.lum_func = model.generate_map(model_params)

if smoothed == 'yes':
   print ('smoothed map created')
   map_obj.map, map_obj.lum_func = src.tools.create_smoothed_map(model, model_params) #<---- TRY 3D


#map_obj.map, map_obj.lum_func = src.tools.create_smoothed_map_3d(
#    model, model_params
#)

# ps, k, _ = src.tools.calculate_power_spec_3d(map_obj, ps_kbins)

# np.save('k_cube', k)
# np.save('ps_cube', ps)

my_map = map_obj.map

plt.imshow(my_map[:,0,:], interpolation='none')


np.save('cube_real', map_obj.map)

sh = my_map.shape
my_map = my_map.reshape(sh[0], sh[1], 4, 64, 16).mean(4)
my_map = my_map.transpose(2, 3, 0, 1)
plt.figure()
plt.imshow(my_map[2, :, :, 0].T, interpolation='none')
plt.show()
rms = np.zeros_like(my_map) + 1.0

sh = my_map.shape
#my_map = np.random.randn(*sh)

outname = sys.argv[1] + '.h5'
f2 = h5py.File(outname, 'w')

f2.create_dataset('x', data=map_obj.pix_bincents_x)
f2.create_dataset('y', data=map_obj.pix_bincents_y)
f2.create_dataset('map_beam', data=my_map)
f2.create_dataset('rms_beam', data=rms)
f2.close()
