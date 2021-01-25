import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import corner
import h5py
import sys

import tools_ps as tools
import map_cosmo
import power_spectrum



#first I want to read a real rms map
real_map = '/mn/stornext/d16/cmbco/comap/protodir/maps/co2_map_signal.h5'
with h5py.File(real_map, mode="r") as my_file:
   real_rms = np.array(my_file['rms_coadd'][:]) #Dataset {4, 64, 120, 120}

#I want to take signal and x, y from the simulated signal map
signal_map = '/mn/stornext/d16/cmbco/comap/jowita/inference/notsmoothed_map.h5'
with h5py.File(signal_map, mode="r") as my_file2:      
   x = np.array(my_file2['x'][:])
   y = np.array(my_file2['y'][:])
   signal_map = np.array(my_file2['map_beam'][:])

print (real_rms.hape, x.shape, y.shape, signal_map.shape)
