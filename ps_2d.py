import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import corner
import h5py
import sys

import tools_ps as tools
import map_cosmo
import power_spectrum


try:
    mappath = sys.argv[1]
except IndexError:
    print('Missing filename!')
    print('Usage: python ps_2d.py mappath')
    sys.exit(1)

my_map = map_cosmo.MapCosmo(mappath)

my_ps = power_spectrum.PowerSpectrum(my_map)

ps_2d, k, nmodes = my_ps.calculate_ps(do_2d=True)

my_ps.make_h5()


