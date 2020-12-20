import numpy as np

n_walkers = 120
n_steps = 8000
n_patches = 3

# number of independent simulations of the model for each
# step in the MCMC
n_realizations = 10

#likelihood = 'chi_squared'
likelihood = 'chi_squared_cov'
cov_output_folder = 'cov_output'
cov_id = '37'#'2'

#observables = ('ps',)
observables = ('ps', 'vid')
#observables =('vid',)
extra_observables = ('lum',)
# mcmc_model = 'wn_ps'
# mcmc_model = 'pl_ps'
# mcmc_model = 'Lco_Pullen'
# mcmc_model = 'Lco_Li'
#mcmc_model = 'simp_Li'
#mcmc_model = 'univ'
#mcmc_model = 'power'
#mcmc_model = 'power_cov'
mcmc_model = 'power_cov_3par'
#mcmc_model = 'simp_power_cov'

prior_params = dict()

# Gaussian prior for white noise power spectrum
prior_params['wn_ps'] = [
    [5.0, 3.0]  # sigma_T
]

# Gaussian prior for power law power spectrum
# [mean, stddev]
prior_params['pl_ps'] = [
    [7.0, 3.0],  # A
    [2.5, 1.7]  # alpha
]

# Gaussian prior for linear L_CO model
prior_params['Lco_Pullen'] = [
    [-6.0, 1.]  # A
]

prior_params['Lco_Li'] = [
    [0.0, 0.3], 	# logdelta_MF
    [1.17, 0.37],	 # alpha - log10 slope
    [0.21, 3.74],	 # beta - log10 intercept
    [0.3, 0.1],		# sigma_SFR
    [0.3, 0.1],		# sigma_Lco
]

prior_params['simp_Li'] = [
    [1.17, 0.37],	 # alpha - log10 slope
    [0.21, 3.74],	 # beta - log10 intercept
    [0.5, 0.3],	     # sigma_tot
]

prior_params['univ'] = [ 
    [-1.66, 2.33], # A
    [0.04, 1.26],   # B
    [10.25, 5.29],  # logC
    [12.41, 1.77],  # logM
    [0.28, 0.07],   # D
    [-0.51, 1.17],  # logG
    [0.4, 0.1],     # sigma
]

var = np.array([0.1, 0.1, 0.1, 0.15, 0.02])
mu = np.array([-2.4, -0.5, 10.4, 12.4, 0.4])

# Real  ###### Oslo:
prior_params['power_cov'] = [ 
    [-3.614278,    0.45687856, 10.68714991, 12.47593157,  0.35517494],
     [[ 2.20772089, 0.08729876, 0.16230234, 0.19120324,-0.02930135],
      [ 0.08729876, 0.99506924, 0.19754548, 0.08097735, 0.02139264],
      [ 0.16230234, 0.19754548, 0.22875155, 0.16540717,-0.0210548 ],
      [ 0.19120324, 0.08097735, 0.16540717, 0.15713717,-0.02256431],
      [-0.02930135, 0.02139264,-0.0210548 ,-0.02256431, 0.02119048]]
]

# Real
prior_params['power_cov_3par'] = [ 
    [-3.614278,    0.45687856, 10.68714991],
     [[ 2.20772089, 0.08729876, 0.16230234],
      [ 0.08729876, 0.99506924, 0.19754548],
      [ 0.16230234, 0.19754548, 0.22875155]]
]


#### Pess
# prior_params['power_cov'] = [ 
#     [-9.83641138, 3.98303605, 10.80410692, 12.29006944, 0.37347993],
#     [[ 2.233868e+01,-3.935810e-01, 2.900729e-01, 6.838738e-01,-1.184618e-01],
#      [-3.935810e-01, 8.800735e+00, 5.035934e-01,-9.183879e-02, 5.628725e-03],
#      [ 2.900729e-01, 5.035934e-01, 1.996875e-01, 1.314272e-01,-3.145308e-02],
#      [ 6.838738e-01,-9.183879e-02, 1.314272e-01, 1.667866e-01,-3.704924e-02],
#      [-1.184618e-01, 5.628725e-03,-3.145308e-02,-3.704924e-02, 3.100206e-02]]
# ]

### Pess
prior_params['simp_power_cov'] = [ 
    [10.80410692,12.29006944, 0.37347993],
    [[1.996875e-01, 1.314272e-01,-3.145308e-02],
     [1.314272e-01, 1.667866e-01,-3.704924e-02],
     [-3.145308e-02,-3.704924e-02, 3.100206e-02]]
]

# prior_params['power_cov'] = [ 
#     [-2.4, -0.5, 10.4, 12.4, 0.4],
#     [[ 0.1       , 0.        , 0.01636731, 0.01922689,-0.0013141 ],
#     [ 0.        , 0.1       , 0.03029383, 0.01585133, 0.00925622],
#     [ 0.01636731, 0.03029383, 0.1       , 0.10685097,-0.01352427],
#     [ 0.01922689, 0.01585133, 0.10685097, 0.15      ,-0.0214177 ],
#     [-0.0013141 , 0.00925622,-0.01352427,-0.0214177 , 0.02      ]]
# ]

prior_params['power'] = [ 
    [-1.66, 2.33], # A
    [0.04, 1.26],   # B
    [10.25, 5.29],  # logC
    [12.41, 1.77],  # logM
    [0.4, 0.1],     # sigma
]

halo_catalogue_folder = '/mn/stornext/d16/cmbco/comap/haavard/im_inference/catalogues_2deg/' #'catalogues/'

min_mass = 2.5e10  # 1e12  # 2.5e10

model_params_true = dict()
model_params_true['wn_ps'] = [8.3]  # sigma_T for wn_ps
model_params_true['pl_ps'] = [8., 1.]  # A and alpha for pw_ps
model_params_true['Lco_Pullen'] = [-7.3]  # np.log10(1e6/5e11)]
model_params_true['Lco_Li'] = [0.0, 1.17, 0.21, 0.3, 0.3]  # [0.0, 1.37, -1.74, 0.3, 0.3]
model_params_true['simp_Li'] = [1.17, 0.21, 0.5]  # alpha, beta, sigma_tot
model_params_true['univ'] = [-1.66, 0.04, 10.25,
                             12.41, 0.28, -0.51,
                             0.4]
model_params_true['power'] = [-1.9, 0.04, 10.25,
                              12.0, 0.4]
# model_params_true['power_cov'] = [-1.9, 0.04, 10.25,
#                               12.0, 0.4]     # oslo model
# model_params_true['power_cov'] = [-3.7, 7.0, 11.1,
#                                   12.5, 0.36]     # pess
model_params_true['power_cov'] = [-2.75, 0.05, 10.61,
                                  12.3, 0.42]     # real
#model_params_true['power_cov'] = [-2.85, -0.42, 10.63,
#                                  12.3, 0.42]     # real+
# model_params_true['power_cov'] = [-2.4, -0.5, 10.45,
#                                   12.21, 0.4]     # opt
# model_params_true['simp_power_cov'] = [10.61, 12.3, 0.42]     # real
model_params_true['power_cov_3par'] = [-2.75, 0.05, 10.61]     # real

# Values for (A, B, log C, log (M/Msol), sigma):
# * pessimistic: (-3.7, 7.0, 11.1, 12.5, 0.36)
# * realistic: (-2.75, 0.05, 10.61, 12.3, 0.42)
# * realistic-plus: (-2.85, -0.42, 10.63, 12.3, 0.42)
# * optimistic: (-2.4, -0.5, 10.45, 12.21, 0.36))
                              #12.41, 0.4]
map_filename = 'trial4.npy' #'testing_output/blob/map_run266.npy' #'trial4.npy'
samples_filename = 'samples_lco_test4.npy'
run_log_file = 'run_log.txt'

output_dir = 'testing_output'
generate_file = True
save_file = False # True

# Use MPI-pool?
pool = False #True
n_threads = 128

# Cosmetics

labels = dict()

labels['simp_Li'] = [r'$\alpha$', r'$\beta$', r'$\sigma$']

labels['Lco_Pullen'] = [r'$\log A$']

labels['Lco_z'] = [r'$\log A$', r'$\sigma_z$']
labels['Lco_z_cov'] = [r'$\log A$', r'$\sigma_z$']

labels['power'] = [r'$A$', r'$B$', r'$\log C$', r'$\log M$', r'$\sigma$']

labels['power_cov'] = [r'$A$', r'$B$', r'$\log C$', r'$\log M$', r'$\sigma$']


labels['simp_power_cov'] = [r'$\log C$', r'$\log M$', r'$\sigma$']
