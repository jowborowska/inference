import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u


class MapObj:
    """
    Class for the map-objects which carry the info about the maps and
    also the maps themselves.
    """
    def __init__(self, exp_params):
        self.exp_params = exp_params
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.286, Ob0=0.047)  # FlatLambdaCDM(H0=cosmos.h*100*u.km/u.s/u.Mpc,
                            #   Om0=cosmos.Omega_M, Ob0=cosmos.Omega_B)
        #cosmo1=getattr(astropy.cosmology, exp_params.cosmology)
        #cosmo = cosmo1.clone(name= exp_params.cosmology+' mod', H0=cosmos.h*100*u.km/u.s/u.Mpc, Om0=cosmos.Omega_M, Ob0=cosmos.Omega_B)
        self.fov_x = float(exp_params.fov_x)
        self.fov_y = float(exp_params.fov_y)
        self.FWHM = float(exp_params.FWHM)
        self.nu_i = float(exp_params.nu_i)
        self.nu_f = float(exp_params.nu_f)
        self.nu_rest = float(exp_params.nu_rest)
        self.n_nu_bins = exp_params.n_nu_bins
        self.n_pix_x = exp_params.n_pix_x
        self.n_pix_y = exp_params.n_pix_y

        # redshift
        self.z_i = self.nu_rest/self.nu_i - 1
        self.z_f = self.nu_rest/self.nu_f - 1

        # instrumental beam
        # exp_params.sigma_x = self.FWHM/60. * np.pi/180. / np.sqrt(8 * np.log(2))  # Don't think this makes sense!!
        # exp_params.sigma_y = self.FWHM/60. * np.pi/180. / np.sqrt(8 * np.log(2))
        # self.sigma_x = exp_params.sigma_x
        # self.sigma_y = exp_params.sigma_y

        # map frequency dimension
        # negative steps as larger observed frequency means lower redshift
        self.dnu = (self.nu_i - self.nu_f)/(self.n_nu_bins)
        self.nu_binedges = np.arange(self.nu_i, self.nu_f-self.dnu, -self.dnu)
        self.nu_bincents = self.nu_binedges[:-1] - self.dnu/2
        self.z_array = self.nu_rest/self.nu_binedges - 1

        self.pix_size_x = self.fov_x/self.n_pix_x
        self.pix_size_y = self.fov_y/self.n_pix_y

        self.Ompix = (self.pix_size_x*np.pi/180.)*(self.pix_size_y*np.pi/180.)

        self.pix_binedges_x = np.arange(-self.fov_x/2.,
                                        self.fov_x/2. + self.pix_size_x,
                                        self.pix_size_x)
        self.pix_binedges_y = np.arange(-self.fov_y/2.,
                                        self.fov_y/2. + self.pix_size_y,
                                        self.pix_size_y)

        self.pix_bincents_x = 0.5*(self.pix_binedges_x[:-1]
                                   + self.pix_binedges_x[1:])
        self.pix_bincents_y = 0.5*(self.pix_binedges_y[:-1]
                                   + self.pix_binedges_y[1:])

        # comoving distances
        self.x = (self.pix_binedges_x*np.pi/180.
                  * np.mean(self.cosmo.comoving_transverse_distance(self.z_array).value))
        self.y = (self.pix_binedges_y*np.pi/180.
                  * np.mean(self.cosmo.comoving_transverse_distance(self.z_array).value))
        self.z = self.cosmo.comoving_distance(self.z_array).value

        self.n_x = len(self.x) - 1
        self.n_y = len(self.y) - 1
        self.n_z = len(self.z) - 1
        self.dx = np.abs(np.mean(np.diff(self.x)))
        self.dy = np.abs(np.mean(np.diff(self.y)))
        self.dz = np.abs(np.mean(np.diff(self.z)))
        self.voxel_volume = self.dx * self.dy * self.dz
        self.volume = ((self.x[-1] - self.x[0])
                       * (self.y[-1] - self.y[0])
                       * (self.z[-1] - self.z[0])
                       )
        self.n_vox = self.n_nu_bins * self.n_pix_x * self.n_pix_y
        # self.voxel_volume = self.volume / self.n_vox
        self.map = None

    def calculate_observables(self, Observables):
        for observable in Observables:
            observable.calculate_observable(self)

    def generate_noise_map(self):
        return self.exp_params.sigma_T * np.random.randn(
            self.n_x, self.n_y, self.n_z)
