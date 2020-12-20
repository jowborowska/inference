import numpy as np
import h5py
import tools_ps as tools
import map_cosmo


class PowerSpectrum():
    def __init__(self, my_map):
        self.map = my_map
        self.weights_are_normalized = False

    def normalize_weights(self):
        self.map.w = self.map.w / np.sqrt(np.mean(self.map.w.flatten() ** 2))
        self.weights_are_normalized = True
    
    def calculate_ps(self, do_2d=False):
        n_k = 15

        if not self.weights_are_normalized: self.normalize_weights()
        if do_2d: #to get 2d PS surface
            self.k_bin_edges_par = np.logspace(-2.0, np.log10(1.0), n_k)
            self.k_bin_edges_perp = np.logspace(-2.0 + np.log10(2), np.log10(1.5), n_k)
            
            self.ps_2d, self.k, self.nmodes = tools.compute_power_spec_perp_vs_par(
                self.map.map * self.map.w, (self.k_bin_edges_perp, self.k_bin_edges_par),
                dx=self.map.dx, dy=self.map.dy, dz=self.map.dz
            )
            return self.ps_2d, self.k, self.nmodes
        else:
            self.k_bin_edges = np.logspace(-2.0, np.log10(1.5), n_k)
            self.ps, self.k, self.nmodes = tools.compute_power_spec3d(
                self.map.map * self.map.w, self.k_bin_edges,
                dx=self.map.dx, dy=self.map.dy, dz=self.map.dz
            )
            return self.ps, self.k, self.nmodes
    
    def run_noise_sims(self, n_sims): #only white noise here
        if not self.weights_are_normalized: self.normalize_weights()
        
        rms_ps = np.zeros((len(self.k_bin_edges) - 1, n_sims))
        for i in range(n_sims):
            randmap = self.map.rms * np.random.randn(*self.map.rms.shape)

            rms_ps[:, i] = tools.compute_power_spec3d(
                randmap * self.map.w, self.k_bin_edges,
                dx=self.map.dx, dy=self.map.dy, dz=self.map.dz
                )[0]
        self.rms_ps_mean = np.mean(rms_ps, axis=1)
        self.rms_ps_std = np.std(rms_ps, axis=1)
        return self.rms_ps_mean, self.rms_ps_std
    
    def make_h5(self, outname=None):
        if outname is None:
            #folder = '/mn/stornext/d16/cmbco/comap/protodir/spectra/'
            folder = 'spectra'         # have changed it to be on my computer
            tools.ensure_dir_exists(folder)
            outname = folder + 'ps' + self.map.save_string + '.h5'            

        f1 = h5py.File(outname, 'w')
        try:
            f1.create_dataset('mappath', data=self.map.mappath)
            f1.create_dataset('ps', data=self.ps)
            f1.create_dataset('k', data=self.k)
            #f1.create_dataset('k_bin_edges', data=self.k_bin_edges)
            f1.create_dataset('nmodes', data=self.nmodes)
        except:
            print('No power spectrum calculated.')
            return 
        try:
            f1.create_dataset('ps_2d', data=self.ps_2d)
            f1.create_dataset('k_bin_edges_perp', data=self.k_bin_edges_perp)
            f1.create_dataset('k_bin_edges_par', data=self.k_bin_edges_par)
        except:
            pass
        
        try:
            f1.create_dataset('rms_ps_mean', data=self.rms_ps_mean)
            f1.create_dataset('rms_ps_std', data=self.rms_ps_std)
        except:
            pass
        f1.close()


