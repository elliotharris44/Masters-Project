import numpy as np
import sxs
import qnm
import matplotlib.pyplot as plt
import scipy
from functions import *
import tqdm

class SXSAnalysis:
    """
    
    """

    def __init__(self, id="SXS:BBH:0305"):
        self.sim = sxs.load(id)

        a_vec = np.array(self.sim.metadata["remnant_dimensionless_spin"])
        self.a = np.linalg.norm(a_vec)

        self.strain = self.sim.strain
        self.time = self.strain.time

        ind = np.flatnonzero((self.sim.psi4.LM == [2,2]).all(axis=1))[0]
        psi4_22 = self.sim.psi4.data[:, ind].real
        peaks, _ = scipy.signal.find_peaks(psi4_22) #maybe need different peaks for real and imag
        main_peak_index = np.argmax(psi4_22[peaks])
        self.t = self.time[peaks[main_peak_index]]+20 #reduces mismatch the most
        print(self.t)

    def graph(self, waveform='h', mode=[2,2], n_overtones=0,
               shift_below=10, shift_above=75, centre=0, fit_start=0, a=None, mass_bh=None):
        """
        Arguments
        waveform(string): h, psi4
        mode(list): list of 2 element lists saying l and m mode
        """
        fig, axs = plt.subplots(2, 1)

        time = self.time
        if mass_bh is None:
            mass_bh = self.sim.metadata['remnant_mass']
  
        if waveform=='h':
            modes = self.strain
            
        else:
            modes = self.sim.psi4

        ind = np.flatnonzero((modes.LM == mode).all(axis=1))[0]
        signal = modes.data[:, ind]

        t = self.t+centre
        max_t = t+shift_above #change to time[-1]
        min = np.abs(time-t).argmin()
        max = np.abs(time-max_t).argmin()
        time_ring = time[min:max]
        time_shift = time_ring - time_ring[0] #put all in init

        dt = time_shift[1]-time_shift[0]
        time_fit = time_shift[int(fit_start/dt):]
        time_plot = time[min-int(shift_below/dt):max] - time_ring[0]

        signal_ring = signal[min+int(fit_start/dt):max]
        signal_plot = signal[min-int(shift_below/dt):max]
        signal_comb = np.concatenate([signal_ring.real, signal_ring.imag])
        time_comb = np.concatenate([time_fit, time_fit+time_fit[-1]+dt])

        p0 = []
        omegas = []
        taus = []
        for n in range(n_overtones+1):
            grav_220 = qnm.modes_cache(s=-2,l=mode[0],m=mode[1],n=n)
            if a is None:
                omega, _, _ = grav_220(self.a)
            else:
                omega, _, _ = grav_220(a)
            omega = omega/mass_bh
            omegas.append(np.real(omega))
            taus.append(-1/np.imag(omega))
            p0 += [0.1, 0] #parameter guesses

        popt, pcov = scipy.optimize.curve_fit(ringdown_comb(omegas, taus), time_comb, signal_comb, p0)
        
        y_fit_real = ringdown_real(omegas, taus)(time_plot, *popt)
        y_fit_imag = ringdown_imag(omegas, taus)(time_plot, *popt)

        for n in range(n_overtones+1):
            A = popt[2*n]
            phi = popt[2*n + 1]
            #print(f"Mode {mode}: Overtone {n}: "f"A = {A:.4e}, "f"phi = {phi:.3f}, "f"omega = {omegas[n]:.4f}, "f"tau = {taus[n]:.4f}")
        
        axs[0].plot(time_plot, y_fit_real, label="Fit")
        axs[1].plot(time_plot, y_fit_imag, label="Fit")
        axs[0].axvline(0, color='black', linestyle=':', label='Start of Ringdown')
        axs[1].axvline(0, color='black', linestyle=':', label='Start of Ringdown')
        axs[0].axvline(fit_start, color='grey', linestyle=':', label='Start of Fitting')
        axs[1].axvline(fit_start, color='grey', linestyle=':', label='Start of Fitting')
        
        axs[0].plot(time_plot, signal_plot.real, label=f"Data: Mode {mode}")
        axs[1].plot(time_plot, signal_plot.imag, label=f"Data: Mode {mode}")
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')
        axs[0].grid()
        axs[1].grid()
        plt.show()
        plt.plot(time_comb, signal_comb)
        plt.show()

        self.time_plot = time_plot
        self.h_data = signal_plot
        self.h_fit = y_fit_real + 1j*y_fit_imag
        if self.total_signal is None:
            self.total_signal = self.h_data.copy()
            self.total_fit = self.h_fit.copy()
        else:
            self.total_signal += self.h_data
            self.total_fit += self.h_fit

    def graphs(self, waveform='h', modes=[[2,2]], n_overtones=0, shift_below=10, shift_above=75,
               centre=0, fit_start=0, a=None, mass_bh=None):
        self.total_signal = None
        self.total_fit = None

        for m in modes:
            self.graph(waveform, m, n_overtones, shift_below, shift_above, centre, fit_start, a, mass_bh)

        plt.semilogy(self.time_plot, np.abs(self.total_signal), label='Data')
        plt.semilogy(self.time_plot, np.abs(self.total_fit), label='Fit')
        plt.grid()
        plt.legend()
        plt.show()
    
    def mismatch(self, modes=[[2,2]], n_overtones=0, shift_above=50, centre=0,
                  fit_start=0, a=None, mass_bh=None):
        self.graphs(modes=modes, n_overtones=n_overtones, shift_above=shift_above,
                    centre=centre, fit_start=fit_start, a=a, mass_bh=mass_bh)
        self.mm = mismatch_function(self.time_plot, self.total_signal, self.total_fit)
        plt.close('all')
    
    def mismatch_test(self):
        a_min = 0
        mm_min = 1
        for i in np.arange(0.6,0.8,0.01):
            self.mismatch(a=i)
            if self.mm<mm_min:
                a_min=i
                mm_min = self.mm.copy()
        print(f"The mismatch {mm_min} is a minimum when a is {a_min}")
    
    def colour_plot(self):
        spin_axis = np.arange(0.69,0.70,0.001) #x-axis
        mass_axis = np.arange(0.945,0.965,0.001) #y-axis
        mismatch_axis = np.zeros((len(mass_axis), len(spin_axis))) #'heat'

        for i,spin in enumerate(tqdm.tqdm(spin_axis)):
            for j,mass in enumerate(mass_axis):
                self.mismatch(a=spin, mass_bh=mass)
                mismatch_axis[j,i] = self.mm.copy()
        
        fig, ax = plt.subplots()
        im = ax.imshow(mismatch_axis, origin='lower', aspect='auto', extent=[spin_axis.min(), spin_axis.max(),
            mass_axis.min(), mass_axis.max()], vmax=0.0005)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Dimensionless spin contant")
        ax.set_ylabel("Black hole mass")
        plt.show()

        min_idx = np.unravel_index(np.argmin(mismatch_axis), mismatch_axis.shape)
        best_mass = mass_axis[min_idx[0]]
        best_spin = spin_axis[min_idx[1]]
        print(f"Minimum mismatch at mass={best_mass}, spin={best_spin}")


test = SXSAnalysis()
#test.graphs("h", modes=[[2,2], [3,3]], n_overtones=0, shift_below=10) 
#test.mismatch()
#test.mismatch_test()
#test.colour_plot()




#Checklist
#check effect of shift above
#reduce length to 50 as reduces error
#make more efficient