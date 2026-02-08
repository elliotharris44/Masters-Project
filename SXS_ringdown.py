import numpy as np
import sxs
import qnm
import matplotlib.pyplot as plt
import scipy
from functions import *

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
        #plt.show()
        plt.plot(time_comb, signal_comb)
        #plt.show()

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

        plt.plot(self.time_plot, self.total_signal.real, label='Data')
        plt.plot(self.time_plot, self.total_fit.real, label='Fit')
        plt.grid()
        plt.legend()
        #plt.show()
    
    def mismatch(self, modes=[[2,2]], n_overtones=0, shift_above=75, centre=0, fit_start=0):
        mm_min = 1
        mass_min = 0
        for i in np.arange(0.9, 1 + 1e-12, 0.002):
            self.graphs(modes=modes, n_overtones=n_overtones, shift_below=0,
                        shift_above=shift_above, fit_start=fit_start, centre=centre, mass_bh=i)
            mm_r = mismatch_function(self.time_plot, self.total_signal.real, self.total_fit.real)
            mm_i = mismatch_function(self.time_plot, self.total_signal.imag, self.total_fit.imag)
            mm = mm_r + mm_i
            print(f"Mismatch is {mm} when mass is {i}")
            if mm<mm_min:
                mm_min = mm
                mass_min = i
        print(f"mass_bh is {mass_min} mismatch is {mm_min}")
    
    def mismatch_test(self):
        for start in np.arange(0,40,10):
            self.mismatch(fit_start=start)
        


test = SXSAnalysis()
#test.graphs("h", modes=[[2,2], [3,3], [4,4]], n_overtones=0, shift_below=10) 
test.mismatch()
#test.mismatch_test()




#Checklist
#dt attribute