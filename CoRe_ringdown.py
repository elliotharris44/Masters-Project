import numpy as np
import qnm
import h5py
import matplotlib.pyplot as plt
import scipy
from functions import *

class CoReAnalysis:
    """
    
    """
    def __init__(self):
        self.R01_data = h5py.File('BAM_0125/data_R01.h5', 'r')

        psi4_22 = self.R01_data['rpsi4_22']['Rpsi4_l2_m2_r00750.txt'][:]
        self.time = psi4_22.T[0]
        psi4_22 = psi4_22.T[1] #real part
        peaks, _ = scipy.signal.find_peaks(psi4_22) #maybe need different peaks for real and imag
        main_peak_index = np.argmax(psi4_22[peaks])
        self.t = self.time[peaks[main_peak_index]] + 35 #minimises mismatch

    def graph(self, waveform='h', n_overtones=0,
               shift_below=10, shift_above=75, centre=0, fit_start=0, a=None, mass_bh=None):
        """
        Arguments: 
        waveform(string): h for strain, 22 ect for psi4 l=2, m=2 mode
        column(int): should be number from 1 to 9, 1 is time then strain ect
        min_time(int): is the minimum time
        max_time(int): is the cutoff time
        """
        fig, axs = plt.subplots(2, 1)

        time = self.time
        if mass_bh is None:
            mass_bh = 2.966531 #make more general, should I be using rest mass?, adm?
        mass_total = 1.500237 + 1.500237

        if waveform=='h':
            series = self.R01_data['rh_22'] #strain for l=2, m=2
            series_1000 = series['Rh_l2_m2_r00750.txt'][:] #at radius 1000Msun, 9 columns, time, Re(strain) ect

        else:
            try:
                s = int(waveform)
            except ValueError:
                raise ValueError("Must enter h or two digit integer")
            
            series = self.R01_data[f'rpsi4_{waveform}']
            series_1000 = series[f'Rpsi4_l{waveform[0]}_m{waveform[1]}_r01000.txt'][:]

        signal = series_1000.T[1] + 1j*series_1000.T[2]

        t = self.t+centre
        max_t = t+shift_above #change to time[-1]?
        min = np.abs(time-t).argmin()
        max = np.abs(time-max_t).argmin()
        time_ring = time[min:max]
        time_shift = time_ring - time_ring[0] #put all in init

        dt = time_shift[1]-time_shift[0]
        time_fit = time_shift[int(fit_start/dt):]
        time_plot = time[min-int(shift_below/dt):max] - time_ring[0]

        signal_ring = signal[min+int(fit_start/dt):max]
        signal_plot = signal[min-int(shift_below/dt):max]
        signal_comb = np.concatenate([np.real(signal_ring), np.imag(signal_ring)])
        time_comb = np.concatenate([time_fit, time_fit+time_fit[-1]+dt])

        p0 = []
        omegas = []
        taus = []
        for n in range(n_overtones+1):
            if waveform=='h':
                grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=n)
            else:
                grav_220 = qnm.modes_cache(s=-2,l=waveform[0],m=waveform[1],n=n)
            if a is None:
                omega, _, _ = grav_220(0.78)
            else:
                omega, _, _ = grav_220(a)
            omega = omega*mass_total/mass_bh
            omegas.append(np.real(omega))
            taus.append(-1/np.imag(omega))
            p0 += [0.1, 0] #parameter guesses

        popt, pcov = scipy.optimize.curve_fit(ringdown_comb(omegas, taus), time_comb, signal_comb, p0)
        
        y_fit_real = ringdown_real(omegas, taus)(time_plot, *popt)
        y_fit_imag = ringdown_imag(omegas, taus)(time_plot, *popt)
        
        for n in range(n_overtones+1):
            A = popt[2*n]
            phi = popt[2*n + 1]
            #print(f"Overtone {n}: "f"A = {A:.4e}, "f"phi = {phi:.3f}, "f"omega = {omegas[n]:.4f}, "f"tau = {taus[n]:.4f}")

        axs[0].plot(time_plot, y_fit_real, label="Fit")
        axs[1].plot(time_plot, y_fit_imag, label="Fit")
        axs[0].axvline(0, color='black', linestyle=':', label='Start of Ringdown')
        axs[1].axvline(0, color='black', linestyle=':', label='Start of Ringdown')
        axs[0].axvline(fit_start, color='grey', linestyle=':', label='Start of Fitting')
        axs[1].axvline(fit_start, color='grey', linestyle=':', label='Start of Fitting')
        
        axs[0].plot(time_plot, np.real(signal_plot), label=f"Data")
        axs[1].plot(time_plot, np.imag(signal_plot), label=f"Data")
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')
        axs[0].grid()
        axs[1].grid()
        plt.show()

        self.time_plot = time_plot
        self.signal_plot = signal_plot
        self.signal_fit = y_fit_real + 1j*y_fit_imag

    def mismatch(self, n_overtones=0, shift_above=75, centre=0, fit_start=0):
        mm_min = 1
        a_min = 0
        for i in np.arange(0.6, 0.85 + 1e-12, 0.01):
            self.graph(n_overtones=n_overtones, shift_below=0,
                        shift_above=shift_above, fit_start=fit_start, centre=centre, a=i)
            mm_r = mismatch_function(self.time_plot, np.real(self.signal_plot), np.real(self.signal_fit))
            mm_i = mismatch_function(self.time_plot, np.imag(self.signal_plot), np.imag(self.signal_fit))
            mm = mm_r + mm_i
            print(f"Mismatch is {mm} when a is {i}")
            if mm<mm_min:
                mm_min = mm
                a_min = i
        print(f"a is {a_min} mismatch {mm_min}")

    def mismatch_test(self):
        for centre in np.arange(0,40,10):
            self.mismatch(centre=centre)

test = CoReAnalysis()
test.graph("h", n_overtones=0)
#test.mismatch()

#Checklist
#find best centre
#check scaling
