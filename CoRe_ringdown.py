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

    def graph(self, waveform='h', column=2, min_t=None, max_t=None, n_overtones=0):
        """
        Arguments: 
        waveform(string): h for strain, 22 ect for psi4 l=2, m=2 mode
        column(int): should be number from 1 to 9, 1 is time then strain ect
        min_time(int): is the minimum time
        max_time(int): is the cutoff time
        """

        if waveform=='h':
            series = self.R01_data['rh_22'] #strain for l=2, m=2
            series_1000 = series['Rh_l2_m2_r01000.txt'][:] #at radius 1000Msun, 9 columns, time, Re(strain) ect

        else:
            try:
                s = int(waveform)
            except ValueError:
                raise ValueError("Must enter h or two digit integer")
            
            series = self.R01_data[f'rpsi4_{waveform}']
            series_1000 = series[f'Rpsi4_l{waveform[0]}_m{waveform[1]}_r01000.txt'][:]

        time = series_1000.T[0]
        signal = series_1000.T[column-1]

        if min_t is None:
            peaks, _ = scipy.signal.find_peaks(signal)
            main_peak_index = np.argmax(signal[peaks])
            t = time[peaks[main_peak_index+3]] #selects peak two after main
        else:
            t = min_t
        if max_t is None:
            max_t = t + 50
        min = np.abs(time-t).argmin()
        max = np.abs(time-max_t).argmin()
        time = time[min:max]
        time_shift = time - time[0]
        signal = signal[min:max]
        if min_t is None:
            p0 = []
            omegas = []
            taus = []
            for n in range(n_overtones+1):
                grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=n) #need to adapt for different modes
                omega, _, _ = grav_220(0.8) #how to get a?
                omegas.append(np.real(omega))
                taus.append(-1/np.imag(omega))
                p0 += [0.1, 0] #parameter guesses

            if column==2:
                popt, pcov = scipy.optimize.curve_fit(ringdown_real(omegas, taus), time_shift, signal, p0)
                y_fit = ringdown_real(omegas, taus)(time_shift, *popt)
            elif column==3:
                popt, pcov = scipy.optimize.curve_fit(ringdown_imag(omegas, taus), time_shift, signal, p0)
                y_fit = ringdown_imag(omegas,taus)(time_shift, *popt)
            
            for n in range(n_overtones+1):
                A = popt[2*n]
                phi = popt[2*n + 1]
                print(f"Overtone {n}: "f"A = {A:.4e}, "f"phi = {phi:.3f}, "f"omega = {omegas[n]:.4f}, "f"tau = {taus[n]:.4f}")

        plt.plot(time, y_fit, label="Fit")
        plt.plot(time, signal, label="Data")
        plt.legend()
        plt.grid()
        plt.show()
        #Column 1 is time then I think Re(strain),Im(strain), ., ., ., dE/dt,...

test = CoReAnalysis()
test.graph("h", column=2, n_overtones=2)
