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
        self.total_signal = np.zeros_like(self.time)

    def graph(self, waveform='h', comp='real', mode=[2,2], min_t=None, max_t=None, n_overtones=0):
        """
        Arguments
        waveform(string): h, psi4
        comp(string): real or imag component
        mode(list): list of 2 element lists saying l and m mode
        """

        time = self.time

        if waveform=='h':
            modes = self.strain
            
        else:
            modes = self.sim.psi4

        matches = np.all(modes.LM == mode, axis=1)
        ind = np.where(matches)[0]
        signal = modes.data[:,ind].squeeze()
        
        if comp=="real":
            signal = signal.real
        else:
            signal = signal.imag

        if min_t is None:
            peaks, _ = scipy.signal.find_peaks(signal)
            main_peak_index = np.argmax(signal[peaks])
            t = time[peaks[main_peak_index+2]] #selects peak two after main, could cause problems for sum
        else:
            t = min_t
        if max_t is None:
            max_t = t + 75
        min = np.abs(time-t).argmin()
        max = np.abs(time-max_t).argmin()
        time_ring = time[min:max]
        time_shift = time_ring - time_ring[0]
        signal_ring = signal[min:max]

        if min_t is None:
            p0 = []
            omegas = []
            taus = []
            for n in range(n_overtones+1):
                grav_220 = qnm.modes_cache(s=-2,l=mode[0],m=mode[1],n=n)
                omega, _, _ = grav_220(self.a)
                omegas.append(np.real(omega))
                taus.append(-1/np.imag(omega))
                p0 += [0.1, 0] #parameter guesses

            if comp=='real':
                popt, pcov = scipy.optimize.curve_fit(ringdown_real(omegas, taus), time_shift, signal_ring, p0)
                y_fit = ringdown_real(omegas, taus)(time_shift, *popt)
            else:
                popt, pcov = scipy.optimize.curve_fit(ringdown_imag(omegas, taus), time_shift, signal_ring, p0)
                y_fit = ringdown_imag(omegas, taus)(time_shift, *popt)

            for n in range(n_overtones+1):
                A = popt[2*n]
                phi = popt[2*n + 1]
                print(f"Mode {mode}: Overtone {n}: "f"A = {A:.4e}, "f"phi = {phi:.3f}, "f"omega = {omegas[n]:.4f}, "f"tau = {taus[n]:.4f}")
            
            plt.plot(time_ring, y_fit, label="Fit")
        plt.plot(time_ring, signal_ring, label=f"Data: Mode {mode}")
        plt.legend()
        plt.grid()
        plt.show()
        self.total_signal += np.array(signal)

    def graphs(self, waveform='h', modes=[[2,2]], min_t=None, max_t=None, n_overtones=0):
        self.total_signal[:] = 0.0

        for m in modes:
            self.graph(waveform, 'real', m, min_t, max_t, n_overtones)

        plt.plot(self.time, self.total_signal) #still need to sum fit and do this in range
        plt.show()

test = SXSAnalysis()
test.graphs("h", modes=[[2,2], [2,1], [3,3], [3,2]], n_overtones=1) #glitches for 3,1 and 2,0




#Checklist
#Combine imaginary and non imaginary
#Log data
#Add error funtion
#Find optimal start time, psi4 max?
#Sum fit of modes
#Scaling