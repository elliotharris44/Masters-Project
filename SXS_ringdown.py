import numpy as np
import sxs
import qnm
import matplotlib.pyplot as plt
import scipy
from functions import *
from scipy.integrate import simpson

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

    def graph(self, waveform='h', mode=[2,2], n_overtones=0,
               shift_below=10, shift_above=75, centre=13, fit_start=0):
        """
        Arguments
        waveform(string): h, psi4
        comp(string): real or imag component
        mode(list): list of 2 element lists saying l and m mode
        """
        fig, axs = plt.subplots(2, 1)

        time = self.time

        if waveform=='h':
            modes = self.strain
            
        else:
            modes = self.sim.psi4

        matches = np.all(modes.LM == mode, axis=1)
        ind = np.where(matches)[0]
        signal = modes.data[:,ind].squeeze()
        
        signal_real = signal.real
        signal_imag = signal.imag

        peaks, _ = scipy.signal.find_peaks(signal_real) #maybe need different peaks for real and imag
        main_peak_index = np.argmax(signal_real[peaks])
        t = time[peaks[main_peak_index]]+centre
        print(t-centre)
        
        max_t = t+shift_above #change to time[-1]
        min = np.abs(time-t).argmin()
        max = np.abs(time-max_t).argmin()
        time_ring = time[min:max]
        time_shift = time_ring - time_ring[0] #put all in init

        dt = time_shift[1]-time_shift[0]
        time_fit = time_shift[int(fit_start/dt):]
        time_plot = time[min-int(shift_below/dt):max] - time_ring[0]

        signal_real_ring = signal_real[min+int(fit_start/dt):max]
        signal_imag_ring = signal_imag[min+int(fit_start/dt):max]
        signal_real_plot = signal_real[min-int(shift_below/dt):max]
        signal_imag_plot = signal_imag[min-int(shift_below/dt):max]
        signal_comb = np.concatenate([signal_real_ring, signal_imag_ring])
        time_comb = np.concatenate([time_fit, time_fit+time_fit[-1]+dt])

        p0 = []
        omegas = []
        taus = []
        for n in range(n_overtones+1):
            grav_220 = qnm.modes_cache(s=-2,l=mode[0],m=mode[1],n=n)
            omega, _, _ = grav_220(self.a)
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
        
        axs[0].plot(time_plot, signal_real_plot, label=f"Data: Mode {mode}")
        axs[1].plot(time_plot, signal_imag_plot, label=f"Data: Mode {mode}")
        axs[0].legend()
        axs[1].legend()
        axs[0].grid()
        axs[1].grid()
        plt.show()
        #plt.plot(time_comb, signal_comb)
        #plt.show()

        #self.total_signal_real += np.array(signal)
        self.time_mm = time_plot
        self.h_data = signal_real_plot + 1j*signal_imag_plot
        self.h_fit = y_fit_real + 1j*y_fit_imag

    def graphs(self, waveform='h', modes=[[2,2]], n_overtones=0, shift_below=10, shift_above=75,
               centre=13, fit_start=0):
        #self.total_signal[:] = 0.0

        for m in modes:
            self.graph(waveform, m, n_overtones, shift_below, shift_above, centre, fit_start)

        #plt.plot(self.time, self.total_signal) #still need to sum fit and do this in range
        #plt.show()
    
    def mismatch(self, mode=[2,2], n_overtones=0, shift_above=75):
        for i in [0,5,10,15,20]:
            self.graph(mode=mode, n_overtones=n_overtones, shift_below=0,
                        shift_above=shift_above, centre=i)
            num = simpson(self.h_data*np.conj(self.h_fit), self.time_mm)
            den = np.sqrt(simpson(self.h_data*np.conj(self.h_data), self.time_mm)*simpson(self.h_fit*np.conj(self.h_fit), self.time_mm))
            self.mismatch = 1-np.abs(num/den)
            print(f"Mismatch is {self.mismatch} when delaying fit by {i} M")


test = SXSAnalysis()
test.graphs("psi4", modes=[[2,2], [3,3], [4,4]], n_overtones=0, shift_below=10) 
#test.mismatch()




#Checklist
#Start 0 at max of psi4 2,2 mode in init, make centre change from there
#Sum fit of modes
#Scaling