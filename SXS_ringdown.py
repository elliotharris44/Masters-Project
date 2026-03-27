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
        lev2 = self.sim.metadata['lev_numbers'][-2]
        self.sim2 = sxs.load(f"{id}/Lev{lev2}") #second highest resolution for noise
        try:
            a_vec = self.sim.metadata["remnant_dimensionless_spin"]
        except KeyError:
            a_vec = 'NaN'
        if isinstance(a_vec,str) and a_vec=='NaN':
            self.a = 0.7 #guess
        else:
            self.a = np.linalg.norm(np.array(a_vec))

        self.strain = self.sim.strain
        self.psi4 = self.sim.psi4
        self.time = self.strain.time
        self.strain2 = self.sim2.strain
        self.psi42 = self.sim2.psi4
        self.time2 = self.strain2.time

        self.total_signal = None
        self.total_fit = None

        self.mass_total = self.sim.metadata['initial_mass1'] + self.sim.metadata['initial_mass2']
        print(f"Total mass is {self.mass_total}")

    def graph(self, waveform='h', mode=[2,2], n_overtones=0, plot_start=0, plot_end=0, 
            ring_start=32, fit_start=0, fit_length=50, a=None, mass_bh=None, plot=True, fit=True,
            neg_freq=False, agn_freq=None, noise_plot=False):
        """
        Arguments
        waveform(string): h, psi4
        mode(list): list of 2 element lists saying l and m mode
        """

        time = self.time
        if mass_bh is None:
            mass_bh = self.sim.metadata['remnant_mass']
            if isinstance(mass_bh,str) and mass_bh=='NaN':
                mass_bh = 0.952
        scale = self.mass_total/mass_bh
        if waveform=='h':
            modes = self.strain
        else:
            modes = self.psi4

        if noise_plot:
            if waveform=='h':
                modes2 = self.strain2
            else:
                modes = self.psi42

        ind = np.flatnonzero((modes.LM == mode).all(axis=1))[0]
        signal = modes.data[:, ind]

        peaks, _ = scipy.signal.find_peaks(signal.real) #maybe need different peaks for real and imag
        main_peak_index = np.argmax(signal.real[peaks])
        peak_t = time[peaks[main_peak_index]] 

        time_shift = time-peak_t #t=0 is at peak, full time
        t_min = np.abs(time_shift-(ring_start+fit_start)).argmin() #start of fit arg
        t_max = np.abs(time_shift-(time_shift[t_min]+fit_length)).argmin() #end of fit arg
        time_fit = time_shift[t_min:t_max]-ring_start

        signal_fit = signal[t_min:t_max]
        signal_comb = np.concatenate([signal_fit.real, signal_fit.imag])
        time_comb = np.concatenate([time_fit, time_fit+time_fit[-1]-time_fit[0]]) #potential error with two time coordinates the same

        p0 = []
        omegas = []
        taus = []
        if neg_freq:
            sign = [+1, -1] #for the two frequencies
        else:
            sign =[+1]
        for s in sign:
            for n in range(n_overtones+1):
                grav_lmn = qnm.modes_cache(s=-2,l=mode[0],m=mode[1],n=n)
                if a is None:
                    omega, _, _ = grav_lmn(self.a)
                else:
                    omega, _, _ = grav_lmn(a)
                omega = omega*scale
                omegas.append(s*np.real(omega))
                taus.append(-1/np.imag(omega))
                p0 += [0.1, 0] #parameter guesses
                #print(f"Re[omega] is {omega.real}, -Im[omega] is {-omega.imag}")
        if agn_freq is not None:
            omegas.append(agn_freq[0]) #in units of total binary mass
            taus.append(1/(agn_freq[1]))
            p0 += [0.1, 0]
        popt, _ = scipy.optimize.curve_fit(ringdown_comb(omegas, taus), time_comb, signal_comb, p0)
        
        plot_min = np.abs(time_shift-plot_start).argmin()
        plot_max = np.abs(time_shift-(time_shift[t_max]+plot_end)).argmin()
        time_plot = time_shift[plot_min:plot_max]
        signal_plot = signal[plot_min:plot_max]

        y_fit_real = ringdown_real(omegas, taus)(time_plot-ring_start, *popt)
        y_fit_imag = ringdown_imag(omegas, taus)(time_plot-ring_start, *popt)

        self.time_plot = time_plot
        self.h_data = signal_plot
        self.h_fit = y_fit_real + 1j*y_fit_imag
        self.fit_min = np.abs(time_plot-(ring_start+fit_start)).argmin()
        self.fit_max = np.abs(time_plot-(time_plot[self.fit_min]+fit_length)).argmin()
        mi = self.fit_min
        ma = self.fit_max
        
        if plot:
            _, axs = plt.subplots(2, 1)
            if noise_plot:
                signal2 = modes2.data[:, ind]
                time2 = self.time2
                peaks2, _ = scipy.signal.find_peaks(signal2.real)
                main_peak_index2 = np.argmax(signal2.real[peaks2])
                peak_t2 = time2[peaks2[main_peak_index2]]
                time2_shifted = time2 - peak_t2 + peak_t  #shift R02 onto R01 peak time
                signal2 = np.interp(time, time2_shifted, signal2)
                signal2_plot = signal2[plot_min:plot_max]
                noise = signal2_plot - signal_plot #divide by some factor in future
                axs[0].plot(time_plot, signal2_plot.real, label="R02")
                axs[1].plot(time_plot, signal2_plot.imag, label="R02")

            if fit:
                axs[0].plot(time_plot, y_fit_real, label="Fit")
                axs[1].plot(time_plot, y_fit_imag, label="Fit")
                axs[0].axvline(ring_start+fit_start, color='grey', linestyle=':', label='Start of Fitting')
                axs[1].axvline(ring_start+fit_start, color='grey', linestyle=':', label='Start of Fitting')
            
            axs[0].plot(time_plot, signal_plot.real, label=f"Data: Mode {mode}")
            axs[1].plot(time_plot, signal_plot.imag, label=f"Data: Mode {mode}")
            axs[0].legend(loc='upper right', fontsize='small')
            axs[1].legend(loc='upper right', fontsize='small')
            axs[0].grid()
            axs[1].grid()
            plt.show()
            _, axs = plt.subplots(2, 1)
            axs[0].semilogy(time_plot[mi:ma], np.abs(signal_plot[mi:ma]), label='Data')
            axs[0].semilogy(time_plot[mi:ma], np.abs(self.h_fit[mi:ma]), label='Fit')
            axs[1].semilogy(time_plot[mi:ma], np.abs(signal_plot[mi:ma]-self.h_fit[mi:ma]), label='Residual')
            if noise_plot:
                axs[1].semilogy(time_plot[mi:ma], np.abs(noise[mi:ma]), label='Noise')
            for ax in axs:
                ax.grid()
                ax.legend()
            plt.show()

        if self.total_signal is None:
            self.total_signal = self.h_data.copy()
            self.total_fit = self.h_fit.copy()
        else:
            self.total_signal += self.h_data
            self.total_fit += self.h_fit

    def graphs(self, waveform='h', modes=[[2,2]], n_overtones=0, plot_start=0, plot_end=0, 
            ring_start=32, fit_start=0, fit_length=50, a=None, mass_bh=None, plot=True, fit=True,
            neg_freq=False, agn_freq=None, noise_plot=False):
        self.total_signal = None
        self.total_fit = None

        for m in modes:
            self.graph(waveform, m, n_overtones, plot_start, plot_end, ring_start, fit_start, fit_length, a, mass_bh, plot, fit, neg_freq, agn_freq, noise_plot)

        if plot is True:
            mi = self.fit_min
            ma = self.fit_max
            _, ax = plt.subplots()
            ax.semilogy(self.time_plot[mi:ma], np.abs(self.total_signal[mi:ma]), label='Data')
            ax.semilogy(self.time_plot[mi:ma], np.abs(self.total_fit[mi:ma]), label='Fit')
            ax.grid()
            ax.legend()
            plt.show()
    
    def mismatch(self, modes=[[2,2]], n_overtones=0, ring_start=32, fit_start=0,
                  fit_length=50, a=None, mass_bh=None, plot=False, agn_freq=None):
        self.graphs(modes=modes, n_overtones=n_overtones, ring_start=ring_start, fit_start=fit_start,
                    fit_length=fit_length, a=a, mass_bh=mass_bh, plot=plot, agn_freq=agn_freq)
        mi = self.fit_min
        ma = self.fit_max
        self.mm = mismatch_function(self.time_plot[mi:ma], self.total_signal[mi:ma], self.total_fit[mi:ma])
        #print(self.mm)
    
    def mismatch_test1(self):
        test_param = []
        mm = []
        for i in np.arange(10,70,1):
            self.mismatch(modes=[[4,4]], ring_start=i, fit_length=80-i, a=0.692, mass_bh=0.952)
            test_param.append(i)
            mm.append(self.mm.copy())
        print(f"The mismatch {np.min(mm)} is a minimum when parameter is {test_param[np.argmin(mm)]}")
        plt.plot(test_param, mm)
        plt.show()
    
    def mismatch_test2(self):
        test_param1 = np.arange(0,80,1)
        test_param2 = np.arange(40,100,1)
        mismatch_axis = np.zeros((len(test_param2), len(test_param1)))
        for i,param1 in enumerate(tqdm.tqdm(test_param1)):
            for k,param2 in enumerate(test_param2):
                self.mismatch(ring_start=param1, fit_length=param2)
                mismatch_axis[k,i] = self.mm.copy() #to encourage long signals

        fig, ax = plt.subplots()
        im = ax.imshow(mismatch_axis, norm='log', origin='lower', aspect='auto', extent=[test_param1.min(), test_param1.max(),
            test_param2.min(), test_param2.max()])
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Start of Ringdown")
        ax.set_ylabel("Length of Fitted Region")
        plt.show()

        min_idx = np.unravel_index(np.argmin(mismatch_axis), mismatch_axis.shape)
        print(f"Minimum mismatch at ring_start={test_param1[min_idx[1]]}, length={test_param2[min_idx[0]]}")
        print(mismatch_axis[min_idx[0], min_idx[1]])

    def colour_plot(self, waveform='h', modes=[[2,2]], n_overtones=0, ring_start=32, fit_length=50):
        spin_axis = np.arange(0.67,0.75,0.001) #x-axis
        mass_axis = np.arange(0.945,0.980,0.001) #y-axis
        mismatch_axis = np.zeros((len(mass_axis), len(spin_axis))) #'heat'

        for i,spin in enumerate(tqdm.tqdm(spin_axis)):
            for j,mass in enumerate(mass_axis):
                self.mismatch(modes=[[2,2]], ring_start=ring_start, fit_length=fit_length, a=spin, mass_bh=mass)
                mismatch_axis[j,i] = self.mm.copy()
        
        fig, ax = plt.subplots()
        im = ax.imshow(mismatch_axis, norm='log', origin='lower', aspect='auto', extent=[spin_axis.min(), spin_axis.max(),
            mass_axis.min(), mass_axis.max()])
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Dimensionless spin contant")
        ax.set_ylabel("Black hole mass")
        plt.show()

        min_idx = np.unravel_index(np.argmin(mismatch_axis), mismatch_axis.shape)
        best_mass = mass_axis[min_idx[0]]
        best_spin = spin_axis[min_idx[1]]
        best_mm = mismatch_axis[min_idx[0],min_idx[1]]
        print(f"Minimum mismatch {best_mm} at mass={best_mass}, spin={best_spin}")

    def freq_colour_plot(self, ring_start, fit_length, a, mass_bh):
        re_axis = np.arange(1.05,1.2,0.01) #x-axis
        im_axis = np.arange(0,0.5,0.01) #y-axis
        mismatch_axis = np.zeros((len(im_axis), len(re_axis))) #'heat'

        for i,re in enumerate(tqdm.tqdm(re_axis)):
            for j,im in enumerate(im_axis):
                self.mismatch(modes=[[4,4]], n_overtones=1, ring_start=ring_start, fit_length=fit_length, a=a, mass_bh=mass_bh, agn_freq=[re,im])
                mismatch_axis[j,i] = self.mm.copy()
        
        fig, ax = plt.subplots()
        ima = ax.imshow(mismatch_axis, norm='log', origin='lower', aspect='auto', extent=[re_axis.min(), re_axis.max(),
            im_axis.min(), im_axis.max()])
        fig.colorbar(ima, ax=ax)
        ax.set_xlabel(r"Re[$\omega$]")
        ax.set_ylabel(r"-Im[$\omega$]")
        plt.plot(1.10945, 0.17025, 'x', color='red') #2w(2,2) for 305 simulation
        plt.plot(1.17652, 0.44684, 'o') #w(4,4,2)
        plt.show()

        min_idx = np.unravel_index(np.argmin(mismatch_axis), mismatch_axis.shape)
        best_im = im_axis[min_idx[0]]
        best_re = re_axis[min_idx[1]]
        best_mm = mismatch_axis[min_idx[0],min_idx[1]]
        print(f"Minimum mismatch {best_mm} at -Im[omega]={best_im}, Re[omega]={best_re}")

if __name__ == "__main__":
    test = SXSAnalysis("SXS:BBH:0305")
    test.graphs(waveform='h', modes=[[2,2]], n_overtones=0, plot_start=0, ring_start=32, fit_length=50, a=0.692, mass_bh=0.952, noise_plot=True) 
    #test.mismatch()
    #test.mismatch_test1()
    #test.colour_plot(modes=[[4,4]], n_overtones=1, ring_start=20, fit_length=60)
    #test.freq_colour_plot(ring_start=20, fit_length=60, a=0.692, mass_bh=0.952)

