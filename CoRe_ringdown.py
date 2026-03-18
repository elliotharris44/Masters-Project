import numpy as np
import qnm
import h5py
import matplotlib.pyplot as plt
import scipy
from functions import *
import tqdm

class CoReAnalysis:
    """
    
    """
    def __init__(self, id = "BAM_0125"):
        self.id_R01 = "Data_Tests/" + id + "/R01"
        self.R01_data = h5py.File(self.id_R01 + "/data.h5", 'r')

        with open(self.id_R01 + "/metadata.txt") as f:
            for line in f:
                if line.startswith("id_mass "):
                    self.mass_total = float(line.split("=")[1].strip())
        print(f"Total mass is {self.mass_total}")
        
        self.total_signal = None
        self.total_fit = None

    def graph(self, waveform='psi4', mode=[2,2], n_overtones=0, plot_start=0, plot_end=0, ring_start=64,
               fit_start=0, fit_length=50, a=None, mass_bh=None, plot=True, skew=False, fit=True, neg_freq=False):
        """
        Arguments: 
        waveform(string): h for strain, 22 ect for psi4 l=2, m=2 mode
        column(int): should be number from 1 to 9, 1 is time then strain ect
        min_time(int): is the minimum time
        max_time(int): is the cutoff time
        """
        if mass_bh is None:
            mass_bh = 2.911 

        if waveform=='h':
            series = self.R01_data[f'rh_{str(mode[0])}{str(mode[1])}']
            keys = list(series.keys())
            series_1000 = series[keys[-1]][:]

        else:
            series = self.R01_data[f'rpsi4_{str(mode[0])}{str(mode[1])}']
            keys = list(series.keys())
            series_1000 = series[keys[-1]][:]

        signal = series_1000.T[1] + 1j*series_1000.T[2]
        time = series_1000.T[0]

        peaks, _ = scipy.signal.find_peaks(signal.real) 
        main_peak_index = np.argmax(signal.real[peaks])
        peak_t = time[peaks[main_peak_index]] 

        time_shift = time-peak_t #t=0 is at peak, full time
        t_min = np.abs(time_shift-(ring_start+fit_start)).argmin() #start of fit arg
        t_max = np.abs(time_shift-(time_shift[t_min]+fit_length)).argmin() #end of fit arg
        time_fit = time_shift[t_min:t_max]-ring_start
        self.N = t_max-t_min #for scaled mismatch

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
                if waveform=='h':
                    grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=n)
                else:
                    grav_220 = qnm.modes_cache(s=-2,l=mode[0],m=mode[1],n=n)
                if a is None:
                    omega, _, _ = grav_220(0.798)
                else:
                    omega, _, _ = grav_220(a)
                omega = omega*self.mass_total/mass_bh
                omegas.append(s*np.real(omega)) #positive and negative frequencies
                taus.append(-1/np.imag(omega))
                p0 += [0.01, 0] #parameter guesses
        if skew:
            p0 += [0.0,0.0,0.0,0.0]
            popt, _ = scipy.optimize.curve_fit(ringdown_comb_lin(omegas, taus), time_comb, signal_comb, p0)
        else:
            popt, _ = scipy.optimize.curve_fit(ringdown_comb(omegas, taus), time_comb, signal_comb, p0)
        plot_min = np.abs(time_shift-plot_start).argmin()
        plot_max = np.abs(time_shift-(time_shift[t_max]+plot_end)).argmin()
        time_plot = time_shift[plot_min:plot_max]
        signal_plot = signal[plot_min:plot_max]

        if skew:
            y_fit_real = ringdown_real_lin(omegas, taus)(time_plot-ring_start, *popt)
            y_fit_imag = ringdown_imag_lin(omegas, taus)(time_plot-ring_start, *popt)
        else:
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
            if fit:
                axs[0].plot(time_plot, y_fit_real, label="Fit")
                axs[1].plot(time_plot, y_fit_imag, label="Fit")
                axs[0].axvline(ring_start+fit_start, color='grey', linestyle=':', label='Start of Fitting')
                axs[1].axvline(ring_start+fit_start, color='grey', linestyle=':', label='Start of Fitting')
            axs[0].axvline(ring_start, color='black', linestyle=':', label='Start of Ringdown')
            axs[1].axvline(ring_start, color='black', linestyle=':', label='Start of Ringdown')
            
            axs[0].plot(time_plot, signal_plot.real, label=f"Data: Mode {mode}")
            axs[1].plot(time_plot, signal_plot.imag, label=f"Data: Mode {mode}")
            axs[0].legend(loc='upper right', fontsize='x-small')
            axs[1].legend(loc='upper right', fontsize='x-small')
            axs[0].grid()
            axs[1].grid()
            plt.show()
            #plt.plot(time_comb, signal_comb)
            #plt.show()
            plt.semilogy(time_plot[mi:ma], np.abs(signal_plot[mi:ma]), label='Data')
            plt.semilogy(time_plot[mi:ma], np.abs(self.h_fit[mi:ma]), label='Fit')
            plt.grid()
            plt.legend()
            plt.show()
        
        if self.total_signal is None:
            self.total_signal = self.h_data.copy()
            self.total_fit = self.h_fit.copy()
        else:
            self.total_signal += self.h_data
            self.total_fit += self.h_fit
    
    def graphs(self, waveform='psi4', modes=[[2,2]], n_overtones=0, plot_start=0, plot_end=0, ring_start=64,
            fit_start=0, fit_length=50, a=None, mass_bh=None, plot=True, skew=False, fit=True, neg_freq=True):
        self.total_signal = None
        self.total_fit = None

        for m in modes:
            self.graph(waveform, m, n_overtones, plot_start, plot_end, ring_start, fit_start, fit_length, a, mass_bh, plot, skew, fit, neg_freq)

        if plot is True:
            mi = self.fit_min
            ma = self.fit_max
            _, ax = plt.subplots()
            ax.semilogy(self.time_plot[mi:ma], np.abs(self.total_signal[mi:ma]), label='Data')
            ax.semilogy(self.time_plot[mi:ma], np.abs(self.total_fit[mi:ma]), label='Fit')
            ax.grid()
            ax.legend()
            plt.show()

    def mismatch(self, waveform='psi4', modes=[[2,2]], n_overtones=0, ring_start=64, fit_start=0,
                  fit_length=50, a=None, mass_bh=None, plot=False, skew=False):
        self.graphs(waveform=waveform, modes=modes, n_overtones=n_overtones, ring_start=ring_start, fit_start=fit_start,
                    fit_length=fit_length, a=a, mass_bh=mass_bh, plot=plot, skew=skew)
        mi = self.fit_min
        ma = self.fit_max
        self.mm = mismatch_function(self.time_plot[mi:ma], self.total_signal[mi:ma], self.total_fit[mi:ma])
        #print(self.mm)
    
    def mismatch_test1(self):
        test_param = []
        mm = []
        for i in np.arange(3,60,1):
            self.mismatch(fit_length=i)
            test_param.append(i)
            mm.append(self.mm.copy())
        print(f"The mismatch {np.min(mm)} is a minimum when parameter is {test_param[np.argmin(mm)]}")
        plt.plot(test_param, mm)
        plt.show()

    def mismatch_test2(self):
        test_param1 = np.arange(0,100,1)
        test_param2 = np.arange(0,80,1)
        mismatch_axis = np.zeros((len(test_param2), len(test_param1)))
        for i,param1 in enumerate(tqdm.tqdm(test_param1)):
            for k,param2 in enumerate(test_param2):
                self.mismatch(ring_start=param1, fit_start=param2)
                mismatch_axis[k,i] = self.mm.copy() #to encourage long signals

        fig, ax = plt.subplots()
        im = ax.imshow(mismatch_axis, norm='log', origin='lower', aspect='auto', extent=[test_param1.min(), test_param1.max(),
            test_param2.min(), test_param2.max()])
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Start of Ringdown")
        ax.set_ylabel("Start of Fitted Region")
        plt.show()

        min_idx = np.unravel_index(np.argmin(mismatch_axis), mismatch_axis.shape)
        print(f"Minimum mismatch at ring_start={test_param1[min_idx[1]]}, length={test_param2[min_idx[0]]}")
        print(mismatch_axis[min_idx[0], min_idx[1]])
    
    def colour_plot(self, ring_start=64, fit_length=50):
        spin_axis = np.arange(0.65,0.75,0.001) #x-axis
        mass_axis = np.arange(2.5,2.55,0.001) #y-axis
        mismatch_axis = np.zeros((len(mass_axis), len(spin_axis))) #'heat'

        for i,spin in enumerate(tqdm.tqdm(spin_axis)):
            for j,mass in enumerate(mass_axis):
                self.mismatch(ring_start=ring_start, fit_length=fit_length, a=spin, mass_bh=mass)
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

if __name__ == "__main__":
    test = CoReAnalysis("BAM_0103")
    test.graphs(waveform='psi4', plot_start=600, ring_start=605, fit_length=50,
                a=0.658, mass_bh=2.556, fit=True, neg_freq=True)
    #test.mismatch(waveform='22')
    #test.colour_plot(ring_start=100, fit_length=50)
    #test.mismatch_test2()

