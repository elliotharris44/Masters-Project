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
    def __init__(self, id = "BAM_0125/data_R01.h5"):
        self.R01_data = h5py.File(id, 'r')

    def graph(self, waveform='h', mode=[2,2], n_overtones=0, plot_start=0, plot_end=0, 
              ring_start=64, fit_start=0, fit_length=50, a=None, mass_bh=None, plot=True, skew=False):
        """
        Arguments: 
        waveform(string): h for strain, 22 ect for psi4 l=2, m=2 mode
        column(int): should be number from 1 to 9, 1 is time then strain ect
        min_time(int): is the minimum time
        max_time(int): is the cutoff time
        """

        mass_total = 3.000475 #should it be adm mass or sum
        if mass_bh is None:
            mass_bh = 2.894 

        if waveform=='h':
            series = self.R01_data['rh_22'] #strain for l=2, m=2
            series_1000 = series['Rh_l2_m2_r01000.txt'][:] #at radius 1000M, 9 columns, time, Re(strain) ect

        else:
            try:
                s = int(waveform)
            except ValueError:
                raise ValueError("Must enter h or two digit integer")
            
            series = self.R01_data[f'rpsi4_{waveform}']
            series_1000 = series[f'Rpsi4_l{waveform[0]}_m{waveform[1]}_r01000.txt'][:]

        if waveform=='h':
            signal = series_1000.T[1] + 1j*series_1000.T[2]
        else:
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
        for n in range(n_overtones+1):
            if waveform=='h':
                grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=n)
            else:
                grav_220 = qnm.modes_cache(s=-2,l=int(waveform[0]),m=int(waveform[1]),n=n)
            if a is None:
                omega, _, _ = grav_220(0.791)
            else:
                omega, _, _ = grav_220(a)
            omega = omega*mass_total/mass_bh
            omegas.append(np.real(omega))
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
        self.signal_plot = signal_plot
        self.signal_fit = y_fit_real + 1j*y_fit_imag
        self.fit_min = np.abs(time_plot-(ring_start+fit_start)).argmin()
        self.fit_max = np.abs(time_plot-(time_plot[self.fit_min]+fit_length)).argmin()
        mi = self.fit_min
        ma = self.fit_max

        if plot:
            _, axs = plt.subplots(2, 1)
            axs[0].plot(time_plot, y_fit_real, label="Fit")
            axs[1].plot(time_plot, y_fit_imag, label="Fit")
            axs[0].axvline(ring_start, color='black', linestyle=':', label='Start of Ringdown')
            axs[1].axvline(ring_start, color='black', linestyle=':', label='Start of Ringdown')
            axs[0].axvline(ring_start+fit_start, color='grey', linestyle=':', label='Start of Fitting')
            axs[1].axvline(ring_start+fit_start, color='grey', linestyle=':', label='Start of Fitting')
            
            axs[0].plot(time_plot, signal_plot.real, label=f"Data: Mode {mode}")
            axs[1].plot(time_plot, signal_plot.imag, label=f"Data: Mode {mode}")
            axs[0].legend(loc='upper right', fontsize='small')
            axs[1].legend(loc='upper right', fontsize='small')
            axs[0].grid()
            axs[1].grid()
            plt.show()
            plt.plot(time_comb, signal_comb)
            plt.show()
            plt.plot(time_plot[mi:ma], signal_plot.real[mi:ma]-y_fit_real[mi:ma])
            plt.show()
            plt.semilogy(time_plot[mi:ma], np.abs(signal_plot[mi:ma]), label='Data')
            plt.semilogy(time_plot[mi:ma], np.abs(self.signal_fit[mi:ma]), label='Fit')
            plt.grid()
            plt.legend()
            plt.show()

    def mismatch(self, waveform='22', mode=[2,2], n_overtones=0, ring_start=64, fit_start=0,
                  fit_length=50, a=None, mass_bh=None, plot=False, skew=False):
        self.graph(waveform=waveform, mode=mode, n_overtones=n_overtones, ring_start=ring_start, fit_start=fit_start,
                    fit_length=fit_length, a=a, mass_bh=mass_bh, plot=plot, skew=skew)
        mi = self.fit_min
        ma = self.fit_max
        self.mm = mismatch_function(self.time_plot[mi:ma], self.signal_plot[mi:ma], self.signal_fit[mi:ma])
        print(self.mm)
    
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
    
    def colour_plot(self):
        spin_axis = np.arange(0.77,0.81,0.001) #x-axis
        mass_axis = np.arange(2.7,3,0.001) #y-axis
        mismatch_axis = np.zeros((len(mass_axis), len(spin_axis))) #'heat'

        for i,spin in enumerate(tqdm.tqdm(spin_axis)):
            for j,mass in enumerate(mass_axis):
                self.mismatch(a=spin, mass_bh=mass)
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
    test = CoReAnalysis()
    test.graph(waveform='22', n_overtones=2, plot_start=40, ring_start=40, fit_start=0, fit_length=50, skew=False)
    #test.mismatch(waveform='22')
    #test.colour_plot()
    #test.mismatch_test2()

#Checklist
#check mismatch still works normal
