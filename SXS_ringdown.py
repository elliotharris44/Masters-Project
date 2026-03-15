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

        a_vec = self.sim.metadata["remnant_dimensionless_spin"]
        if isinstance(a_vec,str) and a_vec=='NaN':
            self.a = 0.7 #guess
        else:
            self.a = np.linalg.norm(np.array(a_vec))

        self.strain = self.sim.strain
        self.time = self.strain.time

        self.total_signal = None
        self.total_fit = None

        self.mass_total = self.sim.metadata['initial_mass1'] + self.sim.metadata['initial_mass2']
        print(f"Total mass is {self.mass_total}")

    def graph(self, waveform='h', mode=[2,2], n_overtones=0, plot_start=0, plot_end=0, 
              ring_start=32, fit_start=0, fit_length=50, a=None, mass_bh=None, plot=True, fit=True):
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
  
        if waveform=='h':
            modes = self.strain
            
        else:
            modes = self.sim.psi4

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
        for n in range(n_overtones+1):
            grav_220 = qnm.modes_cache(s=-2,l=mode[0],m=mode[1],n=n)
            if a is None:
                omega, _, _ = grav_220(self.a)
            else:
                omega, _, _ = grav_220(a)
            omega = omega*self.mass_total/mass_bh
            omegas.append(np.real(omega))
            taus.append(-1/np.imag(omega))
            p0 += [0.1, 0] #parameter guesses

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
            if fit:
                axs[0].plot(time_plot, y_fit_real, label="Fit")
                axs[1].plot(time_plot, y_fit_imag, label="Fit")
                axs[0].axvline(ring_start+fit_start, color='grey', linestyle=':', label='Start of Fitting')
                axs[1].axvline(ring_start+fit_start, color='grey', linestyle=':', label='Start of Fitting')
            axs[0].axvline(ring_start, color='black', linestyle=':', label='Start of Ringdown')
            axs[1].axvline(ring_start, color='black', linestyle=':', label='Start of Ringdown')
            
            
            axs[0].plot(time_plot, signal_plot.real, label=f"Data: Mode {mode}")
            axs[1].plot(time_plot, signal_plot.imag, label=f"Data: Mode {mode}")
            axs[0].legend(loc='upper right', fontsize='small')
            axs[1].legend(loc='upper right', fontsize='small')
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

    def graphs(self, waveform='h', modes=[[2,2]], n_overtones=0, plot_start=0, plot_end=0, 
              ring_start=32, fit_start=0, fit_length=50, a=None, mass_bh=None, plot=True, fit=True):
        self.total_signal = None
        self.total_fit = None

        for m in modes:
            self.graph(waveform, m, n_overtones, plot_start, plot_end, ring_start, fit_start, fit_length, a, mass_bh, plot, fit)

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
                  fit_length=50, a=None, mass_bh=None, plot=False):
        self.graphs(modes=modes, n_overtones=n_overtones, ring_start=ring_start, fit_start=fit_start,
                    fit_length=fit_length, a=a, mass_bh=mass_bh, plot=plot)
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

    def colour_plot(self, ring_start=32, fit_length=50):
        spin_axis = np.arange(0.65,0.75,0.001) #x-axis
        mass_axis = np.arange(0.925,0.975,0.001) #y-axis
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
    test = SXSAnalysis("SXS:BBH:0305")
    #test.graphs(modes=[[2,2]], plot_start=-30, a=0.691, mass_bh=0.946, fit=False) 
    #test.mismatch()
    #test.mismatch_test2()
    test.colour_plot()




#Checklist
#check effect of shift above
#reduce length to 50 as reduces error
#make more efficient
#add fit end line
#overones causing problems