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
        self.R01_data = h5py.File(f"Data_Tests/{id}/R01/data.h5", 'r')
        try:
            self.R02_data = h5py.File(f"Data_Tests/{id}/R02/data.h5", 'r')
        except FileNotFoundError:
            self.R02_data = None #claude code
        with open(f"Data_Tests/{id}/R01/metadata.txt") as f:
            for line in f:
                if line.startswith("id_mass "):
                    self.mass_total = float(line.split("=")[1].strip())
        print(f"Total mass is {self.mass_total}")
        
        self.total_signal = None
        self.total_fit = None

    def graph(self, waveform='psi4', mode=[2,2], n_overtones=0, plot_start=0, plot_end=0, ring_start=64,
               fit_start=0, fit_length=50, a=None, mass_bh=None, plot=True, skew=False, fit=True,
               neg_freq=False, agn_freq=None, noise_plot=False, rad=-1):
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
            series_1000 = series[keys[rad]][:]

        else:
            series = self.R01_data[f'rpsi4_{str(mode[0])}{str(mode[1])}']
            keys = list(series.keys())
            series_1000 = series[keys[rad]][:]
        
        if noise_plot:
            if self.R02_data is None:
                print("Warning: R02 data not available, disabling noise_plot.")
                noise_plot = False
            else:
                if waveform=='h':
                    series2 = self.R02_data[f'rh_{str(mode[0])}{str(mode[1])}']
                    keys2 = list(series2.keys())
                    series_10002 = series2[keys2[rad]][:]
                else:
                    series2 = self.R02_data[f'rpsi4_{str(mode[0])}{str(mode[1])}']
                    keys2 = list(series2.keys())
                    series_10002 = series2[keys2[rad]][:]

        signal = series_1000.T[1] + 1j*series_1000.T[2]
        time = series_1000.T[0]

        peaks, _ = scipy.signal.find_peaks(np.abs(signal)) 
        main_peak_index = np.argmax(np.abs(signal[peaks]))
        peak_t = time[peaks[main_peak_index]] 

        time_shift = time-peak_t #t=0 is at peak, full time
        t_min = np.abs(time_shift-(ring_start+fit_start)).argmin() #start of fit arg
        t_max = np.abs(time_shift-(time_shift[t_min]+fit_length)).argmin() #end of fit arg
        time_fit = time_shift[t_min:t_max]-ring_start
        self.N = t_max-t_min #for scaled mismatch

        signal_fit = signal[t_min:t_max]
        signal_comb = np.concatenate([signal_fit.real, signal_fit.imag])
        time_comb = np.concatenate([time_fit, time_fit]) #potential error with two time coordinates the same

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
                    omega, _, _ = grav_lmn(0.798)
                else:
                    omega, _, _ = grav_lmn(a)
                omega = omega*self.mass_total/mass_bh
                omegas.append(s*np.real(omega)) #positive and negative frequencies
                taus.append(-1/np.imag(omega))
                p0 += [0.01, 0] #parameter guesses
                #print(f"Re[omega] is {omega.real}, -Im[omega] is {-omega.imag}")
        if agn_freq is not None:
            omegas.append(agn_freq[0]) #in units of total binary mass
            taus.append(1/(agn_freq[1]))
            p0 += [0.1, 0]
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
            if noise_plot:
                signal2 = series_10002.T[1] + 1j*series_10002.T[2]
                time2 = series_10002.T[0]
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
                axs[1].axvline(ring_start+fit_start, color='grey', linestyle=':')
            axs[0].plot(time_plot, signal_plot.real, label=f"Data: Mode {mode}")
            axs[1].plot(time_plot, signal_plot.imag)

            axs[0].set_ylabel(r"$\mathrm{Re}[\Psi_{4}$]", fontsize='large')
            axs[1].set_ylabel(r"$\mathrm{Im}[\Psi_{4}$]", fontsize='large')
            axs[1].set_xlabel("Time (M)", fontsize='large')
            axs[0].legend(loc='upper right', fontsize='small')
            for ax in axs:
                ax.grid()
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
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
    
    def graphs(self, waveform='psi4', modes=[[2,2]], n_overtones=0, plot_start=0, plot_end=0, ring_start=64,
            fit_start=0, fit_length=50, a=None, mass_bh=None, plot=True, skew=False, fit=True, neg_freq=False,
            agn_freq=None, noise_plot=False, rad=-1):
        self.total_signal = None
        self.total_fit = None

        for m in modes:
            self.graph(waveform, m, n_overtones, plot_start, plot_end, ring_start, fit_start, fit_length, a, mass_bh, plot, skew, fit, neg_freq, agn_freq, noise_plot, rad)

        if plot is True:
            mi = self.fit_min
            ma = self.fit_max
            _, ax = plt.subplots()
            ax.semilogy(self.time_plot[mi:ma], np.abs(self.total_signal[mi:ma]), label='Data')
            ax.semilogy(self.time_plot[mi:ma], np.abs(self.total_fit[mi:ma]), label='Fit')
            ax.grid()
            ax.legend()
            plt.show()

    def savedata(self):
        self.graph(waveform='psi4', mode=[2,2], plot_start=50, ring_start=64, fit_length=50, a=0.8, mass_bh=2.958, fit=True, neg_freq=False)
        mi = self.fit_min
        ma = self.fit_max
        np.savetxt("Runs/fit_output2.txt", np.column_stack([self.time_plot[mi:ma].real, self.h_data[mi:ma].real, self.h_data[mi:ma].imag, self.h_fit[mi:ma].real, self.h_fit[mi:ma].imag]))

    def mismatch(self, waveform='psi4', modes=[[2,2]], n_overtones=0, ring_start=64, fit_start=0,
                  fit_length=50, a=None, mass_bh=None, plot=False, skew=False, neg_freq=False, agn_freq=None, rad=-1, printing=False):
        self.graphs(waveform=waveform, modes=modes, n_overtones=n_overtones, ring_start=ring_start, fit_start=fit_start,
                    fit_length=fit_length, a=a, mass_bh=mass_bh, plot=plot, skew=skew, neg_freq=neg_freq, agn_freq=agn_freq, rad=rad)
        mi = self.fit_min
        ma = self.fit_max
        self.mm = mismatch_function(self.time_plot[mi:ma], self.total_signal[mi:ma], self.total_fit[mi:ma])
        if printing:
            print(self.mm)
    
    def mismatch_test1(self, modes=[[2,2]], n_overtones=1, a=None, mass_bh=None, neg_freq=False, end=80, rad=-1):
        test_param = []
        mm = []
        for i in np.arange(0,40,1):
            self.mismatch(modes=modes, n_overtones=n_overtones, ring_start=i, fit_length=end-i, a=a, mass_bh=mass_bh, neg_freq=neg_freq, rad=rad)
            test_param.append(i)
            mm.append(self.mm.copy())
        print(f"The mismatch {np.min(mm)} is a minimum when parameter is {test_param[np.argmin(mm)]}")
        plt.semilogy(test_param, mm)
        plt.grid()
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
    
    def colour_plot(self, waveform='psi4', modes=[[2,2]], n_overtones=0, ring_start=64, fit_length=50, neg_freq=False, mass_plot=True, rad=-1):
        spin_axis = np.arange(0.5,0.8,0.01) #x-axis
        mass_axis = np.arange(2.2,3,0.01) #y-axis

        # spin_axis = np.arange(0.66,0.7,0.001) #x-axis
        # mass_axis = np.arange(2.5,2.54,0.001) #y-axis

        mismatch_axis = np.zeros((len(mass_axis), len(spin_axis))) #'heat'

        for i,spin in enumerate(tqdm.tqdm(spin_axis)):
            for j,mass in enumerate(mass_axis):
                self.mismatch(waveform=waveform, modes=modes, n_overtones=n_overtones, ring_start=ring_start, fit_length=fit_length, a=spin, mass_bh=mass, neg_freq=neg_freq, rad=rad)
                mismatch_axis[j,i] = self.mm.copy()
        
        if mass_plot:
            fig, ax = plt.subplots()
            im = ax.imshow(mismatch_axis, norm='log', origin='lower', aspect='auto', extent=[spin_axis.min(), spin_axis.max(),
                mass_axis.min(), mass_axis.max()])
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Mismatch", fontsize='medium')
            ax.set_xlabel(r"Dimensionless Spin $a/M$")
            ax.set_ylabel(r"Black Hole Mass $M_f$ $(M_\odot)$")
            plt.show()

        min_idx = np.unravel_index(np.argmin(mismatch_axis), mismatch_axis.shape)
        self.best_mass = mass_axis[min_idx[0]]
        self.best_spin = spin_axis[min_idx[1]]
        self.best_mm = mismatch_axis[min_idx[0],min_idx[1]]
        print(f"Minimum mismatch {self.best_mm} at mass={self.best_mass}, spin={self.best_spin}")
    
    def freq_colour_plot(self, ring_start, fit_length, a, mass_bh, freq_plot=True, rad=-1):
        re_axis = np.arange(0.4,1.5,0.02) #x-axis
        im_axis = np.arange(0,0.5,0.01) #y-axis
        mismatch_axis = np.zeros((len(im_axis), len(re_axis))) #'heat'

        for i,re in enumerate(tqdm.tqdm(re_axis)):
            for j,im in enumerate(im_axis):
                self.mismatch(modes=[[4,4]], n_overtones=1, ring_start=ring_start, fit_length=fit_length, a=a, mass_bh=mass_bh, neg_freq=False, agn_freq=[re,im], rad=rad)
                mismatch_axis[j,i] = self.mm.copy()
        
        if freq_plot:
            fig, ax = plt.subplots()
            ima = ax.imshow(mismatch_axis, norm='log', origin='lower', aspect='auto', extent=[re_axis.min(), re_axis.max(),
                im_axis.min(), im_axis.max()])
            fig.colorbar(ima, ax=ax)
            ax.set_xlabel(r"Re[$\omega$]")
            ax.set_ylabel(r"-Im[$\omega$]")
            scale = self.mass_total/mass_bh
            omega_22, _, _ = qnm.modes_cache(s=-2,l=2,m=2,n=0)(a)
            omega_22 = omega_22*scale
            omega_44, _, _ = qnm.modes_cache(s=-2,l=4,m=4,n=2)(a)
            omega_44 = omega_44*scale
            plt.plot(2*omega_22.real, -2*omega_22.imag, 'x', color='red', label='(2,2,0)x(2,2,0)') #2w(2,2)
            plt.plot(omega_44.real, -omega_44.imag, 'o', label='(4,4,2)') #w(4,4,2) 
            plt.legend()
            plt.show()

        min_idx = np.unravel_index(np.argmin(mismatch_axis), mismatch_axis.shape)
        self.best_im = im_axis[min_idx[0]]
        self.best_re = re_axis[min_idx[1]]
        self.best_mm = mismatch_axis[min_idx[0],min_idx[1]]
        print(f"Minimum mismatch {self.best_mm} at Re[omega]={self.best_re}, -Im[omega]={self.best_im}")
        print(f"Non-linear mode (2,2,0)x(2,2,0): Re[omega]={2*omega_22.real}, -Im[omega]={-2*omega_22.imag}")
        print(f"Second overtone (4,4,2): Re[omega]={omega_44.real}, -Im[omega]={-omega_44.imag}")
    
    def mass_time_drift(self, fit_start_range, fit_end, rad=-1):
        times = []
        best_masses = []
        best_spins = []
        best_mms = []

        for t in tqdm.tqdm(np.arange(fit_start_range[0], fit_start_range[1], 5)):
            self.colour_plot(n_overtones=1, ring_start=t, fit_length=fit_end-t, mass_plot=False, rad=rad)
            times.append(t)
            best_masses.append(self.best_mass)
            best_spins.append(self.best_spin)
            best_mms.append(self.best_mm)
        
        fig, ax = plt.subplots()
        sc = ax.scatter(best_spins, best_masses, c=times)
        fig.colorbar(sc, ax=ax, label='Initial Times')
        ax.grid()
        ax.set_xlabel(r"Spins")
        ax.set_ylabel(r"Masses")
        ax.plot(0.756, 3.234, 'x', color='red') #must change each time
        
        fig2, ax2 = plt.subplots()
        ax2.semilogy(times, best_mms)
        ax2.grid()
        plt.show()

    def freq_time_drift(self, fit_start_range, fit_end, a, mass_bh, rad=-1):
        times = []
        best_ims = []
        best_res = []
        best_mms = []

        for t in tqdm.tqdm(np.arange(fit_start_range[0], fit_start_range[1], 1)):
            self.freq_colour_plot(t, fit_end-t, a, mass_bh, freq_plot=False, rad=rad)
            times.append(t)
            best_ims.append(self.best_im)
            best_res.append(self.best_re)
            best_mms.append(self.best_mm)
        
        fig, ax = plt.subplots()
        sc = ax.scatter(best_res, best_ims, c=times)
        fig.colorbar(sc, ax=ax, label='Initial Times')
        ax.grid()
        ax.set_xlabel(r"Re[$\omega$]")
        ax.set_ylabel(r"-Im[$\omega$]")
        scale = self.mass_total/mass_bh
        omega_22, _, _ = qnm.modes_cache(s=-2,l=2,m=2,n=0)(a)
        omega_22 = omega_22*scale
        omega_44, _, _ = qnm.modes_cache(s=-2,l=4,m=4,n=2)(a)
        omega_44 = omega_44*scale
        ax.plot(2*omega_22.real, -2*omega_22.imag, 'x', color='red', label='(2,2,0)x(2,2,0)') #2w(2,2)
        ax.plot(omega_44.real, -omega_44.imag, '*', color='pink', label='(4,4,2)') #w(4,4,2)
        ax.legend()
        
        fig2, ax2 = plt.subplots()
        ax2.semilogy(times, best_mms)
        ax2.grid()
        plt.show()

if __name__ == "__main__":
    test = CoReAnalysis("BAM_0214")
    # test.graphs(waveform='psi4', modes=[[4,4]], n_overtones=1, plot_start=-600, ring_start=10, fit_length=65,
    #             a=0.528, mass_bh=6.105, fit=False, neg_freq=False, agn_freq=None, noise_plot=False, rad=-1)
    #test.colour_plot(modes=[[2,2]], n_overtones=1, ring_start=126, fit_length=40, neg_freq=False, rad=-1)
    #test.mismatch_test1(modes=[[4,4]], n_overtones=1, a=0.528, mass_bh=6.105, neg_freq=False, end=75)
    #test.mismatch(waveform='psi4', modes=[[2,2]], n_overtones=1, ring_start=53, fit_length=60, a=0.756, mass_bh=3.234, rad=-7, printing=True)
    #test.savedata()
    test.freq_colour_plot(ring_start=8, fit_length=67, a=0.528, mass_bh=6.105)
    #test.mass_time_drift([60,100], 100)
    #test.freq_time_drift([5,25], 75, a=0.528, mass_bh=6.105)

