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
               fit_start=0, fit_length=50, a=None, mass_bh=None, skew=False,
               neg_freq=False, retrograde=False, agn_freq=None, shift_time=None, rad=-1, resolution='R01'):
        """
        Arguments:
        waveform(string): h for strain, 22 ect for psi4 l=2, m=2 mode
        column(int): should be number from 1 to 9, 1 is time then strain ect
        min_time(int): is the minimum time
        max_time(int): is the cutoff time
        """
        if mass_bh is None:
            mass_bh = 2.911

        data = self.R02_data if resolution == 'R02' else self.R01_data
        if data is None:
            raise ValueError("R02 data not available for this simulation")

        if waveform=='h':
            series = data[f'rh_{str(mode[0])}{str(mode[1])}']
            keys = list(series.keys())
            series_1000 = series[keys[rad]][:]

        else:
            series = data[f'rpsi4_{str(mode[0])}{str(mode[1])}']
            keys = list(series.keys())
            series_1000 = series[keys[rad]][:]

        self.rad_value = int(keys[rad].split('r')[-1].split('.')[0])
        signal = series_1000.T[1] + 1j*series_1000.T[2]
        time = series_1000.T[0]

        peaks, _ = scipy.signal.find_peaks(np.abs(signal)) 
        main_peak_index = np.argmax(np.abs(signal[peaks]))
        peak_t = time[peaks[main_peak_index]] 

        time_shift = time-peak_t #t=0 is at peak, full time
        if shift_time is not None:
            signal -= np.mean(signal[np.abs(time_shift-(shift_time[0])).argmin():np.abs(time_shift-(shift_time[1])).argmin()])

        t_min = np.abs(time_shift-(ring_start+fit_start)).argmin() #start of fit arg
        t_max = np.abs(time_shift-(time_shift[t_min]+fit_length)).argmin() #end of fit arg
        time_fit = time_shift[t_min:t_max]-ring_start

        signal_fit = signal[t_min:t_max]
        signal_comb = np.concatenate([signal_fit.real, signal_fit.imag])
        time_comb = np.concatenate([time_fit, time_fit]) #potential error with two time coordinates the same

        if agn_freq=="NL":
            omega_22, _, _ = qnm.modes_cache(s=-2,l=2,m=2,n=0)(a)
            omega_22 = omega_22*self.mass_total/mass_bh
            agn_freq = [2*omega_22.real, -2*omega_22.imag]

        p0 = []
        omegas = []
        taus = []
        if neg_freq:
            sign = [+1,-1] #for the two frequencies
        else:
            sign =[+1]
        if retrograde:
            sign_r = [+1, -1]
        else:
            sign_r = [+1]
        for r in sign_r:
            for s in sign:
                for n in range(n_overtones+1):
                    grav_lmn = qnm.modes_cache(s=-2,l=mode[0],m=r*mode[1],n=n)
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

        if self.total_signal is None:
            self.total_signal = self.h_data.copy()
            self.total_fit = self.h_fit.copy()
        else:
            self.total_signal += self.h_data
            self.total_fit += self.h_fit

        amplitudes = [] #complex amplitudes
        N = len(omegas) #not len(popt)//2: skew appends extra non-amplitude params to popt
        for n in range(N):
            A = popt[2*n]
            #print(A)
            phi = popt[2*n+1]
            omega = omegas[n] - 1j/taus[n]
            amplitudes.append(A*np.exp(1j*(omega*ring_start-phi)))

        #print(f"Peak Amplitudes: {np.abs(amplitudes)}")

    def graphs(self, modes=[[2,2]], models=[{}], fit=True, resolutions=None, **kwargs):
        """
        Made with Claude code
        """
        if resolutions is None:
            resolutions = ['R01']
        multi_res = len(resolutions) > 1

        self.total_signal = None
        self.total_fit = None

        results = []
        for res in resolutions:
            for m in modes:
                for model_spec in models:
                    model_spec = model_spec.copy()
                    custom_label = model_spec.pop('label', None)
                    n = model_spec.get('n_overtones', kwargs.get('n_overtones', 0))
                    neg = model_spec.get('neg_freq', kwargs.get('neg_freq', False))
                    ret = model_spec.get('retrograde', kwargs.get('retrograde', False))
                    agn = model_spec.get('agn_freq', kwargs.get('agn_freq', None))
                    rad = model_spec.get('rad', None)
                    self.total_signal = None
                    self.total_fit = None
                    self.graph(mode=m, resolution=res, **{**kwargs, **model_spec})
                    label_parts = []
                    if multi_res:
                        label_parts.append(res)
                    if len(modes) > 1:
                        label_parts.append(f"({m[0]},{m[1]})")
                    label_parts.append(f"n={n}")
                    if neg:
                        label_parts.append("+ neg")
                    if ret:
                        label_parts.append("+ ret")
                    if agn is not None:
                        label_parts.append("+ agn")
                    if rad is not None:
                        label_parts.append(f"r={self.rad_value}")
                    label = custom_label if custom_label is not None else ' '.join(label_parts)
                    results.append({
                        'label': label,
                        'mode': m,
                        'resolution': res,
                        'rad': self.rad_value,
                        'time': self.time_plot.copy(),
                        'h_data': self.h_data.copy(),
                        'h_fit': self.h_fit.copy(),
                        'fit_min': self.fit_min,
                        'fit_max': self.fit_max,
                    })

        ring_start = kwargs.get('ring_start', 64)
        fit_start = kwargs.get('fit_start', 0)
        multi_rad = len({r['rad'] for r in results}) > 1

        _, axs = plt.subplots(2, 1)
        plotted_data = set()
        for r in results:
            mode_key = tuple(r['mode'])
            data_key = (mode_key, r['resolution'], r['rad'])
            data_label = (f"Data: {r['resolution']} " if multi_res else "Data: ") + f"Mode {r['mode']}" + (f" r={r['rad']}" if multi_rad else "")
            if data_key not in plotted_data:
                axs[0].plot(r['time'], r['h_data'].real, label=data_label)
                axs[1].plot(r['time'], r['h_data'].imag)
                plotted_data.add(data_key)
        if fit:
            for r in results:
                axs[0].plot(r['time'], r['h_fit'].real, linestyle='--', label=f"Fit: {r['label']}")
                axs[1].plot(r['time'], r['h_fit'].imag, linestyle='--')
            axs[0].axvline(ring_start + fit_start, color='grey', linestyle=':', label='Start of Fitting')
            axs[1].axvline(ring_start + fit_start, color='grey', linestyle=':')
        axs[0].set_ylabel(r"$\mathrm{Re}[r\Psi_4] [M^{-1}]$", fontsize='large')
        axs[1].set_ylabel(r"$\mathrm{Im}[r\Psi_4] [M^{-1}]$", fontsize='large')
        axs[1].set_xlabel("Time [M]", fontsize='large')
        axs[0].legend(loc='upper right', fontsize='small')
        for ax in axs:
            ax.grid()
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.show()

        _, axs = plt.subplots(2, 1)
        plotted_data = set()
        for r in results:
            mode_key = tuple(r['mode'])
            data_key = (mode_key, r['resolution'], r['rad'])
            data_label = (f"Data: {r['resolution']} " if multi_res else "Data: ") + f"Mode {r['mode']}" + (f" r={r['rad']}" if multi_rad else "")
            mi, ma = r['fit_min'], r['fit_max']
            t = r['time']
            if data_key not in plotted_data:
                axs[0].semilogy(t[mi:ma], np.abs(r['h_data'][mi:ma]), label=data_label)
                plotted_data.add(data_key)
            if fit:
                axs[0].semilogy(t[mi:ma], np.abs(r['h_fit'][mi:ma]), linestyle='--', label=f"Fit: {r['label']}")
            axs[1].semilogy(t[mi:ma], np.abs(r['h_data'][mi:ma] - r['h_fit'][mi:ma]), label=f"Residual: {r['label']}")
        axs[0].set_ylabel(r"$|r\psi_4^{\rm NR}| [M^{-1}]$", fontsize='large')
        axs[1].set_ylabel(r"$|r\psi_4^{\rm NR} - r\psi_4^{\rm QNM}| [M^{-1}]$", fontsize='large')
        axs[1].set_xlabel("Time [M]", fontsize='large')
        for ax in axs:
            ax.grid()
            ax.legend()
        plt.show()

    def savedata(self):
        self.graph(waveform='psi4', mode=[2,2], plot_start=50, ring_start=64, fit_length=50, a=0.8, mass_bh=2.958, fit=True, neg_freq=False)
        mi = self.fit_min
        ma = self.fit_max
        np.savetxt("Runs/fit_output2.txt", np.column_stack([self.time_plot[mi:ma].real, self.h_data[mi:ma].real, self.h_data[mi:ma].imag, self.h_fit[mi:ma].real, self.h_fit[mi:ma].imag]))

    def mismatch(self, printing=False, modes=None, **kwargs):
        self.total_signal = None
        self.total_fit = None
        if modes is not None and 'mode' not in kwargs:
            kwargs['mode'] = modes[0]
        self.graph(**kwargs)
        mi = self.fit_min
        ma = self.fit_max
        self.mm = mismatch_function(self.time_plot[mi:ma], self.total_signal[mi:ma], self.total_fit[mi:ma])
        if printing:
            print(self.mm)
    
    def mismatch_test1(self, window=[0,90], end=80, models=[{}], **kwargs):
        show_legend = len(models) > 1

        for model_spec in models:
            model_spec = model_spec.copy()
            n = model_spec.get('n_overtones', kwargs.get('n_overtones', 0))
            neg = model_spec.get('neg_freq', kwargs.get('neg_freq', False))
            ret = model_spec.get('retrograde', kwargs.get('retrograde', False))
            agn = model_spec.get('agn_freq', kwargs.get('agn_freq', None))
            label_parts = [f"n={n}"]
            if neg:
                label_parts.append("+ neg")
            if ret:
                label_parts.append("+ ret")
            if agn is not None:
                label_parts.append("+ agn")
            label = model_spec.pop('label', ' '.join(label_parts))

            kw = {**kwargs, **model_spec}
            test_param = []
            mm = []
            for i in np.arange(window[0], window[1], 1):
                self.mismatch(ring_start=i, fit_length=end-i, **kw)
                test_param.append(i)
                mm.append(self.mm.copy())
            print(f"{label}: mismatch {np.min(mm):.2e} minimum at start time {test_param[np.argmin(mm)]}")
            plt.semilogy(test_param, mm, label=label if show_legend else None)

        plt.xlabel("Start Time [M]", fontsize='large')
        plt.ylabel("Mismatch", fontsize='large')
        if show_legend:
            plt.legend(fontsize='large')
        plt.grid()
        plt.show()

    def mismatch_test2(self, rads, **kwargs):
        #made with Claude code
        rad_values = []
        mm = []
        for rad in rads:
            self.mismatch(rad=rad, **kwargs)
            rad_values.append(self.rad_value)
            mm.append(self.mm.copy())
        print(f"The mismatch {np.min(mm)} is a minimum at radius {rad_values[np.argmin(mm)]}")
        plt.semilogy(rad_values, mm, 'o-')
        plt.xlabel(r"Extraction Radius $r$ [M]", fontsize='large')
        plt.ylabel("Mismatch", fontsize='large')
        plt.grid()
        plt.show()
    
    def colour_plot(self, ring_start=64, fit_length=50, mass_plot=True, **kwargs):
        spin_axis = np.arange(0.3,0.9,0.01) #x-axis
        mass_axis = np.arange(2.3,3.2,0.01) #y-axis

        # spin_axis = np.arange(0.78,0.82,0.001) #x-axis
        # mass_axis = np.arange(3.23,3.27,0.001) #y-axis

        mismatch_axis = np.zeros((len(mass_axis), len(spin_axis))) #'heat'

        for i,spin in enumerate(tqdm.tqdm(spin_axis)):
            for j,mass in enumerate(mass_axis):
                self.mismatch(ring_start=ring_start, fit_length=fit_length, a=spin, mass_bh=mass, **kwargs)
                mismatch_axis[j,i] = self.mm.copy()
        
        if mass_plot:
            fig, ax = plt.subplots()
            im = ax.imshow(mismatch_axis, norm='log', origin='lower', aspect='auto', extent=[spin_axis.min(), spin_axis.max(),
                mass_axis.min(), mass_axis.max()])
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Mismatch", fontsize='medium')
            ax.set_xlabel(r"Dimensionless Spin $\chi$")
            ax.set_ylabel(r"Black Hole Mass $M_f$ $[M_\odot]$")
            plt.show()

        min_idx = np.unravel_index(np.argmin(mismatch_axis), mismatch_axis.shape)
        self.best_mass = mass_axis[min_idx[0]]
        self.best_spin = spin_axis[min_idx[1]]
        self.best_mm = mismatch_axis[min_idx[0],min_idx[1]]
        print(f"Minimum mismatch {self.best_mm} at mass={self.best_mass}, spin={self.best_spin}")
    
    def freq_colour_plot(self, waveform, ring_start, fit_length, a, mass_bh, freq_plot=True, shift_time=None, rad=-1, neg_freq=False):
        re_axis = np.arange(0.4,1.5,0.02) #x-axis
        im_axis = np.arange(0,0.6,0.01) #y-axis
        mismatch_axis = np.zeros((len(im_axis), len(re_axis))) #'heat'

        for i,re in enumerate(tqdm.tqdm(re_axis)):
            for j,im in enumerate(im_axis):
                self.mismatch(waveform=waveform, modes=[[4,4]], n_overtones=1, ring_start=ring_start, fit_length=fit_length, a=a, mass_bh=mass_bh, neg_freq=neg_freq, agn_freq=[re,im], shift_time=shift_time, rad=rad)
                mismatch_axis[j,i] = self.mm.copy()
        
        scale = self.mass_total/mass_bh
        omega_22, _, _ = qnm.modes_cache(s=-2,l=2,m=2,n=0)(a)
        omega_22 = omega_22*scale
        omega_44, _, _ = qnm.modes_cache(s=-2,l=4,m=4,n=1)(a)
        omega_44 = omega_44*scale

        if freq_plot:
            fig, ax = plt.subplots()
            ima = ax.imshow(mismatch_axis, norm='log', origin='lower', aspect='auto', extent=[re_axis.min(), re_axis.max(),
                im_axis.min(), im_axis.max()])
            fig.colorbar(ima, ax=ax, label='Mismatch')
            ax.set_xlabel(r"Re[$M\omega$]", fontsize='large')
            ax.set_ylabel(r"-Im[$M\omega$]", fontsize='large')
            plt.plot(2*omega_22.real, -2*omega_22.imag, 'x', color='red', label='(2,2,0)x(2,2,0)') #2w(2,2)
            plt.plot(omega_44.real, -omega_44.imag, '*', color='black', label='(4,4,2)') #w(4,4,2) 
            plt.legend()
            plt.show()

        min_idx = np.unravel_index(np.argmin(mismatch_axis), mismatch_axis.shape)
        self.best_im = im_axis[min_idx[0]]
        self.best_re = re_axis[min_idx[1]]
        self.best_mm = mismatch_axis[min_idx[0],min_idx[1]]
        print(f"Minimum mismatch {self.best_mm} at Re[omega]={self.best_re}, -Im[omega]={self.best_im}")
        print(f"Non-linear mode (2,2,0)x(2,2,0): Re[omega]={2*omega_22.real}, -Im[omega]={-2*omega_22.imag}")
        print(f"Second overtone (4,4,2): Re[omega]={omega_44.real}, -Im[omega]={-omega_44.imag}")

    def fund_colour_plot(self, waveform, ring_start, fit_length, a, mass_bh, freq_plot=True, neg_freq=False):
                re_axis = np.arange(0.5,0.6,0.001) #x-axis
                im_axis = np.arange(0.07,0.12,0.001) #y-axis
                mismatch_axis = np.zeros((len(im_axis), len(re_axis))) #'heat'
        
                for i,re in enumerate(tqdm.tqdm(re_axis)):
                    for j,im in enumerate(im_axis):
                        self.mismatch(waveform=waveform, modes=[[2,2]], n_overtones=-1, ring_start=ring_start, fit_length=fit_length, a=a, mass_bh=mass_bh, neg_freq=neg_freq, agn_freq=[re,im])
                        mismatch_axis[j,i] = self.mm.copy()
        
                omega_22, _, _ = qnm.modes_cache(s=-2,l=2,m=2,n=0)(a)
                omega_22 = omega_22*self.mass_total/mass_bh
        
                if freq_plot:
                    fig, ax = plt.subplots()
                    ima = ax.imshow(mismatch_axis, norm='log', origin='lower', aspect='auto', extent=[re_axis.min(), re_axis.max(),
                        im_axis.min(), im_axis.max()])
                    fig.colorbar(ima, ax=ax, label='Mismatch')
                    ax.set_xlabel(r"Re[$M\omega$]", fontsize='large')
                    ax.set_ylabel(r"-Im[$M\omega$]", fontsize='large')
                    plt.plot(omega_22.real, -omega_22.imag, 'x', color='red', label='(2,2,0) qnm Package')
                    plt.legend()
                    plt.show()
        
                min_idx = np.unravel_index(np.argmin(mismatch_axis), mismatch_axis.shape)
                self.best_im = im_axis[min_idx[0]]
                self.best_re = re_axis[min_idx[1]]
                self.best_mm = mismatch_axis[min_idx[0],min_idx[1]]
                print(f"Minimum mismatch {self.best_mm} at Re[omega]={self.best_re}, -Im[omega]={self.best_im}")
                print(f"Fundamental mode (2,2,0): Re[omega]={omega_22.real}, -Im[omega]={-omega_22.imag}")
    
    def mass_time_drift(self, fit_start_range, fit_end, **kwargs):
        times = []
        best_masses = []
        best_spins = []
        best_mms = []

        for t in tqdm.tqdm(np.arange(fit_start_range[0], fit_start_range[1], 1)):
            self.colour_plot(ring_start=t, fit_length=fit_end-t, mass_plot=False, **kwargs)
            times.append(t)
            best_masses.append(self.best_mass)
            best_spins.append(self.best_spin)
            best_mms.append(self.best_mm)
        
        fig, ax = plt.subplots()
        sc = ax.scatter(best_spins, best_masses, c=times)
        fig.colorbar(sc, ax=ax, label='Start Times [M]')
        ax.grid()
        ax.set_xlabel(r"Dimensionless Spin $\chi$", fontsize='large')
        ax.set_ylabel(r"Black Hole Mass $M_f$ $[M_\odot]$", fontsize='large')

        ax.plot(0.756, 3.234, 'x', color='red', label=r'Literature $M_f$, $\chi$') #must change each time
        ax.legend(fontsize='large')
        
        fig2, ax2 = plt.subplots()
        ax2.semilogy(times, best_mms)
        ax2.grid()
        plt.show()

    def mass_rad_drift(self, rads, ring_start=64, fit_length=50, **kwargs):
        rad_values = []
        best_masses = []
        best_spins = []
        best_mms = []

        for rad in tqdm.tqdm(rads):
            self.colour_plot(ring_start=ring_start, fit_length=fit_length, mass_plot=False, rad=rad, **kwargs)
            rad_values.append(self.rad_value)
            best_masses.append(self.best_mass)
            best_spins.append(self.best_spin)
            best_mms.append(self.best_mm)

        fig, ax = plt.subplots()
        sc = ax.scatter(best_spins, best_masses, c=rad_values)
        fig.colorbar(sc, ax=ax, label=r"Extraction Radius $r$ [M]")
        ax.grid()
        ax.set_xlabel(r"Dimensionless Spin $\chi$", fontsize='large')
        ax.set_ylabel(r"Black Hole Mass $M_{f}$ $(M_\odot)$", fontsize='large')

        fig2, ax2 = plt.subplots()
        ax2.semilogy(rad_values, best_mms, 'o-')
        ax2.grid()
        ax2.set_xlabel(r"Extraction Radius $r$ [M]")
        ax2.set_ylabel("Mismatch")
        plt.show()

    def freq_time_drift(self, waveform, fit_start_range, fit_end, a, mass_bh, **kwargs):
        times = []
        best_ims = []
        best_res = []
        best_mms = []

        for t in tqdm.tqdm(np.arange(fit_start_range[0], fit_start_range[1], 1)):
            self.freq_colour_plot(waveform, t, fit_end-t, a, mass_bh, freq_plot=False, **kwargs)
            times.append(t)
            best_ims.append(self.best_im)
            best_res.append(self.best_re)
            best_mms.append(self.best_mm)
        
        fig, ax = plt.subplots()
        sc = ax.scatter(best_res, best_ims, c=times)
        fig.colorbar(sc, ax=ax, label='Start Times [M]')
        ax.grid()
        ax.set_xlabel(r"Re[$M\omega$]", fontsize='large')
        ax.set_ylabel(r"-Im[$M\omega$]", fontsize='large')
        scale = self.mass_total/mass_bh
        omega_22, _, _ = qnm.modes_cache(s=-2,l=2,m=2,n=0)(a)
        omega_22 = omega_22*scale
        omega_44, _, _ = qnm.modes_cache(s=-2,l=4,m=4,n=2)(a)
        omega_44 = omega_44*scale
        ax.plot(2*omega_22.real, -2*omega_22.imag, 'x', color='red', label='(2,2,0)x(2,2,0)') #2w(2,2)
        ax.plot(omega_44.real, -omega_44.imag, '*', color='black', label='(4,4,2)') #w(4,4,2)
        ax.legend()
        
        fig2, ax2 = plt.subplots()
        ax2.semilogy(times, best_mms)
        ax2.grid()
        ax2.set_xlabel("Start Time [M]", fontsize='large')
        ax2.set_ylabel("Mismatch", fontsize='large')
        plt.show()

if __name__ == "__main__":
    test = CoReAnalysis("BAM_0140")
    # test.graphs(waveform='psi4', modes=[[2,2]], models=[{'n_overtones':0}],
    #             plot_start=-300, ring_start=0, fit_length=3000, a=0.79, mass_bh=2.864, neg_freq=False, fit=False)
    #test.colour_plot(modes=[[2,2]], n_overtones=1, ring_start=0, fit_length=34, neg_freq=False)
    #test.mismatch_test1(waveform='psi4', mode=[2,2], models=[{'n_overtones':0}], a=0.73, mass_bh=2.67, neg_freq=False, window=[0,20], end=34)
    #test.mismatch_test2(rads=[-1, -2, -3, -4, -5, -6,-7, -8, -9, -10, -11, -12], n_overtones=1, ring_start=53, fit_length=60, a=0.756, mass_bh=3.234)
    #test.mismatch(waveform='psi4', mode=[2,2], n_overtones=0, ring_start=55, fit_length=50, a=0.79, mass_bh=2.864, neg_freq=True, printing=True)
    #test.savedata()
    #test.freq_colour_plot(waveform='psi4', ring_start=22, fit_length=50, a=0.79, mass_bh=3.26, neg_freq=False)
    test.fund_colour_plot(waveform='psi4', ring_start=40, fit_length=68, a=0.795, mass_bh=3.231)
    #test.mass_time_drift(waveform='psi4', mode=[2,2], n_overtones=0, fit_start_range=[888,893], fit_end=914)
    #test.freq_time_drift('psi4', [18,28], 72, a=0.79, mass_bh=3.26, neg_freq=False)
    #test.mass_rad_drift(rads=[-1,-2,-3,-4,-5,-6], n_overtones=1, ring_start=882, fit_length=32)

