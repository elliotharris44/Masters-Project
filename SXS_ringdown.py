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
            self.a_metadata = 0.7 #guess
        else:
            self.a_metadata = np.linalg.norm(np.array(a_vec))

        self.strain = self.sim.strain
        self.psi4 = self.sim.psi4
        self.time = self.strain.time
        self.strain2 = self.sim2.strain
        self.psi42 = self.sim2.psi4
        self.time2 = self.strain2.time

        self.total_signal = None
        self.total_fit = None

        self.mass_total = self.sim.metadata['initial_mass1'] + self.sim.metadata['initial_mass2']

    def graph(self, waveform='psi4', mode=[2,2], n_overtones=0, plot_start=0, plot_end=0,
            ring_start=32, fit_start=0, fit_length=50, a=None, mass_bh=None,
            neg_freq=False, retrograde=False, agn_freq=None):
        """
        Arguments
        waveform(string): h, psi4
        mode(list): l and m mode
        """

        time = self.time
        if mass_bh is None:
            mass_bh = self.sim.metadata['remnant_mass']
            if isinstance(mass_bh,str) and mass_bh=='NaN':
                mass_bh = 0.952
        self.scale = self.mass_total/mass_bh
        if waveform=='h':
            modes = self.strain
        else:
            modes = self.psi4

        ind = np.flatnonzero((modes.LM == mode).all(axis=1))[0]
        signal = modes.data[:, ind]

        peaks, _ = scipy.signal.find_peaks(np.abs(signal))
        main_peak_index = np.argmax(np.abs(signal[peaks]))
        peak_t = time[peaks[main_peak_index]]

        time_shift = time-peak_t
        t_min = np.abs(time_shift-(ring_start+fit_start)).argmin()
        t_max = np.abs(time_shift-(time_shift[t_min]+fit_length)).argmin()
        time_fit = time_shift[t_min:t_max]-ring_start

        signal_fit = signal[t_min:t_max]
        signal_comb = np.concatenate([signal_fit.real, signal_fit.imag])
        time_comb = np.concatenate([time_fit, time_fit])

        if a is None:
            self.a = self.a_metadata.copy()
        else:
            self.a = a

        if agn_freq=="NL":
            omega_22, _, _ = qnm.modes_cache(s=-2,l=2,m=2,n=0)(self.a)
            omega_22 = omega_22*self.scale
            agn_freq = [2*omega_22.real, -2*omega_22.imag]

        p0 = []
        omegas = []
        taus = []
        if neg_freq:
            sign = [+1, -1]
        else:
            sign = [+1]
        if retrograde:
            sign_r = [+1, -1]
        else:
            sign_r = [+1]
        for r in sign_r:
            for s in sign:
                for n in range(n_overtones+1):
                    grav_lmn = qnm.modes_cache(s=-2,l=mode[0],m=r*mode[1],n=n)
                    omega, _, mix = grav_lmn(self.a)
                    omega = omega*self.scale
                    omegas.append(s*np.real(omega))
                    taus.append(-1/np.imag(omega))
                    p0 += [0.01, 0]
                    #print(f"Re[omega] is {s*omega.real}, -Im[omega] is {-omega.imag}")
        if agn_freq is not None:
            omegas.append(agn_freq[0])
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

        if self.total_signal is None:
            self.total_signal = self.h_data.copy()
            self.total_fit = self.h_fit.copy()
        else:
            self.total_signal += self.h_data
            self.total_fit += self.h_fit
        
        amplitudes = [] #complex amplitudes
        N = len(popt)//2 
        for n in range(N):
            A = popt[2*n]
            phi = popt[2*n+1]
            omega = omegas[n] - 1j/taus[n]
            amplitudes.append(A*np.exp(1j*(omega*ring_start+phi)))
        # if mode == [4,4]:
        #     print(np.real(mix[0]))
        #     amplitudes = amplitudes/np.real(mix[0])

        #print(f"Peak Amplitudes: {np.abs(amplitudes)}")

    def graphs(self, modes=[[2,2]], models=[{}], fit=True, **kwargs):
        self.total_signal = None
        self.total_fit = None

        results = []
        for m in modes:
            for model_spec in models:
                model_spec = model_spec.copy()
                self.total_signal = None
                self.total_fit = None
                label_parts = []
                if len(modes) > 1:
                    label_parts.append(f"({m[0]},{m[1]})")
                n = model_spec.get('n_overtones', kwargs.get('n_overtones', 0))
                neg = model_spec.get('neg_freq', kwargs.get('neg_freq', False))
                ret = model_spec.get('retrograde', kwargs.get('retrograde', False))
                agn = model_spec.get('agn_freq', kwargs.get('agn_freq', None))
                label_parts.append(f"n={n}")
                if neg:
                    label_parts.append("+ neg")
                if ret:
                    label_parts.append("+ ret")
                if agn is not None:
                    label_parts.append("+ agn")
                label = model_spec.pop('label', ' '.join(label_parts))
                self.graph(mode=m, **{**kwargs, **model_spec})
                results.append({
                    'label': label,
                    'mode': m,
                    'time': self.time_plot.copy(),
                    'h_data': self.h_data.copy(),
                    'h_fit': self.h_fit.copy(),
                    'fit_min': self.fit_min,
                    'fit_max': self.fit_max,
                })

        ring_start = kwargs.get('ring_start', 32)
        fit_start = kwargs.get('fit_start', 0)
        waveform = kwargs.get('waveform', 'h')

        if waveform == 'h':
            re_label = r"$\mathrm{Re}[rh] [M]$"
            im_label = r"$\mathrm{Im}[rh] [M]$"
            abs_label = r"$|rh^{\rm NR}| [M]$"
            res_label = r"$|rh^{\rm NR} - rh^{\rm QNM}| [M]$"
        else:
            re_label = r"$\mathrm{Re}[r\Psi_4] [M^{-1}]$"
            im_label = r"$\mathrm{Im}[r\Psi_4] [M^{-1}]$"
            abs_label = r"$|r\psi_4^{\rm NR}| [M^{-1}]$"
            res_label = r"$|r\psi_4^{\rm NR} - r\psi_4^{\rm QNM}| [M^{-1}]$"

        _, axs = plt.subplots(2, 1)
        plotted_modes = set()
        for r in results:
            mode_key = tuple(r['mode'])
            if mode_key not in plotted_modes:
                axs[0].plot(r['time'], r['h_data'].real, label=f"Data: Mode {r['mode']}")
                axs[1].plot(r['time'], r['h_data'].imag)
                plotted_modes.add(mode_key)
        if fit:
            for r in results:
                axs[0].plot(r['time'], r['h_fit'].real, linestyle='--', label=f"Fit: {r['label']}")
                axs[1].plot(r['time'], r['h_fit'].imag, linestyle='--')
            axs[0].axvline(ring_start + fit_start, color='grey', linestyle=':', label='Start of Fitting')
            axs[1].axvline(ring_start + fit_start, color='grey', linestyle=':')
        axs[0].set_ylabel(re_label, fontsize='large')
        axs[1].set_ylabel(im_label, fontsize='large')
        axs[1].set_xlabel("Time [M]", fontsize='large')
        axs[0].legend(loc='upper right', fontsize='small')
        for ax in axs:
            ax.grid()
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        #plt.show()

        _, axs = plt.subplots(2, 1)
        plotted_modes = set()
        for r in results:
            mode_key = tuple(r['mode'])
            mi, ma = r['fit_min'], r['fit_max']
            t = r['time']
            if mode_key not in plotted_modes:
                axs[0].semilogy(t[mi:ma], np.abs(r['h_data'][mi:ma]), label=f"Data: Mode {r['mode']}")
                plotted_modes.add(mode_key)
            axs[0].semilogy(t[mi:ma], np.abs(r['h_fit'][mi:ma]), linestyle='--', label=f"Fit: {r['label']}")
            axs[1].semilogy(t[mi:ma], np.abs(r['h_data'][mi:ma] - r['h_fit'][mi:ma]), label=f"Residual: {r['label']}")
        axs[0].set_ylabel(abs_label, fontsize='large')
        axs[1].set_ylabel(res_label, fontsize='large')
        axs[1].set_xlabel("Time [M]", fontsize='large')
        for ax in axs:
            ax.grid()
            ax.legend()
        #plt.show()

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

    def mismatch_test1(self, window=[0,50], end=80, models=[{}], **kwargs):
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

    def mismatch_test2(self):
        test_param1 = np.arange(0,80,1)
        test_param2 = np.arange(0,80,1)
        mismatch_axis = np.zeros((len(test_param2), len(test_param1)))
        for i,param1 in enumerate(tqdm.tqdm(test_param1)):
            for k,param2 in enumerate(test_param2):
                self.mismatch(ring_start=param1, fit_start=param2)
                mismatch_axis[k,i] = self.mm.copy()

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

    def colour_plot(self, ring_start=32, fit_length=50, mass_plot=True, **kwargs):
        spin_axis = np.arange(0.675,0.71,0.0002)
        mass_axis = np.arange(0.94,0.97,0.0002)
        mismatch_axis = np.zeros((len(mass_axis), len(spin_axis)))

        for i,spin in enumerate(tqdm.tqdm(spin_axis)):
            for j,mass in enumerate(mass_axis):
                self.mismatch(ring_start=ring_start, fit_length=fit_length, a=spin, mass_bh=mass, **kwargs)
                mismatch_axis[j,i] = self.mm.copy()

        if mass_plot:
            fig, ax = plt.subplots()
            im = ax.imshow(mismatch_axis, norm='log', origin='lower', aspect='auto', extent=[spin_axis.min(), spin_axis.max(),
                mass_axis.min(), mass_axis.max()])
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Mismatch", fontsize='large')
            ax.set_xlabel(r"Dimensionless Spin $\chi$", fontsize='large')
            ax.set_ylabel(r"Black Hole Mass $M_f$ [M]", fontsize='large')
            ax.plot(0.692, 0.952, 'x', color='red', label=r'Literature Values: $\chi$=0.692, $M_{f}$=0.952M')
            ax.plot(0.693,0.954, 'x', color='orange', label=r'Extracted Values: $\chi$=0.693, $M_{f}$=0.954M')
            ax.legend(fontsize=11)
            plt.show()

        min_idx = np.unravel_index(np.argmin(mismatch_axis), mismatch_axis.shape)
        self.best_mass = mass_axis[min_idx[0]]
        self.best_spin = spin_axis[min_idx[1]]
        self.best_mm = mismatch_axis[min_idx[0],min_idx[1]]
        print(f"Minimum mismatch {self.best_mm} at mass={self.best_mass}, spin={self.best_spin}")

    def freq_colour_plot(self, waveform, ring_start, fit_length, a, mass_bh, freq_plot=True, neg_freq=False):
        re_axis = np.arange(1,1.2,0.01) #x-axis
        im_axis = np.arange(0,0.6,0.02) #y-axis
        mismatch_axis = np.zeros((len(im_axis), len(re_axis))) #'heat'

        for i,re in enumerate(tqdm.tqdm(re_axis)):
            for j,im in enumerate(im_axis):
                self.mismatch(waveform=waveform, modes=[[4,4]], n_overtones=1, ring_start=ring_start, fit_length=fit_length, a=a, mass_bh=mass_bh, neg_freq=neg_freq, agn_freq=[re,im])
                mismatch_axis[j,i] = self.mm.copy()

        omega_22, _, _ = qnm.modes_cache(s=-2,l=2,m=2,n=0)(self.a)
        omega_22 = omega_22*self.scale
        omega_44, _, _ = qnm.modes_cache(s=-2,l=4,m=4,n=2)(self.a)
        omega_44 = omega_44*self.scale

        if freq_plot:
            fig, ax = plt.subplots()
            ima = ax.imshow(mismatch_axis, norm='log', origin='lower', aspect='auto', extent=[re_axis.min(), re_axis.max(),
                im_axis.min(), im_axis.max()])
            fig.colorbar(ima, ax=ax, label='Mismatch')
            ax.set_xlabel(r"Re[$M\omega$]", fontsize='large')
            ax.set_ylabel(r"-Im[$M\omega$]", fontsize='large')
            plt.plot(2*omega_22.real, -2*omega_22.imag, 'x', color='red', label='(2,2,0)x(2,2,0)')
            plt.plot(omega_44.real, -omega_44.imag, '*', color='black', label='(4,4,2)')
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
            im_axis = np.arange(0.06,0.1,0.001) #y-axis
            mismatch_axis = np.zeros((len(im_axis), len(re_axis))) #'heat'
    
            for i,re in enumerate(tqdm.tqdm(re_axis)):
                for j,im in enumerate(im_axis):
                    self.mismatch(waveform=waveform, modes=[[2,2]], n_overtones=-1, ring_start=ring_start, fit_length=fit_length, a=a, mass_bh=mass_bh, neg_freq=neg_freq, agn_freq=[re,im])
                    mismatch_axis[j,i] = self.mm.copy()
    
            omega_22, _, _ = qnm.modes_cache(s=-2,l=2,m=2,n=0)(self.a)
            omega_22 = omega_22*self.scale
    
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
            #print(f"Minimum mismatch {self.best_mm} at Re[omega]={self.best_re}, -Im[omega]={self.best_im}")
            #print(f"Fundamental mode (2,2,0): Re[omega]={omega_22.real}, -Im[omega]={-omega_22.imag}")

    def mass_time_drift(self, fit_start_range, fit_end, **kwargs):
        times = []
        best_masses = []
        best_spins = []
        best_mms = []

        for t in tqdm.tqdm(np.arange(fit_start_range[0], fit_start_range[1], 5)):
            self.colour_plot(ring_start=t, fit_length=fit_end-t, mass_plot=False, **kwargs)
            times.append(t)
            best_masses.append(self.best_mass)
            best_spins.append(self.best_spin)
            best_mms.append(self.best_mm)

        fig, ax = plt.subplots()
        sc = ax.scatter(best_spins, best_masses, c=times)
        fig.colorbar(sc, ax=ax, label='Start Times [M]')
        ax.grid()
        ax.set_xlabel(r"Dimensionless Spin $\chi$")
        ax.set_ylabel(r"Black Hole Mass $M_f$ [M]")

        fig2, ax2 = plt.subplots()
        ax2.semilogy(times, best_mms)
        ax2.grid()
        plt.show()

    def freq_time_drift(self, waveform, fit_start_range, fit_end, a, mass_bh, **kwargs):
        times = []
        best_ims = []
        best_res = []
        best_mms = []

        for t in tqdm.tqdm(np.arange(fit_start_range[0], fit_start_range[1], 5)):
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

        omega_22, _, _ = qnm.modes_cache(s=-2,l=2,m=2,n=0)(self.a)
        omega_22 = omega_22*self.scale
        omega_44, _, _ = qnm.modes_cache(s=-2,l=4,m=4,n=2)(self.a)
        omega_44 = omega_44*self.scale
        ax.plot(2*omega_22.real, -2*omega_22.imag, 'x', color='red', label='(2,2,0)x(2,2,0)')
        ax.plot(omega_44.real, -omega_44.imag, '*', color='black', label='(4,4,2)')
        ax.legend()

        fig2, ax2 = plt.subplots()
        ax2.semilogy(times, best_mms)
        ax2.grid()
        ax2.set_xlabel("Start Time [M]", fontsize='large')
        ax2.set_ylabel("Mismatch", fontsize='large')
        plt.show()

if __name__ == "__main__":
    #test = SXSAnalysis(f"SXS:BBH:0305")
    for id in {'1502', '1476', '1506', '1508', '1474', '1505', '1504', '1485', '1486', '1500', '1492', '1465', '1458', '1438', '1430'}:
        test = SXSAnalysis(f"SXS:BBH:{id}")
        print(f"Simulation: {id}")
        test.graphs(waveform='h', modes=[[4,4]], models=[{'n_overtones':1}], 
                    plot_start=0, ring_start=15, fit_length=85, fit=True)
    #test.mismatch(mode=[2,2], n_overtones=0, ring_start=25, fit_length=45, printing=True)
    # test.mismatch_test1(mode=[2,2], models=[{'n_overtones':0, 'neg_freq':True}, {'n_overtones':0, 'retrograde':True}],
    #                     window=[0,50], end=100)
    #test.colour_plot(modes=[[2,2]], n_overtones=1, ring_start=20, fit_length=50)
    #test.freq_colour_plot('psi4', ring_start=20, fit_length=80)
    #test.fund_colour_plot('psi4', ring_start=20, fit_length=50, a=None, mass_bh=None)
    #test.freq_time_drift('psi4', [5,30], 80)
