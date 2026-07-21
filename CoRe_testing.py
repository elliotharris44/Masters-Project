import numpy as np
import h5py
import qnm
import matplotlib.pyplot as plt
from watpy.coredb.coredb import *
import os
from scipy.integrate import cumulative_trapezoid

class CoReSelection:
    def __init__(self, load=True):
        if load:
            self.cdb = CoRe_db('./Data_Tests/') #clones files here
            self.idb = self.cdb.idb #idb.index gives list of all simulations, .data[] of these gives metadata

    def metadata(self, id='BAM:0125'):
        for i in self.idb.index: #checking metadata
            if i.data['database_key'] == id:
                for j, k in i.data.items():
                    print(f"{j} = {k}")
    
    def selection(self, eos=None, reference_bibkey=None, mass=None, mass_ratio=None, id_type=None, binary_type=None, sync=False, printing=False):
        """
        Examples:
        eos='SLy', reference_bibkey='Dietrich:2017aum', mass=[2.5,3], mass_ratio=[0.9,1.1], id_type='Irrotational'
        """
        self.sim_id = []
        bibkeys = []
        mass_list = []
        mass_ratio_list = []
        eos_list = []
        binary_type_list = []
        self.sim_list = []
        for i in self.idb.index:
            m = i.data
            if ((eos is None or m['id_eos']==eos) and 
                (reference_bibkey is None or reference_bibkey in m['reference_bibkeys']) and 
                (mass is None or mass[0]<=float(m['id_mass'])<=mass[1]) and 
                (mass_ratio is None or mass_ratio[0]<=float(m['id_mass_ratio'])<=mass_ratio[1]) and 
                (id_type is None or m['id_type']==id_type) and (binary_type is None or m['binary_type']==binary_type)):
                self.sim_id.append(m['database_key'])
                self.sim_list.append([m['database_key'], float(m['id_mass']), float(m['id_mass_ratio']), m['id_eos']])
                mass_list.append(float(m['id_mass']))
                mass_ratio_list.append(float(m['id_mass_ratio']))
                eos_list.append(m['id_eos'])

                # path = f"Data_Tests/{m['database_key'].replace(':', '_')}/R01/data.h5"
                # try:
                #     R01_data = h5py.File(path, 'r')
                # except OSError:
                #     print(f"Skipping {id} - file could not be opened (possibly corrupted)")
                #     break

                if m['reference_bibkeys'] not in bibkeys:
                    # if 'rpsi4_44' in R01_data:
                    bibkeys.append(m['reference_bibkeys'])
                
                if m['binary_type'] not in binary_type_list:
                    binary_type_list.append(m['binary_type'])
        if sync:
            self.cdb.sync(dbkeys=self.sim_id, lfs=True, prot='https')
        if printing:
            #print(self.sim_id)
            # print(bibkeys)
            # print(len(self.sim_id))
            #print(mass_list)
            # print(np.mean(mass_list))
            #print(mass_ratio_list)
            # print(np.mean(mass_ratio_list))
            #print(eos_list)
            #print(binary_type_list)
            print(self.sim_list)
    
    def plot(self, id='BAM:0125', show=True, mode='rpsi4_22', rad=-1, ax1=None, ax2=None):
        if ax1 is None or ax2 is None:
            _, ax1 = plt.subplots()
            _, ax2 = plt.subplots()

        path = f"Data_Tests/{id.replace(':', '_')}/R01/data.h5"
        try:
            R01_data = h5py.File(path, 'r')
        except OSError:
            print(f"Skipping {id} - file could not be opened (possibly corrupted)")
            return
        try:
            if mode not in R01_data:
                print(f"Skipping {id} - mode '{mode}' not found")
                return
        except RuntimeError:
            print(f"Skipping {id} - file corrupted (bad HDF5 structure)")
            return
        series = R01_data[mode]
        keys = [k for k in series.keys() if series[k].shape[0] > 0 and k.split('r')[-1].split('.')[0].lstrip('-').isdigit()]
        if not keys:
            print(f"Skipping {id} - no valid extraction radii found")
            return
        #print(keys)
        series_r = series[keys[rad]][:]
        signal = series_r.T[1] + 1j*series_r.T[2]
        time = series_r.T[0]
        
        radius = int(keys[rad].split('r')[-1].split('.')[0])
        if rad == -1:
            coeff = (2 - 1) * (2 + 2) / (2.0 * radius)
            integral = cumulative_trapezoid(signal, time, initial=0)
            signal2 = signal - coeff*integral
            # ax1.plot(time, np.real(signal2), label=f"Extraction radius {radius}, correction")
            # ax2.semilogy(time, np.abs(signal2), label=f"Extraction radius {radius}, correction")

        try:
            series_2 = h5py.File(f"Data_Tests/{id.replace(':', '_')}/R02/data.h5", 'r')['rpsi4_22'] #change between resolutions and modes
        except OSError:
            print(f"Skipping {id} - file could not be opened (possibly corrupted)")
            return
        keys2 = [k for k in series_2.keys() if series_2[k].shape[0] > 0 and k.split('r')[-1].split('.')[0].lstrip('-').isdigit()]
        try:
            series_r_2 = series_2[keys2[rad]][:]
        except OSError:
            print(f"Skipping {id} - file could not be opened (possibly corrupted)")
            return
        signal_2 = series_r_2.T[1] + 1j*series_r_2.T[2]
        time_2 = series_r_2.T[0]
        signal.real = signal.real/np.max(signal.real)
        signal_2.real = signal_2.real/np.max(signal_2.real)
        ax1.plot(time_2, np.real(signal_2), label=r"$R02$ (2,2) waveform")
        ax1.plot(time, np.real(signal), label=r"$R01$ (2,2) waveform")

        ax1.set_xlabel(r"Time [M]", fontsize='large')
        ax1.set_ylabel(r"$\mathrm{Re}[r\psi_{4}] [M^{-1}]$", fontsize='large')
        #ax1.plot(time, np.real(signal), label=f"Extraction radius {radius}, no correction")
        ax2.semilogy(time, np.abs(signal), label=f"Extraction radius {radius}, no correction")
        #ax1.set_title(f"{id} Re[{mode}]")
        ax1.grid()
        ax2.grid()

        if show:
            ax1.legend()
            ax2.legend()
            plt.show()

    def plot_extradius(self, id='BAM:0125', mode='rpsi4_22', len=5):
        _, ax1 = plt.subplots()
        _, ax2 = plt.subplots()
        for i in range(1, len):
            self.plot(id=id, mode=mode, show=False, rad=-i, ax1=ax1, ax2=ax2)
        ax1.legend()
        ax2.legend()
        plt.show()
    
    def plot_selection(self, eos=None, reference_bibkey=None, mass=None, mass_ratio=None, id_type=None):
        self.selection(eos, reference_bibkey, mass, mass_ratio, id_type)
        
        local_dirs = [
            d.replace('_', ':', 1)
            for d in os.listdir('Data_Tests')
            if d.startswith('BAM_') and os.path.isdir(f'Data_Tests/{d}')
            and 176 <= int(d.split('_')[1]) <= 226
        ] #used claude code
        all_ids = list(self.sim_id) + [i for i in local_dirs if i not in self.sim_id]
        for i in all_ids:
            self.plot(i)


def plot_log():
    data = np.loadtxt("Runs/fit_output1.txt")
    pos = np.loadtxt("Runs/fit_output2.txt")
    t = data[:,0]
    signal = data[:,1] + 1j*data[:,2]
    fit = data[:,3] + 1j*data[:,4]
    fit_pos = pos[:,3] + 1j*pos[:,4]
    plt.semilogy(t, np.abs(signal), label='Data')
    plt.semilogy(t, np.abs(fit), label='Fit (Positive + Negative Frequencies)')
    plt.semilogy(t, np.abs(fit_pos), label='Fit (Positive Frequency)')
    plt.xlabel("Time (M)", fontsize='large')
    plt.ylabel(r"$\log|\psi_{4}|$", fontsize='large')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()

obj = CoReSelection(load=False)
#obj.selection(eos='DD2', mass=[3.29,3.33], mass_ratio=[0.99,1.01], printing=True)
obj.plot('BAM:0012', mode='rpsi4_22', rad=-1)
#obj.plot_selection()
#plot_log()
#obj.plot_extradius(id="BAM:0138", len=7)

