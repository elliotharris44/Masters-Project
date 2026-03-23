import numpy as np
import h5py
import qnm
import matplotlib.pyplot as plt
from watpy.coredb.coredb import *
import os

class CoReSelection:
    def __init__(self):
        self.cdb = CoRe_db('./Data_Tests/') #clones files here
        self.idb = self.cdb.idb #idb.index gives list of all simulations, .data[] of these gives metadata

    def metadata(self, id='BAM:0125'):
        for i in self.idb.index: #checking metadata
            if i.data['database_key'] == id:
                for j, k in i.data.items():
                    print(f"{j} = {k}")

    def selection(self, eos=None, reference_bibkey=None, mass=None, mass_ratio=None, id_type=None, sync=False, printing=False):
        """
        Examples:
        eos='SLy', reference_bibkey='Dietrich:2017aum', mass=[2.5,3], mass_ratio=[0.9,1.1], id_type='Irrotational'
        """
        self.sim_id = []
        # bibkeys = []
        # mass_list = []
        # mass_ratio_list = []
        # eos_list = []
        for i in self.idb.index:
            m = i.data
            if ((eos is None or m['id_eos']==eos) and 
                (reference_bibkey is None or reference_bibkey in m['reference_bibkeys']) and 
                (mass is None or mass[0]<=float(m['id_mass'])<=mass[1]) and 
                (mass_ratio is None or mass_ratio[0]<=float(m['id_mass_ratio'])<=mass_ratio[1]) and 
                (id_type is None or m['id_type']==id_type)):
                self.sim_id.append(m['database_key'])
                # mass_list.append(float(m['id_mass']))
                # mass_ratio_list.append(float(m['id_mass_ratio']))
                # if m['reference_bibkeys'] not in bibkeys:
                #     bibkeys.append(m['reference_bibkeys'])
        if sync:
            self.cdb.sync(dbkeys=self.sim_id, lfs=True, prot='https')
        if printing:
            print(self.sim_id)
            # print(bibkeys)
            # print(len(self.sim_id))
            # print(mass_list)
            # print(np.mean(mass_list))
            # print(mass_ratio_list)
            # print(np.mean(mass_ratio_list))
    
    def plot(self, id='BAM:0125', mode='rpsi4_22'):
        path = f"Data_Tests/{id.replace(':', '_')}/R02/data.h5"
        if not os.path.exists(path):
            print(f"Skipping {id} - not downloaded")
            return
        R01_data = h5py.File(path, 'r')
        series = R01_data[mode]
        keys = list(series.keys())
        series_r = series[keys[-1]][:] #selects largest extraction radius
        signal = series_r.T[1] + 1j*series_r.T[2]
        time = series_r.T[0]
        print(len(time))
        plt.plot(time[600:1000],np.real(signal)[600:1000])
        plt.title(f"{id} Re[{mode}]")
        plt.grid()
        plt.show()
    
    def plot_selection(self, eos=None, reference_bibkey=None, mass=None, mass_ratio=None, id_type=None):
        self.selection(eos, reference_bibkey, mass, mass_ratio, id_type)
        for i in self.sim_id:
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

#obj = CoReSelection()
#obj.selection(printing=True)
#obj.plot('BAM:0125')
#obj.plot_selection(eos='SLy')
plot_log()
