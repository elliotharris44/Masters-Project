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

    def selection(self, eos=None, reference_bibkey=None, mass=None, mass_ratio=None, id_type=None):
        """
        Examples:
        eos='SLy', reference_bibkey='Dietrich:2017aum', mass=[2.5,3], mass_ratio=[0.9,1.1], id_type='Irrotational'
        """
        self.sim_id = []
        mass_list = []
        mass_ratio_list = []
        for i in self.idb.index:
            m = i.data
            if ((eos is None or m['id_eos']==eos) and 
                (reference_bibkey is None or reference_bibkey in m['reference_bibkeys']) and 
                (mass is None or mass[0]<=float(m['id_mass'])<=mass[1]) and 
                (mass_ratio is None or mass_ratio[0]<=float(m['id_mass_ratio'])<=mass_ratio[1]) and 
                (id_type is None or m['id_type']==id_type)):
                self.sim_id.append(m['database_key'])
                mass_list.append(float(m['id_mass']))
                mass_ratio_list.append(float(m['id_mass_ratio']))
        #print(self.sim_id)
        #print(len(self.sim_id))
        print(mass_list)
        print(np.mean(mass_list))
        print(mass_ratio_list)
        print(np.mean(mass_ratio_list))

    
    def plot(self, id='BAM:0125', mode='rh_22'):
        path = f"Data_Tests/{id.replace(':', '_')}/R01/data.h5"
        if not os.path.exists(path):
            return
        self.cdb.sync(dbkeys=[id], lfs=True, prot='https') #plot function clones sim
        R01_data = h5py.File(path, 'r')
        series = R01_data[mode]
        keys = list(series.keys())
        series_r = series[keys[-1]][:] #selects largest extraction radius
        signal = series_r.T[1] + 1j*series_r.T[2]
        time = series_r.T[0]
        plt.plot(time,np.real(signal))
        plt.title(f"{id} Re[{mode}]")
        plt.grid()
        plt.show()
    
    def plot_selection(self, eos=None, reference_bibkey=None, mass=None, mass_ratio=None, id_type=None):
        self.selection(eos, reference_bibkey, mass, mass_ratio, id_type)
        for i in self.sim_id:
            self.plot(i)

obj = CoReSelection()
#obj.selection(eos='SLy')
obj.plot('BAM:0130')
#obj.plot_selection(eos='SLy')