import numpy as np
import qnm
import h5py
import matplotlib.pyplot as plt

R01_data = h5py.File('BAM_0125/data_R01.h5', 'r')

rh_22 = R01_data['rh_22'] #strain of mode 2,2 at different distances
rpsi4_20 = R01_data['rpsi4_20']
rpsi4_22 = R01_data['rpsi4_22']

#print(list(rpsi4_22.keys()))

rh_22_key0 = list(rh_22.keys())[0]
rpsi4_20_key0 = list(rpsi4_20.keys())[0]
rpsi4_22_key0 = list(rpsi4_22.keys())[0]

Rh_l2_m2_r00700 = rh_22[rh_22_key0][:]
Rpsi4_l2_m0_r00700 = rpsi4_20[rpsi4_20_key0][:]
Rpsi4_l2_m2_r00700 = rpsi4_22[rpsi4_22_key0][:]

#print(np.shape(Rpsi4_l2_m0_r00700))
#print(Rpsi4_l2_m0_r00700[0])

time = []
for i in range(len(Rpsi4_l2_m2_r00700)):
    time.append(Rpsi4_l2_m2_r00700[i][0])

Re_psi4 = []
for i in range(len(Rpsi4_l2_m2_r00700)):
    Re_psi4.append(Rpsi4_l2_m2_r00700[i][4])

#plt.plot(time,Re_psi4)
#plt.show()

