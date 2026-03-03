import numpy as np
import matplotlib.pyplot as plt
import qnm
from scipy.interpolate import griddata

a_vals = np.linspace(0.01, 0.999, 500)

freq = np.zeros_like(a_vals)
tau = np.zeros_like(a_vals)

grav_220 = qnm.modes_cache(s=-2, l=2, m=2, n=0)

for i,a in enumerate(a_vals):
    omega, _, _ = grav_220(a)
    freq[i] = np.real(omega)
    tau[i] = -1 / np.imag(omega)

# plt.plot(freq,tau,'+')
# plt.xlabel('Frequency')
# plt.ylabel('Decay Constant')
# plt.show()

mismatch = np.zeros_like(a_vals)
Mf_vals  = np.zeros_like(a_vals)
tau_data = 12.79
freq_data = 0.59

for i in range(len(a_vals)):
    Mf_from_tau  = tau_data/tau[i]
    Mf_from_freq = freq[i]/freq_data
    Mf_vals[i] = 0.5 * (Mf_from_tau + Mf_from_freq)
    mismatch[i] = abs(Mf_from_tau - Mf_from_freq)

best_index = np.argmin(mismatch)
best_a  = a_vals[best_index]
best_Mf = Mf_vals[best_index]

print(best_a)
print(best_Mf)