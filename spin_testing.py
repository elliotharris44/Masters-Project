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

mass_ratio = np.linspace(0.2,1,500)
change = []
for j,mass in enumerate(mass_ratio):
    a = a_vals[np.abs(freq-0.6*mass).argmin()] #0.628 from BAM:0125 data
    tau_a = tau[np.abs(freq-0.6*mass).argmin()]
    change.append(np.abs((7-tau_a*mass)*100/7)) #percentage difference, 6.65 from data
print(mass_ratio[np.array(change).argmin()])
plt.plot(mass_ratio, change)
plt.show()