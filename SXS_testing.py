import numpy as np
import sxs
import qnm
import matplotlib.pyplot as plt
import matplotlib

sim = sxs.load("SXS:BBH:0305", download=False)

strain = sim.psi4

h = strain.data[:,4]

#print(sxs.sxs_directory("cache"))
#print(sim.metadata)
#print(strain.metadata)
##print(sim.psi4.time)
#print(sim.psi4.data)

time = strain.time

h_plus = h.real
h_cross = h.imag

print(dir(strain.metadata))
#print(len(h_plus))

plt.plot(time,h_cross, label=r"(2,2) waveform")
plt.xlabel(r"Time [M]", fontsize='large')
plt.ylabel(r"$\mathrm{Re}[r\psi_{4}] [M^{-1}]$", fontsize='large')
plt.grid()
#plt.plot(time[start:end],h_plus[start:end])
#plt.legend()
#plt.show()



