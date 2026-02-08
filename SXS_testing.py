import numpy as np
import sxs
import qnm
import matplotlib.pyplot as plt

sim = sxs.load("SXS:BBH:0305")

strain = sim.strain

h = strain.data[:,4]


#print(sim.metadata)
#print(strain.metadata)
##print(sim.psi4.time)
#print(sim.psi4.data)

time = strain.time

h_plus = h.real
h_cross = h.imag

print(sim.keys())
#plt.plot(time,h_cross)
#plt.show()