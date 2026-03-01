import numpy as np
import sxs
import qnm
import matplotlib.pyplot as plt
import matplotlib

sim = sxs.load("SXS:NSNS:0003")

strain = sim.strain

h = strain.data[:,4]


#print(sim.metadata)
#print(strain.metadata)
##print(sim.psi4.time)
#print(sim.psi4.data)

time = strain.time

h_plus = h.real
h_cross = h.imag

#print(dir(strain.data))
#print(len(h_plus))
start = 5900
end = 8000
plt.plot(time[start:end],h_cross[start:end])
#plt.plot(time[start:end],h_plus[start:end])
plt.show()

