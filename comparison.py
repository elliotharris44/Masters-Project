import numpy as np
import matplotlib.pyplot as plt
from CoRe_ringdown import *
from SXS_ringdown import *
import scipy

nsns = CoReAnalysis("BAM_0125")
bhbh = SXSAnalysis("SXS:BBH:0389")

ns_adm = nsns.mass_total
bh_adm = bhbh.mass_total #total initial adm
m_solar = 1.98847e30
G = 6.67430e-11
c = 299792458

nsns.graph(waveform='22', plot_start=64, ring_start=64, fit_length=50, a=0.798, mass_bh=2.911, plot=False)
time_ns = nsns.time_plot
time_ns -= time_ns[0]
signal_ns = nsns.signal_plot
peaks_ns, _ = scipy.signal.find_peaks(signal_ns.real)
peak_idx_ns = peaks_ns[0]
time_ns = time_ns[peak_idx_ns:] - time_ns[peak_idx_ns]
signal_ns = signal_ns[peak_idx_ns:]
time_ns = time_ns*ns_adm*m_solar*G*1000/(c**3) #time in ms
scaled_signal_ns = signal_ns.real/signal_ns.real[0]

bhbh.graphs(waveform='psi4', plot_start=50, ring_start=50, fit_length=50, a=0.687, mass_bh=0.947, plot=False) #different ring_start, potential problem
time_bh = bhbh.time_plot
time_bh -= time_bh[0]
signal_bh = bhbh.total_signal
peaks_bh, _ = scipy.signal.find_peaks(signal_bh.real)
peak_idx_bh = peaks_bh[0]
time_bh = time_bh[peak_idx_bh:] - time_bh[peak_idx_bh]
signal_bh = signal_bh[peak_idx_bh:]
time_bh = time_bh*ns_adm*m_solar*G*1000/(bh_adm*c**3) #time in ms, scaled to have same total mass as ns
scaled_signal_bh = signal_bh.real/signal_bh.real[0]

plt.plot(time_ns, scaled_signal_ns, label="NS-NS Merger")
plt.plot(time_bh, scaled_signal_bh, label="BH-BH Merger")
plt.title("Real Part of Ringdown")
plt.xlabel("Time (ms)")
plt.ylabel("Scaled amplitude")
plt.legend()
plt.grid()
plt.show()