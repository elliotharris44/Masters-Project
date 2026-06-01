import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design

gps_merger = 1420878141.2 #this is GW250114

#extracting data from 15 seconds either side of merger, can input just H1 and L1
data = TimeSeries.get("H1", gps_merger-15, gps_merger+15) 

#cpoied from example code for GW150914
bp = filter_design.bandpass(50, 250, data.sample_rate)

notch_frequencies = [60, 120, 180] #Multiples of AC frequencies in US
notches = [filter_design.notch(f, data.sample_rate) for f in notch_frequencies]

zpk = filter_design.concatenate_zpks(bp, *notches)
hfilt = data.filter(zpk, filtfilt=True)

hdata = data.crop(*data.span.contract(1))
hfilt = hfilt.crop(*hfilt.span.contract(1))

plt.plot(hfilt.times.value-gps_merger, hfilt.value)
plt.show()
