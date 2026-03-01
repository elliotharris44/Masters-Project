import numpy as np
from scipy.integrate import simpson

def ringdown_comb(omegas, taus):
    def model(t, *params):
        start = len(t)//2
        sig = np.zeros_like(t)
        N = len(params)//2
        for n in range(N):
            A = params[2*n]
            phi = params[2*n+1]
            sig[:start] += A * np.exp(-t[:start]/taus[n]) * np.cos(omegas[n]*t[:start] + phi)
            t_shifted = t[start:] - t[start] + t[0]
            sig[start:] += - A * np.exp(-t_shifted/taus[n]) * np.sin(omegas[n]*t_shifted + phi) 
        return sig
    return model

def ringdown_real(omegas, taus):
    def model(t, *params):
        sig = np.zeros_like(t)
        N = len(params)//2
        for n in range(N):
            A = params[2*n]
            phi = params[2*n+1]
            sig += A * np.exp(-t/taus[n]) * np.cos(omegas[n]*t + phi)
        return sig
    return model

def ringdown_imag(omegas, taus):
    def model(t, *params):
        sig = np.zeros_like(t)
        N = len(params)//2
        for n in range(N):
            A = params[2*n]
            phi = params[2*n+1]
            sig += -A * np.exp(-t/taus[n]) * np.sin(omegas[n]*t + phi)
        return sig
    return model

def mismatch_function(time, data, fit):
    num = simpson(data*np.conj(fit), time)
    den = np.sqrt(simpson(data*np.conj(data), time)*simpson(fit*np.conj(fit), time))
    return 1-np.abs(num/den)