import numpy as np

def ringdown_comb(omegas, taus):
    def model(t, *params):
        start = len(t)//2
        sig = np.zeros_like(t)
        N = len(params)//2
        for n in range(N):
            A = params[2*n]
            phi = params[2*n+1]
            sig[:start] += A * np.exp(-t[:start]/taus[n]) * np.cos(omegas[n]*t[:start] + phi)
            sig[start:] += -A * np.exp(-t[:start]/taus[n]) * np.sin(omegas[n]*t[:start] + phi)
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
