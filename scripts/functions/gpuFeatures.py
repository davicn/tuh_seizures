import torch
import numpy as np
from numba import njit

device = torch.device("cuda:0")


def energy(x):
    x = torch.tensor(x, device=device)
    return x.pow(2).abs().sum().item()

@njit
def mpp(x):
    return np.array([np.mean(x[:, i]) for i in range(len(x[0]))])


def variancia(s, j, eeg):
    m = np.zeros((2, len(j)-1))
    for i in range(len(m)):
        m[i] = np.array([np.var(s[i, j[ii]:j[ii+1]])
                         for ii in range(len(j)-1)])
    return m


def variancia(sig, t, fs, eeg):
    interval = int(t*fs)

    if eeg == False:
        sig = torch.tensor(sig, device=device)
        aux = torch.empty(sig.shape[0]//interval, device=device)
        for i in range(aux.shape[0]-1):
            aux[i] = sig[i*interval:(i+1)*interval].var()
    else:
        # , device=device)
        aux = torch.empty(sig.shape[0], sig.shape[1]//interval)
        for ii in range(len(aux)):
            for i in range(aux.shape[1]-1):
                aux[ii] = np.var(sig[i*interval:(i+1)*interval])
    return mpp(aux.numpy())


def energia(sig, t, fs, eeg):
    interval = int(t*fs)

    if eeg == False:
        sig = torch.tensor(sig, device=device)
        aux = torch.empty(sig.shape[0]//interval, device=device)
        for i in range(aux.shape[0]-1):
            aux[i] = energy(sig[i*interval:(i+1)*interval])
    else:
        # , device=device)
        aux = torch.empty(sig.shape[0], sig.shape[1]//interval)
        for ii in range(len(aux)):
            for i in range(aux.shape[1]-1):
                aux[ii] = energy(sig[i*interval:(i+1)*interval])
    return mpp(aux.numpy())
    # return aux
