import torch
import numpy as np

device = torch.device("cuda:0")


def energy(x):
    return x.pow(2).abs().sum().item()


def mpp(x):
    return np.array([x[:, i]].mean().item() for i in range(len(x[0])))


def variancia(s, j, eeg):
    m = np.zeros((2, len(j)-1))
    for i in range(len(m)):
        m[i] = np.array([np.var(s[i, j[ii]:j[ii+1]])
                         for ii in range(len(j)-1)])
    return m


def variancia(sig, t, fs, eeg):
    interval = int(t*fs)

    sig = torch.tensor(sig, device=device)
    if eeg == False:
        aux = torch.empty(sig.shape[0]//interval, device=device)
        for i in range(aux.shape[0]-1):
            aux[i] = sig[i*interval:(i+1)*interval].var()
    else:
        aux = torch.empty(sig.shape[0], sig.shape[1]//interval,device=device)
        for ii in range(len(aux)):
            for i in range(aux.shape[1]-1):
                aux[ii] = sig[i*interval:(i+1)*interval].var()
    return aux

def energia(sig, t, fs, eeg):
    interval = int(t*fs)

    sig = torch.tensor(sig, device=device)
    if eeg == False:
        aux = torch.empty(sig.shape[0]//interval, device=device)
        for i in range(aux.shape[0]-1):
            aux[i] = energy(sig[i*interval:(i+1)*interval])
    else:
        aux = torch.empty(sig.shape[0], sig.shape[1]//interval,device=device)
        for ii in range(len(aux)):
            for i in range(aux.shape[1]-1):
                aux[ii] = energy(sig[i*interval:(i+1)*interval])
    return aux
    # return aux
