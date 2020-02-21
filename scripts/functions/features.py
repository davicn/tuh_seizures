import numpy as np
from scipy.stats import skew, kurtosis
from numba import jit

fs = 256
jw = fs//.25


@jit(nopython=True)
def energy(x):
    return np.sum(np.abs(x)**2)

# Parâmetro EEG é booleano
# retorna ou não a média ponto a ponto


@jit(nopython=True)
def mpp(x):
    return np.array([np.mean(x[:, i]) for i in range(len(x[0]))])


def curtose(s, j, eeg):
    m = np.zeros((2, len(j)-1))
    for i in range(len(m)):
        m[i] = np.array([kurtosis(s[i, j[ii]:j[ii+1]])
                         for ii in range(len(j)-1)])
    return m


def assimetria(s, j, eeg):
    m = np.zeros((2, len(j)-1))
    for i in range(len(m)):
        m[i] = np.array([skew(s[i, j[ii]:j[ii+1]]) for ii in range(len(j)-1)])
    return m


def variancia(s, j, eeg):
    m = np.zeros((2, len(j)-1))
    for i in range(len(m)):
        m[i] = np.array([np.var(s[i, j[ii]:j[ii+1]])
                         for ii in range(len(j)-1)])
    return m


def energia(s, j, eeg):
    m = np.zeros((2, len(j)-1))
    for i in range(len(m)):
        m[i] = np.array([energy(s[i, j[ii]:j[ii+1]])
                         for ii in range(len(j)-1)])
    return m


def janela(jfs, s): return np.array([i*jfs for i in range(s//jfs+1)])
