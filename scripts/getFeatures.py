import numpy as np
import pandas as pd
import mne
import os
from functions import gpuFeatures as gf

PATH = os.getcwd()
HOME = os.path.expanduser("~")

w = pd.read_csv(PATH+'/docs/class_w.csv', header=None)
n = pd.read_csv(PATH+'/docs/class_n.csv', header=None)

use = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
       'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
       'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
       'EEG T6-REF', 'EEG T1-REF', 'EEG T2-REF', 'EEG FZ-REF', 'EEG CZ-REF',
       'EEG PZ-REF']

t = .25
fs = 256

w = np.unique(w.iloc[:,0].to_numpy())
n = np.unique(n.iloc[:,0].to_numpy())

# eegs = []
w_ecgs_var = []
w_emgs_var = []
w_ecgs_eng = []
w_emgs_eng = []

n_ecgs_var = []
n_emgs_var = []
n_ecgs_eng = []
n_emgs_eng = []

for i in range(len(w)):
    raw = mne.io.read_raw_edf(
        HOME + '/Documentos/tuh_edf/' + w[i].replace('tse', 'edf'), preload=False, verbose=False)

    d = raw.to_data_frame().T
    # eeg = d.loc[use, :].to_numpy()
    ecg = d.loc['EEG EKG1-REF', :].to_numpy()
    emg = d.loc['EMG-REF', :].to_numpy()

    # eeg_var = gf.variancia(eeg, t, fs, True)
    ecg_var = gf.variancia(ecg, t, fs, False)
    emg_var = gf.variancia(ecg, t, fs, False)
    ecg_eng = gf.energia(ecg, t, fs, False)
    emg_eng = gf.energia(ecg, t, fs, False)

    # print(eeg_var)
    w_ecgs_var.append(ecg_var)
    w_emgs_var.append(emg_var)
    w_ecgs_eng.append(ecg_eng)
    w_emgs_eng.append(emg_eng)

    print(w[i])

np.save("w_ecg_var.npy",w_ecgs_var)
np.save("w_emg_var.npy",w_emgs_var)
np.save("w_ecg_eng.npy",w_ecgs_eng)
np.save("w_emg_eng.npy",w_emgs_eng)


for i in range(len(n)):
    raw = mne.io.read_raw_edf(
        HOME + '/Documentos/tuh_edf/' + n[i].replace('tse', 'edf'), preload=False, verbose=False)

    d = raw.to_data_frame().T
    # eeg = d.loc[use, :].to_numpy()
    ecg = d.loc['EEG EKG1-REF', :].to_numpy()
    emg = d.loc['EMG-REF', :].to_numpy()

    # eeg_var = gf.variancia(eeg, t, fs, True)
    ecg_var = gf.variancia(ecg, t, fs, False)
    emg_var = gf.variancia(ecg, t, fs, False)
    ecg_eng = gf.energia(ecg, t, fs, False)
    emg_eng = gf.energia(ecg, t, fs, False)

    # print(eeg_var)
    n_ecgs_var.append(ecg_var)
    n_emgs_var.append(emg_var)
    n_ecgs_eng.append(ecg_eng)
    n_emgs_eng.append(emg_eng)

    print(n[i])

np.save("n_ecg_var.npy",n_ecgs_var)
np.save("n_emg_var.npy",n_emgs_var)
np.save("n_ecg_eng.npy",n_ecgs_eng)
np.save("n_emg_eng.npy",n_emgs_eng)


# np.save('tamanhos2.npy',np.array(lens))

# w = np.load(PATH + '/data/w_train.npy', allow_pickle=True)
# w = pd.DataFrame(w)
# w.columns = ['path', 'start', 'end', 'type', 'fs', 'duration', 'montage']

# n = np.load(PATH + '/data/n_train.npy', allow_pickle=True)
# n = pd.DataFrame(n)
# n.columns = ['path', 'fs', 'duration', 'montage']
