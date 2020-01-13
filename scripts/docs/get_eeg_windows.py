import numpy as np
import pandas as pd
import os

path = os.getcwd()

ch = ['EEG C3-REF', 'EEG C4-REF', 'EEG CZ-REF', 'EEG EKG1-REF',
      'EEG F3-REF', 'EEG F4-REF', 'EEG F7-REF', 'EEG F8-REF',
      'EEG FP1-REF', 'EEG FP2-REF', 'EEG FZ-REF', 'EEG O1-REF',
      'EEG O2-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG PZ-REF',
      'EEG T1-REF', 'EEG T2-REF', 'EEG T3-REF', 'EEG T4-REF',
      'EEG T5-REF', 'EEG T6-REF', 'EMG-REF']


mx_n = pd.DataFrame(data=np.load(path+'/data/matriz_sem_crise.npy'), index=ch)
mx_w = pd.DataFrame(data=np.load(path+'/data/matriz_com_crise.npy'), index=ch)

mx_n = mx_n.iloc[:, :mx_w.shape[1]]


mx_n = mx_n.iloc[:, 1:]
mx_w = mx_w.iloc[:, 1:]

np.save('EMG_sem_crise.npy', mx_n.loc['EMG-REF', :].to_numpy())
np.save('EMG_com_crise.npy', mx_w.loc['EMG-REF', :].to_numpy())

np.save('ECG_sem_crise.npy', mx_n.loc['EEG EKG1-REF', :].to_numpy())
np.save('ECG_com_crise.npy', mx_w.loc['EEG EKG1-REF', :].to_numpy())
