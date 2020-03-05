import numpy as np
import pandas as pd
import os
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, auc, roc_curve

def conf_matrix(y, y_pred):
    m = confusion_matrix(y, y_pred)
    return m.astype('float')/m.sum(axis=1)[:, np.newaxis]

def metrics(cm):
    VP, FN, FP, VN = cm.ravel()
    names = ["ACU: ","SENS: ","ESP: "]
    print(names[0], 
          np.round(100*(VP+VN)/cm.sum(), 
                   decimals=3))
    print(names[1], 
          np.round((100*VP/(VP+FN)), decimals=3))
    print(names[2], 
          np.round((100*VN/(VN+FP)), decimals=3))

PATH = os.getcwd()
HOME = os.path.expanduser("~")
N = np.load(PATH + "/vars/n_eeg_var.npy", allow_pickle=True)
W = np.load(PATH + "/vars/w_eeg_var.npy", allow_pickle=True)
# N = np.array([ aux[i].cpu() for i in range(len(aux))])

X = np.vstack((N, W))
y = np.hstack((np.zeros(len(N)), np.ones(len(W))))
print(X.shape)
print(y.shape)

knn_ = KNeighborsClassifier(n_neighbors=5)
y_pred = cross_val_predict(knn_, X, y, cv=5)
# y_prob = cross_val_predict(knn_, X, y, cv=10, method='predict_proba')
cm = conf_matrix(y, y_pred)
metrics(cm)

# print(N.shape)
# print(W.shape)
# print(y_pred)

# print(X)
