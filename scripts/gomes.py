import numpy  as np 
import pandas as pd 

col = ['path','freq','duração','montage']      

train = pd.DataFrame(data=np.load('info_train.npy'),columns=col)

print(train.head())