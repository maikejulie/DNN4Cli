# script to calculate SHAP values

import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import tensorflow as tf


monthlySSH = xr.open_mfdataset('/content/gdrive/MyDrive/SSHdata/SSH.*.nc', combine='by_coords')
SSH20mean = monthlySSH['SSH'].mean(axis=0).values
Bathm = np.transpose(loadmat('/content/gdrive/MyDrive/DNN4Cli/THOR/Step2/Data_Step2/H_wHFacC.mat')['val'])
curlTau = np.transpose(np.load('/content/gdrive/MyDrive/DNN4Cli/THOR/Step2/Data_Step2/curlTau.npy'))
ecco_label = np.transpose(np.load('/content/gdrive/MyDrive/DNN4Cli/THOR/Step2/Data_Step2/kCluster6.npy'))
ecco_label[ecco_label==-1] = np.nan

lonRoll = np.roll(monthlySSH['lat'].values, axis=0, shift=-1)
Londiff = lonRoll - monthlySSH['lat'].values  # equivalent to doing x_{i} - x_{i-1}

lat = monthlySSH['lat'].values
latDiff=1.111774765625000e+05 # what is this value?
latY=np.gradient(lat, axis=0)*latDiff
lonX=np.abs(np.cos(lat*np.pi/180))*latDiff*Londiff

Omega=7.2921e-5 # coriolis parameter (slightly different to wiki)
f = (2*Omega*np.sin(lat*np.pi/180))

def grad(d,y,x):
    grady=np.gradient(d, axis=0)/y
    gradx=np.gradient(d, axis=1)/x
    return grady, gradx

gradSSH_y, gradSSH_x = grad(SSH20mean,latY,lonX)
gradBathm_y, gradBathm_x = grad(Bathm,latY,lonX)

missingdataindex = np.isnan(curlTau*SSH20mean*gradSSH_x*gradSSH_y*Bathm*gradBathm_x*gradBathm_y)

maskTraining = (~missingdataindex).copy()
maskTraining[:,200:400]=False

maskVal = (~missingdataindex).copy()
maskVal[:,list(range(200))+list(range(400,720))]=False

TotalDataset = np.stack((curlTau[~missingdataindex],
                         SSH20mean[~missingdataindex],
                         gradSSH_x[~missingdataindex],
                         gradSSH_y[~missingdataindex],
                         Bathm[~missingdataindex],
                         gradBathm_x[~missingdataindex],
                         gradBathm_y[~missingdataindex],
                         f[~missingdataindex]),1)

TrainDataset = np.stack((curlTau[maskTraining],
                         SSH20mean[maskTraining],
                         gradSSH_x[maskTraining],
                         gradSSH_y[maskTraining],
                         Bathm[maskTraining],
                         gradBathm_x[maskTraining],
                         gradBathm_y[maskTraining],
                         f[maskTraining]),1)

ValDataset = np.stack((curlTau[maskVal],
                         SSH20mean[maskVal],
                         gradSSH_x[maskVal],
                         gradSSH_y[maskVal],
                         Bathm[maskVal],
                         gradBathm_x[maskVal],
                         gradBathm_y[maskVal],
                       f[maskVal]),1)

TotalDataset.shape, TrainDataset.shape, ValDataset.shape

train_label = ecco_label[maskTraining]
val_label = ecco_label[maskVal]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(TrainDataset)
scaler.mean_,scaler.scale_

X_train_scaled = scaler.transform(TrainDataset)
X_val_scaled = scaler.transform(ValDataset)

Y_train = tf.keras.utils.to_categorical(train_label)
Y_val = tf.keras.utils.to_categorical(val_label)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import shap

shap_values_list = []

for i in range(100):
    det_model = Sequential([Dense(input_shape = (8,), units = 24, activation = tf.keras.activations.tanh),
      Dense(units = 24, activation = tf.keras.activations.tanh),
      Dense(units = 16, activation = tf.keras.activations.tanh),
      Dense(units = 16, activation = tf.keras.activations.tanh),
      Dense(units = 6, activation = tf.keras.activations.softmax)])

    det_model.load_weights('/content/gdrive/MyDrive/DNN4Cli/THOR/bayesian_ensemble/model_' + str(i) + '.h5')
    det_model_def = tf.keras.models.Model(
        inputs=det_model.inputs, outputs=det_model.outputs, name=det_model.name
    )

    explainer = shap.Explainer(det_model_def, X_train_scaled, algorithm = "exact", feature_names = ['curlTau', 'SSH20mean', 
                                                               'gradSSHx', 'gradSSHy', 
                                                               'bathm', 'gradBathm_x', 
                                                               'gradBathm_y', 'F'])
    shap_values_list.append(explainer(X_val_scaled))

    shap_ens_values = np.array([shap_values_list[i].values for i in range(len(shap_values_list))])
    shap_ens_base_values = np.array([shap_values_list[i].base_values for i in range(len(shap_values_list))])
    shap_ens_data = np.array([shap_values_list[i].data for i in range(len(shap_values_list))])

    np.save('shap_values_1.npy', shap_ens_values)
    np.save('shap_base_1.npy', shap_ens_base_values)
    np.save('shap_data_1.npy', shap_ens_data)
