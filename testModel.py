# import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras, initializers
from keras import Model
from keras import layers
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split

print('\nLoading Data...')
vm_norm_T= np.load('vm_norm_T.npy')
print('\nvm_norm_T: ', vm_norm_T.shape)
ecgNorm= np.load('ecg_norm.npy')
print('ecgNorm: ', ecgNorm.shape)

vm_norm_T_70= np.load('vm_norm_T_70.npy')
print('vm_norm_T_70: ', vm_norm_T_70.shape)
ecgNorm_70= np.load('ecgNorm_70.npy')
print('ecgNorm_70: ', ecgNorm_70.shape)

# print(vm_norm_T[0,0,:], '\n')
# print(ecgNorm[0,0,:])
# print(vm_norm_T[0,0,:], '\n')
# print(ecgNorm[0,0,:])
# print(ecgNorm_test[0,0,:])
# print(volt_test_T[0,0,:])
#------------------------------------------------------------------------Predict----------------------------------------------------------------|

ecgNorm_train, ecgNorm_test, volt_train_T, volt_test_T= train_test_split(ecgNorm, vm_norm_T, test_size= 0.05, random_state=42)
print('\necg_train, ecg_test, vm_train, vm_test: ', ecgNorm_train.shape, ecgNorm_test.shape, volt_train_T.shape, volt_test_T.shape)

ecg2vm_mdl= keras.models.load_model('ecg2vm_mdl_e82.h5')
ecg2vm_mdl.summary()

[X1, X2] = [ecgNorm_test, volt_test_T]
ix = np.random.randint(0, len(X1), 1)
src_ecg, tar_vm = X1[ix], X2[ix]

# generate image from source
gen_volt = ecg2vm_mdl.predict(src_ecg)
print('results: ', gen_volt.shape)
print(gen_volt, '\n')

print(src_ecg.shape,'--->', tar_vm.shape,':==   ', gen_volt.shape)
plt.figure(figsize = (20//2, 10//2))
plt.plot(range(500), gen_volt[0])
plt.show()

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('ECG')
plt.imshow(src_ecg[0])

plt.subplot(232)
plt.title('Actual Volt')
plt.imshow(tar_vm[0])

plt.subplot(233)
plt.title('Predicted Volt')
plt.imshow(gen_volt[0])
plt.show()

print(gen_volt)