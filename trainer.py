# import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras, initializers
from keras import Model
from keras import layers
from keras.models import Sequential, load_model, save_model
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, Input, Conv2D, Conv1D, Concatenate, \
     MaxPool2D, MaxPool1D, GlobalAvgPool2D, Activation, Dropout, PReLU,BatchNormalization, Flatten,UpSampling2D, LeakyReLU, Dense, Input, add
from keras.losses import MeanSquaredError
from keras.applications import VGG19
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from fxModules import *


generateNewModel= False
loadData= False
splitData=False
loadModel= False
trainModel= False
currentModel= 'transVolt_v1_e11.h5'
predict= False


if(loadData):
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

#------------------------------------------------------------Build Model----------------------------------------------------------------------------|
#[+:]-- The first two dimensions of the shape parameter specify the size of the image, and the third dimension specifies the number of color channels (3 for RGB, 1 for greyscale).
# ecg2vm_mdl= build_ecg2vm(ecg2vm_fx, finalConv_fx, ecg_ip, vm_ip )
# ecg2vm_mdl.compile(loss="mse", optimizer="adam")
# ecg2vm_mdl.save("ecg2vm_mdl_base.h5", ecg2vm_mdl)
# ecg2vm_mdl_base = keras.models.load_model('ecg2vm_mdl_base.h5')
# ecg2vm_mdl_base.summary()
# ecg2vm_mdl.summary()

def transVoltGenerator_new(batch_size=12):
    ecg_ip= Input(shape= (500, 12, 1), batch_size=batch_size)
    # vm_ip= Input(shape= (500, 75, 1), batch_size=batch_size)
    print('\necg_ip shape: (batch_size, timeStep, signals, channels) = ',ecg_ip.shape)
    mdl= transVolt_build(ip= ecg_ip, div=5 )
    mdl.compile(loss="mse", optimizer="adam")
    mdl.summary()
    return mdl

#----------------------------------------------------------------Train Model-------------------------------------------------------------------------|

def trainModel( X,
                Y,
                mdl,
                mdlName= 'mdl_',
                num_epochs=5,
                printerval=2,
                saveInterval= 1,
                batchSize= 12):
    for e in range(num_epochs):
        for b in tqdm(range(X.shape[0]//batchSize)):
            beginDex= b*batchSize
            enDex= beginDex+batchSize
            xB= X[beginDex:enDex]
            yB= Y[beginDex:enDex]
            loss= mdl.train_on_batch(xB,yB)
            if (b+0)%printerval==0: print('    [+] -- ecg batches made: ', b, 'loss: ', loss)
        if (e+1)%saveInterval==0: 
          keras.models.save_model(mdl,'/home/koogleblitz/Cardiac-Electrocardiography/notebooks/'+ mdlName+ "_e"+str(e+1) +".h5")
  
        # if (e+1)%saveInterval==0: mdl.save(mdlName+ "_e"+str(e+1) +".h5", mdl)
        print("\n[:+:] --------------| epoch:", e+1 ,"/",num_epochs,"| loss: ", loss, '| -----------> [model saved][:+:]')
    return mdl
#--------------------------------------------------------------------------------------------------------------------------------------------------|

#--------------------------------------------------------------------------------------------------------------------------------------------------|
if (splitData):
    print('\nSplitting Data...')
    ecg_train, ecg_test, volt_train, volt_test= train_test_split(ecgNorm_70, vm_norm_T_70, test_size= 0.05, random_state=42)
    print('\necg_train, ecg_test, vm_train, vm_test: ', ecg_train.shape, ecg_test.shape, volt_train.shape, volt_test.shape)

if (generateNewModel):
    mdl= transVoltGenerator_new(batch_size= 12)

if(loadModel):
    mdl= keras.models.load_model(currentModel, compile=True)

if (trainModel):
        mdl= trainModel(
                        mdl= mdl, 
                        num_epochs=3, 
                        saveInterval=5, 
                        printerval= 24,
                        X= ecg_train, 
                        Y= volt_train,
                        mdlName= currentModel,
                        #batchCnt=1
                        )

# transVolt_v1_mdl.save('transVolt_v1.h5', transVolt_v1)
# transVolt_v1= keras.models.load_model('transVolt_v1.h5')
keras.models.save_model(transVolt_v1_e11, 'transVolt_v1_e14.h5')
keras.models.load_model('transVolt_v1_e14.h5')
print('\n\nmodel loaded')






