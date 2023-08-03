# import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras, initializers
# from keras import Model
# from keras import layers
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.layers import Input
from fxModules import *

#------------------------------------------------------------------------Predict----------------------------------------------------------------|
loadFullData= False
loadPartialData= True
splitData=True
loadModel= True
currentModel= 'transVolt_v1_e1019'
mdlLbl= "Attention-based Deep Network for Heart Disease"
mapLbl= "ADHD Prediction"
epochs= '1019'
# currentModel= 'transVolt_v1_e'+epochs
predict= True
metrics= True

# loadFullData= False
# loadPartialData= True
# splitData=True
# loadModel= True
# currentModel= 'ecg2vm_mdl_e82'
# # currentModel= 'transVolt_v1_e1019'
# mdlLbl= "SqueezeNet 1D"
# mapLbl= "Target Vm"
# epochs= '82'
# # currentModel= 'transVolt_v1_e'+epochs
# predict= True
# metrics= True

#-----------------------------------------------------
# ecg_ip= Input(shape= (500, 12, 1), batch_size=32)
# vm_ip= Input(shape= (500, 75, 1), batch_size=32)
# print('ecg_ip shape: (batch_size, timeStep, signals, channels) = ',ecg_ip.shape)
# mikelMdl= build_ecg2vm(ecg2vm_fx, homeStretch_fx, ecg_ip, vm_ip )
# keras.

#[+]--------------------------------------------------------------------------


# print('\n [:+:] -- currentModel: ', currentModel)
if(loadFullData):
    print('\nLoading Data...')
    vm_norm_T= np.load('vm_norm_T.npy')
    print('\nvm_norm_T: ', vm_norm_T.shape)
    ecgNorm= np.load('ecg_norm.npy')
    print('ecgNorm: ', ecgNorm.shape)

if(loadPartialData):
    vm_norm_T_70= np.load('vm_norm_T_70.npy')
    print('vm_norm_T_70: ', vm_norm_T_70.shape)
    ecgNorm_70= np.load('ecgNorm_70.npy')
    print('ecgNorm_70: ', ecgNorm_70.shape)

if (splitData):
    print('\nSplitting Data...')
    ecg_train, ecg_test, volt_train, volt_test= train_test_split(ecgNorm_70, vm_norm_T_70, test_size= 0.05, random_state=42)
    print('\necg_train, ecg_test, vm_train, vm_test: ', ecg_train.shape, ecg_test.shape, volt_train.shape, volt_test.shape)
    testEcg_tmp= np.load('testEcg_tmp.npy')
    testVm_tmp= np.load('testVm_tmp.npy')

if(loadModel):
    mdl= keras.models.load_model(currentModel+'.h5', compile=True)


if(predict):
    [X1, X2] = [testEcg_tmp, testVm_tmp]
    ix = np.random.randint(0, len(X1), 1)
    src_ecg, tar_vm = X1[ix], X2[ix]

    gen_volt = mdl.predict(src_ecg)
    print('prediction: ', gen_volt.shape)
    predAvg= np.load('vmTruth.npy')
    print('vmTruth: ', predAvg.shape)


#----------------------------------------------------------------------------
    print(src_ecg.shape,'--->', tar_vm.shape,':==   ', gen_volt.shape)
    plt.figure(figsize = (8,6 ))
    plt.plot(range(500), predAvg[0], label= 'Ground Truth')
    plt.plot(range(500), np.mean(gen_volt[0], axis=1), label= 'Prediction')
    plt.xlabel('Time(msec)')
    plt.ylabel('Potential(Vm)')
    plt.title(mdlLbl)
    plt.legend()
    plt.show()



    # plt.figure(figsize=(16, 8))
    # plt.subplot(231)
    # plt.title('ECG')
    # plt.imshow(src_ecg[0])

    # plt.subplot(232)
    # plt.title('Actual Volt')
    # plt.imshow(tar_vm[0])

    # plt.subplot(233)
    # plt.title('Predicted Volt')
    # plt.imshow(gen_volt[0])
    # plt.show()

#----------------------------------- Map ----------------------------------
    pECGData= gen_volt
    VmData= tar_vm
    row = 1
    column = 3

    plt.figure(figsize=(20,5))
    plt.subplot(row, column, 1)
    plt.imshow(gen_volt.T, cmap='jet', interpolation='nearest', aspect='auto')
    plt.title('Predicted Vm')
    plt.subplot(row, column, 2)

    plt.text(0.5, 0.5, '-------->', fontsize=40, horizontalalignment='center', verticalalignment='center')

    plt.axis('off')
    plt.subplot(row, column, 3)
    plt.imshow(VmData.T, cmap='jet', interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.title('Target Vm')
    plt.show()
    plt.close()


#need: rse, mse, rmse, mae
def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))
def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0)
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean()
def Corr(pred, true):
    sig_p = np.std(pred, axis=0)
    sig_g = np.std(true, axis=0)
    m_p = pred.mean(0)
    m_g = true.mean(0)
    ind = (sig_g != 0)
    corr = ((pred - m_p) * (true - m_g)).mean(0) / (sig_p * sig_g)
    corr = (corr[ind]).mean()
    return corr
def MAE(pred, true):
    return np.mean(np.abs(pred-true))
def MSE(pred, true):
    return np.mean((pred-true)**2)
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))
def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))
def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    #corr1 = CORR(pred, true)
    corr = Corr(pred, true)
    return mae,mse,rmse,mape,mspe,corr

if(metrics):
    pred= gen_volt
    true= tar_vm
    print('RSE', RSE(pred, true))
    # print('CORR', CORR(pred, true))
    # print('Corr',(pred, true))
    print('MAE',MAE(pred, true))
    # print('MSEMSE',(pred, true))
    print('RMSE',RMSE(pred, true))
    # print('MAPE',MAPE(pred, true))
    # print('MSPE',MSPE(pred, true))
    # print('metric', metric(pred, true))
    print('MSE', MSE(pred,true))




# vm_norm_T= np.load('vm_norm_T.npy')
# vmAvg= np.mean(vm_norm_T, axis=2)
# print('vmAvg', vmAvg.shape)
# plt.plot(vmAvg)


