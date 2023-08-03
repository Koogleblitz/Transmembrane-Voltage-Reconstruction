
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras, initializers
from keras import Model
from keras import layers
from keras.models import Sequential, load_model, save_model
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, Input, Conv2D, Conv1D, Concatenate, \
        MaxPool2D, MaxPool1D, GlobalAvgPool2D, Activation, Dropout, PReLU,BatchNormalization, Flatten,\
        UpSampling2D, LeakyReLU, Dense, Input, add, Multiply
from keras.losses import MeanSquaredError
from keras.applications import VGG19
from sklearn.model_selection import train_test_split
from tqdm import tqdm


### __[+]__ Super Resolution GAN Modules---------------------------------
def res_block(ip):
    res_model = Conv2D(64, (3,3), padding = "same")(ip)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    res_model = PReLU(shared_axes = [1,2])(res_model)
    res_model = Conv2D(64, (3,3), padding = "same")(res_model)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    return add([ip,res_model])
def upscale_block(ip):
    up_model = Conv2D(256, (3,3), padding="same")(ip)
    up_model = UpSampling2D( size = 2 )(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)
    return up_model
def buildGenerator(gen_ip, num_res_block):
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)
    temp = layers
    for i in range(num_res_block):
        layers = res_block(layers)
    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers,temp])
    layers = upscale_block(layers)
    layers = upscale_block(layers)
    op = Conv2D(3, (9,9), padding="same")(layers)
    return Model(inputs=gen_ip, outputs=op)
def discriminator_block(ip, filters, strides=1, batNorm=True):
    # disc_model = Conv2D(filters, (3,3), strides = strides, padding="same")(ip)
    # if batNorm:
    #     disc_model = BatchNormalization( momentum=0.8 )(disc_model)
    # disc_model = LeakyReLU( alpha=0.2 )(disc_model)
    # return disc_model
    tensor= Conv2D(filters, (3,3), strides= strides, padding="same")(ip)
    if batNorm: tensor= BatchNormalization(momentum=0.8)(tensor)
    tensor= LeakyReLU(alpha=0.2)(tensor)
    return tensor
def buildDiscrminator(disc_ip):
    df = 64
    d1 = discriminator_block(disc_ip, df, batNorm=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)
    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)
    return Model(disc_ip, validity)
def build_vgg(hr_shape):
    vgg = VGG19(weights="imagenet",include_top=False, input_shape=hr_shape)
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)
def buildSuperRez(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    gen_features = vgg(gen_img)
    disc_model.trainable = False
    validity = disc_model(gen_img)
    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])
#-----------------------------------------------------------------------------------//

#[+]-----------MikelNet----------------------------------------------------------------------------\\
# torch.nn.Conv1d(n_channels =12, out_channels =96, kernel_size=7, stride=2)
#[+:] sqExpand: the output shape of the squeeze layer is the input shape of the expand layers
def fire_1D_fx(x,  initShape,sqExpand,exFilter_1x1, exFilter_3x3):
    squeezed     =  Conv1D(input_shape= (None, initShape), filters= sqExpand, kernel_size=1, activation='relu')(x)
    expanded_1x1 =  Conv1D(input_shape= (None,sqExpand), filters=exFilter_1x1,  kernel_size=1, activation='relu')(squeezed)
    expanded_3x3 =  Conv1D(input_shape= (None,sqExpand), filters= exFilter_3x3, kernel_size=3, activation='relu', padding='same')(squeezed)
    x= Concatenate()([expanded_1x1, expanded_3x3])
    print('        [fire_1D_fx:]:==> x', x.shape)
    # x= Activation('relu')(x)
    return x  
def ecg2vm_fx(x, dropout= 0.5, kernel_size = 3):
    print('\n[:+:] --------------------------------------------------------------------------- [:+:]\n')
    # Conv1D(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    #[+]-- Conv1D layers expect their input to have a rank of 3, with dimensions (batch_size, steps, input_dim). so not sure if (12,500) is correct
    # x = Reshape((12, 500))(x)
    print('x--> [ecg2vm_fx]: ', x.shape)
    x = Reshape((500,12))(x)
    print('x--> [ecg2vm_fx.Conv1D] reshaped: ', x.shape)
    x= Conv1D(input_shape= (None,12), filters=64, kernel_size=kernel_size, strides=1, padding='same')(x)
    print('    \nx--> [MaxPool1D_1] : ', x.shape)
    x= MaxPool1D(pool_size=kernel_size, strides=1, padding='same')(x)
    x= fire_1D_fx(x,64, 16, 64, 64)
    x= fire_1D_fx(x,128, 16, 64, 64)
    print('    \nx--> [MaxPool1D_2]: ', x.shape)
    x= MaxPool1D(pool_size=kernel_size, strides=1, padding='same')(x)
    x= fire_1D_fx(x,128, 32, 128, 128)
    x= fire_1D_fx(x,256, 32, 128, 128)
    print('    \nx--> [MaxPool1D_3]: ', x.shape)
    x= MaxPool1D(pool_size=kernel_size, strides=1, padding='same')(x)
    x= fire_1D_fx(x,256, 48, 192, 192)
    x= fire_1D_fx(x,384, 48, 192, 192)
    x= fire_1D_fx(x,384, 64, 256, 256)
    x= fire_1D_fx(x,512, 64, 256, 256)
    print('\n    [ecg2vm_fx:]:==> x', x.shape)
    return x

def build_ecg2vm(main_fx, homeStretch_fx, ecg_ip, vm_ip):
    x= main_fx(ecg_ip)
    y_pred= homeStretch_fx(x)
    print('\nbuild_ecg2vm: homeStretch_fx(x):= y_pred=    ', y_pred.shape, type(y_pred))
    y_truth= vm_ip
    print('build_ecg2vm: y_truth=     ', y_truth.shape, type(y_truth))
    print(" type of 'none': ", type(y_truth.shape[0]))
    # mse= MeanSquaredError(y_truth, y_pred)
    #[+]--
    return Model(inputs= ecg_ip, outputs= y_pred )
    # return Model(inputs= [ecg_ip, vm_ip], outputs= y_pred )

def transVolt_fx(x, dropout= 0.5, kernel_size = 3, div= 1):
    print('\n[:+:] --------------  x--> [transVolt_fx]: ', x.shape,'----------------------------------------------------------- [:+:]\n')
    # Conv1D(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    #[+]-- Conv1D layers expect their input to have a rank of 3, with dimensions (batch_size, steps, input_dim). so not sure if (12,500) is correct
    # x = Reshape((12, 500))(x)

    x = Reshape((500//div,12))(x)
    print('x--> [ecg2vm_fx.Conv1D] reshaped: ', x.shape)

    x= Conv1D(input_shape= (None,12//div), filters=64//div, kernel_size=kernel_size, strides=1, padding='same')(x)
    print('    \nx--> [MaxPool1D_1] : ', x.shape)

    x= MaxPool1D(pool_size=kernel_size, strides=1, padding='same')(x)
    x= fire_1D_fx(x,64//div, 16//div, 64//div, 64//div)
    x= fire_1D_fx(x,128//div, 16//div, 64//div, 64//div)
    print('    \nx--> [MaxPool1D_2]: ', x.shape)

    x= MaxPool1D(pool_size=kernel_size, strides=1, padding='same')(x)
    x= fire_1D_fx(x,128//div, 32//div, 128//div, 128//div)
    x= fire_1D_fx(x,256//div, 32//div, 128//div, 128//div)
    print('    \nx--> [MaxPool1D_3]: ', x.shape)

    x= MaxPool1D(pool_size=kernel_size, strides=1, padding='same')(x)
    x= fire_1D_fx(x,256//div, 48//div, 192//div, 192//div)
    x= fire_1D_fx(x,384//div, 48//div, 192//div, 192//div)
    x= fire_1D_fx(x,384//div, 64//div, 256//div, 256//div)
    x= fire_1D_fx(x,512//div, 64//div, 256//div, 256//div)
    print('\n    [ecg2vm_fx:]:==> x', x.shape)
    return x

def homeStretch_fx(x, dropout= 0.5):
    print('\nx --> [homeStretch_fx]: ', x.shape)
    Dropout(rate=dropout)
    x=Conv1D(input_shape=(None, 512), filters=75, kernel_size=1, padding='valid' )(x)
    return x

def attentionMatrix_fx(inputs, r=2):
    print('\n[:+:]-------------------- x-->[attentionMatrix_fx] (bat, h, w, ch)=', inputs.shape)
    # bat, h, w, ch= inputs.shape
    ch= inputs.shape[3]

    #[+]-- Squeeze each channel matrix into a single 1x1 value
    x= GlobalAveragePooling2D()(inputs)
    print('    [GlobalAveragePooling2D](squeezed) x -->: ', x.shape)

    #[+]-- The bottleneck MLP is used to generate the weights to scale each channel of the feature map adaptively
        #[+] reduce features by a factor of 'r', which has dramatic inverse relation with parameter count
    x= Dense(units=ch//r, activation= "relu", use_bias= False)(x)
    print('    [Dense](exite 1) x -->: ', x.shape)

        #[+] Sigmoid outputs a val from 0-1; gives each feature a 'relavance score'
    x= Dense(units=ch, activation= "sigmoid", use_bias= False)(x)
    print('    [Dense](exite 2) x -->: ', x.shape)

    #[+]-- Applies score, only works if batch_size == width, thus, we will use a batch size of 12
    x= inputs*x
    
    print('\n        [ x=inputs*x](Return scaled) x -->: ', x.shape, '\n')
    return x




'''
#[+]-
    - [x] seperate each 500x12 input into 5 100x12 inputs
    - [x] stack the pieces along the channel layers
    - [x] feed into squeeze_100x12_fx
    - [x] apply the squeeze and exitation layer to for channel-wise attention
    - [x] concatenate the pieces together
    bonus
      - [] embed each label as a 100x12 matrix and stack along the channel dim
      - [] svd: replace a maxpool layer, or append/replace the squeeze+exitation
'''
def transVolt_build(ip, 
                    ecg2vm_fx= ecg2vm_fx, 
                    attentionMatrix_fx=attentionMatrix_fx, 
                    homeStretch_fx= homeStretch_fx, 
                    div=5):
    print('\n[:+:] --------------  x--> [trasVolt_mdl]: ', ip.shape,'----------------------------------------------------------- [:+:]\n')


    #[+]----------------------squeezeNet Branch-----------------------||
    x1= ecg2vm_fx(ip)
    y1= homeStretch_fx(x1)
    print('\n          [homeStretch_fx(x1)]:=====> x', y1.shape, '\n')
    #----------------------------------------------------------


    #-------------------------Attention Matrix------------------------------------||
    attMat = tf.split(ip, num_or_size_splits=div, axis=1)
    print('\n[%]-- attMatSplit: ', len(attMat) ,'x', attMat[0].shape)
    attMatStacked = tf.concat(attMat, axis=-1)
    print('[=]-- attMatStacked:   ', attMatStacked.shape)
    attMatBat= attentionMatrix_fx(attMatStacked)
    print('\n[+]-- attMatBatch:   ', attMatBat.shape)
    # attMatConcat= attMatBat[:, :, :, 0]
    # attMatConcat= tf.convert_to_tensor([(Concatenate(axis=1)([attMatConcat, attMatBat[:, :, :, i]])) for i in range(1,div)])
    attMatConcat= Concatenate(axis=1)([attMatBat[:, :, :, 0], attMatBat[:, :, :, 1], 
                                       attMatBat[:, :, :, 2], attMatBat[:, :, :, 3],
                                       attMatBat[:, :, :, 4]]) 
    print('[_]-- attMatConcatFlat', attMatConcat.shape)
    attMatSplat= homeStretch_fx(attMatConcat)
    print('\n      [homeStretch_fx(attMatConcat)]:=====> attMatSplat', attMatSplat.shape)
    #----------------------------------------------------------------------------

    attMatFlatMultiplat= Multiply()([attMatSplat, y1])
    print('\n            [:+:]|[travsVolt Prediction]:====================> attMatFlatMultiplat', attMatFlatMultiplat.shape)
    return Model(inputs= ip, outputs= attMatFlatMultiplat)
    # return Model(inputs= ecg_ip, outputs= y1 )
#-----------------------------------------------------------------------------------//


# sampInput= Input(shape= (500, 12, 1), batch_size=32)
# print('\nsampInput: ', sampInput.shape, type(sampInput))
# transVolt_build(ecg_ip= sampInput, div=5)


def testMdl(build_fx ,ip,shape= (500,12,1), batch_size=12, div=5):
    ecg_ip= Input(shape= shape, batch_size=batch_size)
    mdl= build_fx(ecg_ip, div=div)
    y = mdl(xB).numpy()
    print('\n\n[+] output:------------->',y.shape)
    plt.plot(y[0])
    plt.title('output')
    plt.show()
    print('row: \n', y[0, :20, 0])


xB= np.load('ecg_norm.npy')[0:12]
print('xB shape: (batch_size, timeStep, signals, channels) = ',xB.shape)#[+]xB--------> (32, 500, 12)
# yB= np.load('vm_norm_T.npy')[0:12]
# print('yB shape: (batch_size, timeStep, signals, channels) = ',yB.shape)#[+]yB--------> (32, 500, 12)

#[:+:]-----------------------------------------------------------------------------------------------------------------------------|||

# testMdl(build_fx= transVolt_build, ip= xB)

