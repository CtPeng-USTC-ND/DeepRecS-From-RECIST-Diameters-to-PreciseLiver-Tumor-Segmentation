from keras.models import *
from keras.layers import *
from keras.optimizers import *

def conv_bn_relu(x, filters, kernel_size):
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x

def m_unet(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)
    conv1 = conv_bn_relu(inputs, 64, (3,3))
    conv1 = conv_bn_relu(conv1, 64, (3,3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    tconv1 = Conv2DTranspose(64, 3, strides=2, activation='relu',padding='same')(pool1)
    skip1 = Subtract()([conv1, tconv1])
    skip1 = conv_bn_relu(skip1, 64, (3, 3))

    conv2 = conv_bn_relu(pool1, 128, (3,3))
    conv2 = conv_bn_relu(conv2, 128, (3,3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    tconv2 = Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(pool2)
    skip2 = Subtract()([conv2, tconv2])
    skip2 = conv_bn_relu(skip2, 128, (3, 3))

    conv3 = conv_bn_relu(pool2, 256, (3,3))
    conv3 = conv_bn_relu(conv3, 256, (3,3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    tconv3 = Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same')(pool3)
    skip3 = Subtract()([conv3, tconv3])
    skip3 = conv_bn_relu(skip3, 256, (3, 3))

    conv4 = conv_bn_relu(pool3, 512, (3,3))
    conv4 = conv_bn_relu(conv4, 512, (3,3))
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    tconv4 = Conv2DTranspose(512, 3, strides=2, activation='relu', padding='same')(pool4)
    skip4 = Subtract()([drop4, tconv4])
    skip4 = conv_bn_relu(skip4, 512, (3, 3))

    conv5 = conv_bn_relu(pool4, 1024, (3,3))
    conv5 = conv_bn_relu(conv5, 1024, (3,3))
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(512, 2, strides=2, activation='relu',padding='same')(drop5)
    merge6 = concatenate([skip4,up6], axis = 3)
    conv6 = conv_bn_relu(merge6, 512, (3,3))
    conv6 = conv_bn_relu(conv6, 512, (3,3))

    up7 = Conv2DTranspose(256,2,strides=2,activation='relu',padding='same')(conv6)
    merge7 = concatenate([skip3,up7], axis = 3)
    conv7 = conv_bn_relu(merge7, 256, (3,3))
    conv7 = conv_bn_relu(conv7, 256, (3,3))

    up8 = Conv2DTranspose(128, 2, strides=2, activation='relu',padding='same')(conv7)
    merge8 = concatenate([skip2,up8], axis = 3)
    conv8 = conv_bn_relu(merge8, 128, (3,3))
    conv8 = conv_bn_relu(conv8, 128, (3,3))

    up9 = Conv2DTranspose(64, 2, strides=2, activation='relu',padding='same')(conv8)
    merge9 = concatenate([skip1,up9], axis = 3)
    conv9 = conv_bn_relu(merge9, 64, (3,3))
    conv9 = conv_bn_relu(conv9, 64, (3,3))
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)


    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


