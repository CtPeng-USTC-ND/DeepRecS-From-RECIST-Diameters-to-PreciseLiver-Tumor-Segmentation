
from keras.models import *
from keras.layers import *
from keras.optimizers import *

def deepGCnet(pretrained_weights = None,input_size = (256,256,4)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)#He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生,其中fan_in权重张量的扇入
    conv1 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization(epsilon=1e-5)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization(epsilon=1e-5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization(epsilon=1e-5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization(epsilon=1e-5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization(epsilon=1e-5)(conv5)
    conv5=Dropout(0.5)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    x=Conv2D(512, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    x = Conv2D(512, 1, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x=BatchNormalization(epsilon=1e-5)(x)
    x=UpSampling2D(size = (2,2))(x)
    x=Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x=UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    x = BatchNormalization(epsilon=1e-5)(x)#
    x = Conv2D(2, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(1, 1, activation = 'sigmoid')(x)

    model = Model(input = inputs, output = x)

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


