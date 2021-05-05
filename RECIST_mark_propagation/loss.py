from keras.layers import *
import keras.backend as K
import tensorflow as tf
import numpy as np

smooth=1.
batch_size = 16

def est_argmax(x):
    x = Activation('sigmoid')(x)
    a = np.arange(64*64)
    aa = tf.convert_to_tensor(a, dtype=tf.int32)
    aa = K.cast(aa, 'float32')
    out = x*aa
    out = K.sum(out)
    return out

def mse_cosine_loss(y_true, y_pred):
    cosine=K.variable([0.])
    for i in range(batch_size):
        topm=y_pred[i,:,:,0]
        bottomm = y_pred[i, :, :, 1]
        leftm = y_pred[i, :, :, 2]
        rightm = y_pred[i, :, :, 3]

        bt_f=K.flatten(bottomm)
        tp_f = K.flatten(topm)
        lt_f = K.flatten(leftm)
        rt_f = K.flatten(rightm)
        b_r = est_argmax(bt_f) / 64
        b_c = est_argmax(bt_f)% 64
        t_r = est_argmax(tp_f) / 64
        t_c = est_argmax(tp_f) % 64
        l_r = est_argmax(lt_f) / 64
        l_c = est_argmax(lt_f) % 64
        r_r = est_argmax(rt_f) / 64
        r_c = est_argmax(rt_f) % 64
        v11 = K.cast(b_r, 'float32') - K.cast(t_r, 'float32')
        v12 = K.cast(b_c, 'float32') - K.cast(t_c, 'float32')
        v21 = K.cast(r_r, 'float32')-K.cast(l_r, 'float32')
        v22 = K.cast(r_c, 'float32')-K.cast(l_c, 'float32')
        cosine = cosine + 0.05*K.abs((v11*v21+v12*v22)/(K.sqrt(v11*v11+v12*v12)*K.sqrt(v21*v21+v22*v22)+smooth))
    return K.mean(K.square(y_pred - y_true), axis=-1)+ cosine
