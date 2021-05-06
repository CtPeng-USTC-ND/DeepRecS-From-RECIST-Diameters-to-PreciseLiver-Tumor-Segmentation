from keras.layers import *
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.losses import mean_squared_error

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
    mse = K.variable([0.])
    cos=K.variable([0.])
    for i in range(batch_size):
        t_r, t_c, b_r, b_c, l_r, l_c, r_r, r_c = get_point(i, y_pred)
        pred_pts = tf.convert_to_tensor([t_r, t_c, b_r, b_c, l_r, l_c, r_r, r_c], 'float32')
        t_r_, t_c_, b_r_, b_c_, l_r_, l_c_, r_r_, r_c_ = get_point(i, y_true)
        gt_pts = tf.convert_to_tensor([t_r_, t_c_, b_r_, b_c_, l_r_, l_c_, r_r_, r_c_], 'float32')
        mse = mse+ mean_squared_error(gt_pts, pred_pts)
        v11 = K.cast(b_r, 'float32') - K.cast(t_r, 'float32')
        v12 = K.cast(b_c, 'float32') - K.cast(t_c, 'float32')
        v21 = K.cast(r_r, 'float32')-K.cast(l_r, 'float32')
        v22 = K.cast(r_c, 'float32')-K.cast(l_c, 'float32')
        cos = cos + K.abs((v11*v21+v12*v22)/(K.sqrt(v11*v11+v12*v12)*K.sqrt(v21*v21+v22*v22)+smooth))
    return 0.05*(mse+cos)

def get_point(i, y_pred):
    topm = y_pred[i, :, :, 0]
    bottomm = y_pred[i, :, :, 1]
    leftm = y_pred[i, :, :, 2]
    rightm = y_pred[i, :, :, 3]
    tp_f = K.flatten(topm)
    bt_f = K.flatten(bottomm)
    lt_f = K.flatten(leftm)
    rt_f = K.flatten(rightm)
    t_r = est_argmax(tp_f) / 64
    t_c = est_argmax(tp_f) % 64
    b_r = est_argmax(bt_f) / 64
    b_c = est_argmax(bt_f) % 64
    l_r = est_argmax(lt_f) / 64
    l_c = est_argmax(lt_f) % 64
    r_r = est_argmax(rt_f) / 64
    r_c = est_argmax(rt_f) % 64
    return t_r, t_c, b_r, b_c, l_r, l_c, r_r, r_c
