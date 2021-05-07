from keras.layers import *
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.losses import mean_squared_error

smooth=1.
batch_size = 20

def est_argmax(x, beta=1e2):
    return tf.reduce_sum(tf.cumsum(tf.ones_like(x)) * tf.exp(beta * x) / tf.reduce_sum(tf.exp(beta * x))) - 1

def mse_cosine_loss(y_true, y_pred):
    # mse = K.variable([0.])
    cos=K.variable([0.])
    for i in range(batch_size):
        long_l_r, long_l_c, long_r_r, long_r_c, short_l_r, short_l_c, short_r_r, short_r_c = get_point(i, y_pred)
        # pred_pts = tf.convert_to_tensor([long_l_r, long_l_c, long_r_r, long_r_c,
        #                                  short_l_r, short_l_c, short_r_r, short_r_c], 'float32')

        # long_l_r_, long_l_c_, long_r_r_, long_r_c_, short_l_r_, short_l_c_, short_r_r_, short_r_c_ = get_point(i, y_true)
        # gt_pts = tf.convert_to_tensor([long_l_r_, long_l_c_, long_r_r_, long_r_c_,
        #                                short_l_r_, short_l_c_, short_r_r_, short_r_c_], 'float32')

        # mse = mse+ mean_squared_error(gt_pts, pred_pts)
        v11 = K.cast(long_r_r, 'float32') - K.cast(long_l_r, 'float32')
        v12 = K.cast(long_r_c, 'float32') - K.cast(long_l_c, 'float32')
        v21 = K.cast(short_r_r, 'float32')-K.cast(short_l_r, 'float32')
        v22 = K.cast(short_r_c, 'float32')-K.cast(short_l_c, 'float32')
        cos = cos + K.abs((v11*v21+v12*v22)/(K.sqrt(v11*v11+v12*v12)*K.sqrt(v21*v21+v22*v22)+smooth))
    # return 0.05*(mse+0*cos)
    return mean_squared_error(y_true,y_pred)+cos/batch_size

def get_point(i, y_pred):
    long_l = y_pred[i, :, :, 0]
    long_r = y_pred[i, :, :, 1]
    short_l = y_pred[i, :, :, 2]
    short_r = y_pred[i, :, :, 3]
    f_long_l = K.flatten(long_l)
    f_long_r = K.flatten(long_r)
    f_short_l = K.flatten(short_l)
    f_short_r = K.flatten(short_r)

    long_l_r = est_argmax(f_long_l/tf.norm(f_long_l)) / 64
    long_l_c = est_argmax(f_long_l/tf.norm(f_long_l)) % 64
    long_r_r = est_argmax(f_long_r/tf.norm(f_long_r)) / 64
    long_r_c = est_argmax(f_long_r/tf.norm(f_long_r)) % 64
    short_l_r = est_argmax(f_short_l/tf.norm(f_short_l)) / 64
    short_l_c = est_argmax(f_short_l/tf.norm(f_short_l)) % 64
    short_r_r = est_argmax(f_short_r/tf.norm(f_short_r)) / 64
    short_r_c = est_argmax(f_short_r/tf.norm(f_short_r)) % 64
    return long_l_r, long_l_c, long_r_r, long_r_c, \
        short_l_r, short_l_c, short_r_r, short_r_c
