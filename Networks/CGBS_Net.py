# -*- coding: utf-8 -*-

"""
This model is based on TF repo:
https://github.com/tensorflow/models/tree/master/research/deeplab
On Pascal VOC, original model gets to 84.56% mIOU

# Reference
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Xception: Deep Learning with Depthwise Separable Convolutions]
    (https://arxiv.org/abs/1610.02357)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import Cropping2D
from keras.engine import Layer
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras import regularizers
from keras.utils.data_utils import get_file
import tensorflow as tf
from attention_module import attach_attention_module


def get_crop_shape(target, refer):
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)
    return (cw1, cw2), (ch1, ch2)


class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                     input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                    input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False)(x)
    x = BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5),
               use_bias=False)(x)
    x = BatchNormalization(epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size), kernel_regularizer=regularizers.l2(1e-5),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate))(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size), kernel_regularizer=regularizers.l2(1e-5),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate))(x)


def _xception_block(inputs, kernel_size, depth_list, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              stride=stride if i == 2 else 1,
                              kernel_size=kernel_size,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1],
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization()(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def Xception_encoder(x, kernel_size, entry_block3_stride, middle_block_rate, exit_block_rates):
    x1 = x
    x = Conv2D(32, (kernel_size, kernel_size), strides=(2, 2), kernel_regularizer=regularizers.l2(1e-5),
               use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = _conv2d_same(x, 64, kernel_size=kernel_size, stride=1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x2 = x
    x = _xception_block(x, kernel_size, [128, 128, 128],
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)
    x, x3 = _xception_block(x, kernel_size,[256, 256, 256],
                               skip_connection_type='conv', stride=2,
                               depth_activation=False, return_skip=True)
    x4 = x
    x = _xception_block(x, kernel_size,[512, 512, 512],
                        skip_connection_type='conv', stride=entry_block3_stride,
                        depth_activation=False)
    for i in range(16):
        x = _xception_block(x, kernel_size,[512, 512, 512],
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)

    x = _xception_block(x, kernel_size,[512, 728, 728],
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                        depth_activation=False)
    x = _xception_block(x, kernel_size,[728, 728, 1024],
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                        depth_activation=True)
    x5 = x
    return x, x1, x2, x3, x4, x5


def aspp_fusion(x, input_shape, OS, atrous_rates):
    b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
    b4 = Conv2D(256, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5),
                use_bias=False)(b4)
    b4 = BatchNormalization(epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-5))(x)
    b0 = BatchNormalization(epsilon=1e-5)(b0)
    b0 = Activation('relu')(b0)

    # rate = 6 (12)
    b1 = SepConv_BN(x, 256,
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256,
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256,
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])
    x = Conv2D(256, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5),
               use_bias=False)(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    return x

def CGBS_Net(input_shape=(256, 256, 4), classes=2, OS=16, crop_rate=3):
    img_input1 = Input(input_shape)
    img_input2 = Input(input_shape)

    entry_block3_stride = 2
    middle_block_rate = 1
    exit_block_rates = (1, 2)

    main_kernel_size = 3
    context_kernel_size = 5

    #main branch
    x1, f1, f2, skip1, f4, f5 = Xception_encoder(img_input1, main_kernel_size,
                                                 entry_block3_stride, middle_block_rate, exit_block_rates)
    #contextual branch
    x2, _,_,skip2,_,_ = Xception_encoder(img_input2, context_kernel_size,
                                         entry_block3_stride, middle_block_rate, exit_block_rates)
    #boundary learning branch
    shape_full, shape_concat_s, shape_concat_l = bl_branch(f1, f2, skip1, f4, f5)
    # x1 = aspp_Xception(x1, 'channel1', input_shape, OS, atrous_rates)
    # x2 = aspp_Xception(x2, 'channel2_', input_shape, OS, atrous_rates)

    x1 = conv_bn_relu(x1, 256, (1,1))
    x2 = conv_bn_relu(x2, 256, (1, 1))

    # decoder
    w, h = x1.get_shape()[2].value, x1.get_shape()[1].value
    x2 = BilinearUpsampling(output_size=(int(np.ceil(h * crop_rate)),
                                         int(np.ceil(w * crop_rate))))(x2)
    cw, ch = get_crop_shape(x2, x1)
    x2 = Cropping2D(cropping=(ch, cw))(x2)
    x = Concatenate()([x1, x2, shape_concat_s])
    x = aspp_fusion(x, input_shape, OS, atrous_rates=(3, 6, 9))
    x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                        int(np.ceil(input_shape[1] / 4))))(x)

    w, h = skip1.get_shape()[2].value, skip1.get_shape()[1].value
    skip2 = BilinearUpsampling(output_size=(int(np.ceil(h * crop_rate)),
                                            int(np.ceil(w * crop_rate))))(skip2)
    cw, ch = get_crop_shape(skip2, skip1)
    skip2 = Cropping2D(cropping=(ch, cw))(skip2)
    skip = Concatenate()([skip1, skip2, shape_concat_l])
    skip = aspp_fusion(skip, input_shape, OS=4, atrous_rates=(6, 12, 18))

    dec_skip1 = BatchNormalization(epsilon=1e-5)(skip)
    dec_skip1 = Activation('relu')(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, depth_activation=True, epsilon=1e-5)

    x = Conv2D(classes, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = Conv2D(1, 1, kernel_regularizer=regularizers.l2(1e-5), activation='sigmoid')(x)
    x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]), name='out_seg')(x)

    model = Model(inputs=[img_input1, img_input2], outputs=[x, shape_full], name='CGBS-Net')
    return model

def conv_bn_relu(x, filters, kernel_size):
    x = Conv2D(filters, kernel_size, padding='same',
               use_bias=False, kernel_regularizer=regularizers.l2(1e-5))(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x

def bl_branch(f1, f2, f3, f4, f5):
    f1 = conv_bn_relu(f1, 64, (3,3))
    f1 = attach_attention_module(f1, f1)
    f2 = conv_bn_relu(f2, 64, (3, 3))
    f2 = attach_attention_module(f2, f2)
    f3 = conv_bn_relu(f3, 64, (3, 3))
    f3 = attach_attention_module(f3, f3)
    f4 = conv_bn_relu(f4, 64, (3, 3))
    f4 = attach_attention_module(f4, f4)
    f5 = conv_bn_relu(f5, 64, (3, 3))
    f5 = attach_attention_module(f5, f5)

    up4 = BilinearUpsampling(output_size=(int(256 / 8), int(256 / 8)))(f5)
    up4 = Concatenate()([f4, up4])
    up4 = Conv2D(64, (3, 3), padding='same', use_bias=False,
                 kernel_regularizer=regularizers.l2(1e-5))(up4)
    up3 = BilinearUpsampling(output_size=(int(256 / 4), int(256 / 4)))(up4)
    up3 = Concatenate()([f3, up3])
    up3 = Conv2D(64, (3, 3), padding='same', use_bias=False,
                 kernel_regularizer=regularizers.l2(1e-5))(up3)
    edge_skip_l = conv_bn_relu(up3, 256, (1,1))
    up2 = BilinearUpsampling(output_size=(int(256 / 2), int(256 / 2)))(up3)
    up2 = Concatenate()([f2, up2])
    up2 = Conv2D(64, (3, 3), padding='same', use_bias=False,
                 kernel_regularizer=regularizers.l2(1e-5))(up2)
    up1 = BilinearUpsampling(output_size=(256, 256))(up2)
    up1 = Concatenate()([f1, up1])
    up1 = conv_bn_relu(up1, 256, (3,3))
    edge_full = Dropout(0.1)(up1)
    edge_skip_s = BilinearUpsampling(output_size=(int(256/16), int(256/16)))(edge_full)
    edge_full = Conv2D(2, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(edge_full)
    edge_full = Conv2D(1, 1, kernel_regularizer=regularizers.l2(1e-5),
                   activation='sigmoid',name='out_shape')(edge_full)
    return edge_full, edge_skip_s, edge_skip_l

