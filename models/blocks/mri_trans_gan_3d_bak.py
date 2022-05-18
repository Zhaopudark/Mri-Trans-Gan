__all__ = [
    'Conv7S1',
    'DownSampling',
    'ResBlocks',
    'UpSampling',    
    'Conv4S2',
    'Conv4S1', 
]
import sys
import os
import logging
import tensorflow as tf
from models.layers.convolutions import Conv3D,UpSampalingConv3D,UpSubpixelConv3D
from models.layers.normalizations import InstanceNormalization
from models.layers.activations import Activation
"""
Cycle GAN Generator blocks
"""
#--------------------------------------------------------------------------------------------#
class Conv7S1(tf.keras.Model):#c7s1_k
    def __init__(self,
                 filters,
                 spectral_normalization=False,
                 use_bias=False,
                 activation='relu',
                 specific_out_dtype=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(Conv7S1,self).__init__(name=name,dtype=dtype)
        self.l1_conv = Conv3D(filters=filters,
                              kernel_size=[7,7,7],
                              strides=[1,1,1],
                              padding='REFLECT',
                              use_bias=use_bias,
                              spectral_normalization=spectral_normalization,
                              dtype=dtype,
                              **kwargs)
        self.l2_norm = InstanceNormalization(dtype=dtype)
        if specific_out_dtype is None:
            self.l3_activation = Activation(activation,dtype=dtype)
        else:
            self.l3_activation = Activation(activation,dtype=specific_out_dtype,**kwargs)
    def build(self,input_shape):
        flow_shape=self.l1_conv.build(input_shape=input_shape)
        flow_shape=self.l2_norm.build(input_shape=flow_shape)
        flow_shape=self.l3_activation.build(input_shape=flow_shape)
        self.built = True
        output_shape = flow_shape[:]
        return output_shape
    def call(self,x,training=True):
        x = self.l1_conv(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y
#--------------------------------------------------------------------------------------------#
class DownSampling(tf.keras.Model):#dk
    def __init__(self,
                 filters,
                 spectral_normalization=False,
                 use_bias=False,
                 specific_out_dtype=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(DownSampling,self).__init__(name=name,dtype=dtype)
        self.l1_conv = Conv3D(filters=filters,
                              kernel_size=[3,3,3],
                              strides=[2,2,2],
                              padding='REFLECT',
                              use_bias=use_bias,
                              spectral_normalization=spectral_normalization,
                              dtype=dtype,
                              **kwargs)
        self.l2_norm = InstanceNormalization(dtype=dtype)
        if specific_out_dtype is None:
            self.l3_activation = Activation('relu',dtype=dtype)
        else:
            self.l3_activation = Activation('relu',dtype=specific_out_dtype)
    def build(self,input_shape):
        flow_shape=self.l1_conv.build(input_shape=input_shape)
        flow_shape=self.l2_norm.build(input_shape=flow_shape)
        flow_shape=self.l3_activation.build(input_shape=flow_shape)
        self.built = True
        output_shape = flow_shape[:]
        return output_shape
    def call(self,x,training=True):
        x = self.l1_conv(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y
# #--------------------------------------------------------------------------------------------#
class ResBlock(tf.keras.Model):#rk
    def __init__(self,
                 filters,
                 spectral_normalization=False,
                 use_bias=False,
                 specific_out_dtype=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(ResBlock,self).__init__(name=name,dtype=dtype)
        self.l1_conv= Conv3D(filters=filters,
                             kernel_size=[3,3,3],
                             strides=[1,1,1],
                             padding='REFLECT',
                             use_bias=use_bias,
                             spectral_normalization=spectral_normalization,
                             dtype=dtype,
                             **kwargs)
        self.l2_norm = InstanceNormalization(dtype=dtype)
        self.l3_activation =Activation('relu',dtype=dtype)
        self.l4_conv= Conv3D(filters=filters,
                             kernel_size=[3,3,3],
                             strides=[1,1,1],
                             padding='REFLECT',
                             use_bias=use_bias,
                             spectral_normalization=spectral_normalization,
                             dtype=dtype,
                             **kwargs)
        self.l5_norm = InstanceNormalization(dtype=dtype)
        if specific_out_dtype is None:
            self.l6_activation = Activation('linear',dtype=dtype)
        else:
            self.l6_activation = Activation('linear',dtype=specific_out_dtype)
    def build(self,input_shape):
        flow_shape=self.l1_conv.build(input_shape=input_shape)
        flow_shape=self.l2_norm.build(input_shape=flow_shape)
        flow_shape=self.l4_conv.build(input_shape=flow_shape)
        flow_shape=self.l5_norm.build(input_shape=flow_shape)
        self.built = True
        output_shape = flow_shape[:]
        return output_shape
    def call(self,x,training=True):
        y = self.l1_conv(x,training=training)
        y = self.l2_norm(y,training=training)
        y = self.l3_activation(y,training=training)
        y = self.l4_conv(y,training=training)
        y = self.l5_norm(y,training=training)
        y = self.l6_activation(y+x,training=training)
        return y
#--------------------------------------------------------------------------------------------#
class ResBlocks(tf.keras.Model):#rks
    def __init__(self,
                 filters,
                 spectral_normalization=False,
                 use_bias=False,
                 n=6,
                 specific_out_dtype=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(ResBlocks,self).__init__(name=name,dtype=dtype)
        self.rs_list = []
        for _ in range(n):
            self.rs_list.append(ResBlock(filters=filters,
                                         spectral_normalization=spectral_normalization,
                                         use_bias=use_bias,
                                         specific_out_dtype=specific_out_dtype,
                                         dtype=dtype,
                                         **kwargs))
    def build(self,input_shape):
        flow_shape = input_shape
        for item in self.rs_list:
            flow_shape=item.build(input_shape=flow_shape)
        self.built = True
        output_shape = flow_shape[:]
        return output_shape
    def call(self,x,training=True):
        in_put = x
        for item in self.rs_list:
            out_put = item(in_put,training=training)
            in_put = out_put
        return out_put
#--------------------------------------------------------------------------------------------#      
class UpSampling(tf.keras.Model):#uk
    def __init__(self,
                 filters,
                 spectral_normalization=False,
                 use_bias=False,specific_out_dtype=None,
                 name=None,
                 dtype=None,
                 up_sampling_method='up_conv',
                 **kwargs):
        super(UpSampling,self).__init__(name=name,dtype=dtype)
        if up_sampling_method == 'up_conv':
            self.l1_up = UpSampalingConv3D(filters=filters,
                                           kernel_size=[3,3,3],
                                           strides=[2,2,2],
                                           padding='SAME',
                                           use_bias=use_bias,
                                           spectral_normalization=spectral_normalization,
                                           dtype=dtype,
                                           **kwargs)
        elif up_sampling_method == 'sub_pixel_up':
            logging.warning("UpSampling has been replaced by UpSubpixelConv3D")
            self.l1_up = UpSubpixelConv3D(filters=filters,
                                          kernel_size=[3,3,3],
                                          strides=[2,2,2],
                                          padding='SAME',
                                          use_bias=use_bias,
                                          spectral_normalization=spectral_normalization,
                                          dtype=dtype,
                                          **kwargs)
        else:
            raise ValueError(f"Unsupported up_sampling_method {up_sampling_method}!")
        self.l2_norm = InstanceNormalization(dtype=dtype)
        if specific_out_dtype is None:
            self.l3_activation = Activation('relu',dtype=dtype)
        else:
            self.l3_activation = Activation('relu',dtype=specific_out_dtype)
    def build(self,input_shape,output_shape=None):
        flow_shape=self.l1_up.build(input_shape=input_shape)
        flow_shape=self.l2_norm.build(input_shape=flow_shape)
        flow_shape=self.l3_activation.build(input_shape=flow_shape)
        self.built = True
        output_shape = flow_shape[:]
        return output_shape
    def call(self,x,training=True):
        x = self.l1_up(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y

"""
CycleGAN Discriminator blocks
"""
#--------------------------------------------------------------------------------------------#
class Conv4S2(tf.keras.Model):#ck
    def __init__(self,
                 filters,
                 spectral_normalization=False,
                 norm=True,
                 use_bias=False,
                 specific_out_dtype=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(Conv4S2,self).__init__(name=name,dtype=dtype)
        self.l1_conv= Conv3D(filters=filters,
                             kernel_size=[4,4,4],
                             strides=[2,2,2],
                             padding='SAME',
                             use_bias=use_bias,
                             spectral_normalization=spectral_normalization,
                             dtype=dtype,
                             **kwargs)
        if norm:
            self.l2_norm = InstanceNormalization(dtype=dtype)
        else:
            self.l2_norm = Activation('linear',dtype=dtype)
        if specific_out_dtype is None:
            self.l3_activation = Activation('leaky_relu',alpha=0.2,dtype=dtype)
        else:
            self.l3_activation = Activation('leaky_relu',alpha=0.2,dtype=specific_out_dtype)
    def build(self,input_shape):
        flow_shape=self.l1_conv.build(input_shape=input_shape)
        flow_shape=self.l2_norm.build(input_shape=flow_shape)
        flow_shape=self.l3_activation.build(input_shape=flow_shape)
        self.built = True
        output_shape = flow_shape[:]
        return output_shape
    def call(self,x,training=True):
        x = self.l1_conv(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y
#--------------------------------------------------------------------------------------------#
class Conv4S1(tf.keras.Model):#ck last
    def __init__(self,
                 use_sigmoid=True,
                 spectral_normalization=False,
                 specific_out_dtype=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(Conv4S1,self).__init__(name=name,dtype=dtype)
        self.l1_conv= Conv3D(filters=1,
                               kernel_size=[4,4,4],
                               strides=[1,1,1],
                               padding='SAME',
                               use_bias=True,
                               spectral_normalization=spectral_normalization,
                               dtype=dtype,
                               **kwargs)
        if use_sigmoid:
            tmp_activation_name = 'sigmoid'
        else:
            tmp_activation_name = 'linear'
        if specific_out_dtype is None:
            self.l2_activation = Activation(tmp_activation_name,dtype=dtype)
        else:
            self.l2_activation = Activation(tmp_activation_name,dtype=specific_out_dtype)
    def build(self,input_shape):
        flow_shape=self.l1_conv.build(input_shape=input_shape)
        flow_shape=self.l2_activation.build(input_shape=flow_shape)
        self.built = True
        output_shape = flow_shape[:]
        return output_shape
    def call(self,x,training=True):
        x = self.l1_conv(x,training=training)
        y = self.l2_activation(x,training=training)
        return y

    