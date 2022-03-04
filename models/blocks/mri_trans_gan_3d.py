__all__ = [
    "Conv7S1",
    "DownSampling",
    "ResBlocks",
    "UpSampling",    
    "Conv4S2",
    "Conv4S1", 
]
import sys
import os
import logging
import tensorflow as tf
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base,'../'))
from _layers.convolutions import Conv3D,UpSampalingConv3D,UpSubpixelConv3D,ConvPadConcretization
from _layers.normalizations import InstanceNormalization
from craft.activations import Activation
"""
Cycle GAN Generator blocks
"""
#--------------------------------------------------------------------------------------------#
class Conv7S1(tf.keras.layers.Layer):#c7s1_k
    def __init__(self,
                 filters,
                 spectral_normalization=False,
                 use_bias=False,
                 activation="relu",
                 specific_out_dtype=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(Conv7S1,self).__init__(name=name,dtype=dtype)
        l1_conv = Conv3D(filters=filters,
                              kernel_size=[7,7,7],
                              strides=[1,1,1],
                              padding="same",
                              use_bias=use_bias,
                              dtype=dtype)
        self.l1_conv = ConvPadConcretization(l1_conv,padding_mode="reflect")                   
        self.l2_norm = InstanceNormalization(dtype=dtype)
        if specific_out_dtype is None:
            self.l3_activation = Activation(activation,dtype=dtype)
        else:
            self.l3_activation = Activation(activation,dtype=specific_out_dtype,**kwargs)
             
    def build(self,input_shape):
        super().build(input_shape)
        flow_shape=self.l1_conv.compute_output_shape(input_shape)
        flow_shape=self.l2_norm.compute_output_shape(flow_shape)
        # flow_shape=self.l3_activation.compute_output_shape(flow_shape)
    def call(self,x,training=True):
        x = self.l1_conv(x,training=training)
        x = self.l2_norm(x,training=training)
        x = x*1.0
        y = self.l3_activation(x,training=training)
        return y
#--------------------------------------------------------------------------------------------#
class DownSampling(tf.keras.layers.Layer):#dk
    def __init__(self,
                 filters,
                 spectral_normalization=False,
                 use_bias=False,
                 specific_out_dtype=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(DownSampling,self).__init__(name=name,dtype=dtype)
        l1_conv = Conv3D(filters=filters,
                              kernel_size=[3,3,3],
                              strides=[2,2,2],
                              padding="same",
                              use_bias=use_bias,
                              dtype=dtype)
        self.l1_conv = ConvPadConcretization(l1_conv,padding_mode="reflect")   
        self.l2_norm = InstanceNormalization(dtype=dtype)
        if specific_out_dtype is None:
            self.l3_activation = Activation("relu",dtype=dtype)
        else:
            self.l3_activation = Activation("relu",dtype=specific_out_dtype)
    def build(self,input_shape):
        super().build(input_shape)
        self.l1_conv.build(input_shape)
        flow_shape=self.l1_conv.compute_output_shape(input_shape)
        flow_shape=self.l2_norm.compute_output_shape(flow_shape)
        # flow_shape=self.l3_activation.compute_output_shape(flow_shape)
    def call(self,x,training=True):
        x = self.l1_conv(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y
#--------------------------------------------------------------------------------------------#
class ResBlock(tf.keras.layers.Layer):#rk
    def __init__(self,
                 filters,
                 spectral_normalization=False,
                 use_bias=False,
                 specific_out_dtype=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(ResBlock,self).__init__(name=name,dtype=dtype)
        l1_conv= Conv3D(filters=filters,
                             kernel_size=[3,3,3],
                             strides=[1,1,1],
                             padding="same",
                             use_bias=use_bias,
                             dtype=dtype)
        self.l1_conv = ConvPadConcretization(l1_conv,padding_mode="reflect") 
        self.l2_norm = InstanceNormalization(dtype=dtype)
        self.l3_activation =Activation("relu",dtype=dtype)
        l4_conv= Conv3D(filters=filters,
                             kernel_size=[3,3,3],
                             strides=[1,1,1],
                             padding="same",
                             use_bias=use_bias,
                             dtype=dtype)
        self.l4_conv = ConvPadConcretization(l4_conv,padding_mode="reflect") 
        self.l5_norm = InstanceNormalization(dtype=dtype)
        if specific_out_dtype is None:
            self.l6_activation = Activation("linear",dtype=dtype)
        else:
            self.l6_activation = Activation("linear",dtype=specific_out_dtype)
    def build(self,input_shape):
        super().build(input_shape)
        flow_shape=self.l1_conv.compute_output_shape(input_shape)
        flow_shape=self.l2_norm.compute_output_shape(flow_shape)
        flow_shape=self.l4_conv.compute_output_shape(flow_shape)
        flow_shape=self.l5_norm.compute_output_shape(flow_shape)
    def call(self,x,training=True):
        y = self.l1_conv(x,training=training)
        y = self.l2_norm(y,training=training)
        y = self.l3_activation(y,training=training)
        y = self.l4_conv(y,training=training)
        y = self.l5_norm(y,training=training)
        y = self.l6_activation(y+x,training=training)
        return y
#--------------------------------------------------------------------------------------------#
class ResBlocks(tf.keras.layers.Layer):#rks
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
                                         use_bias=use_bias,
                                         specific_out_dtype=specific_out_dtype,
                                         dtype=dtype))
    def build(self,input_shape):
        super().build(input_shape)
        flow_shape = input_shape
        for item in self.rs_list:
            flow_shape=item.compute_output_shape(flow_shape)
    def call(self,x,training=True):
        in_put = x
        for item in self.rs_list:
            out_put = item(in_put,training=training)
            in_put = out_put
        return out_put
#--------------------------------------------------------------------------------------------#      
class UpSampling(tf.keras.layers.Layer):#uk
    def __init__(self,
                 filters,
                 spectral_normalization=False,
                 use_bias=False,specific_out_dtype=None,
                 name=None,
                 dtype=None,
                 up_sampling_method="up_conv",
                 **kwargs):
        super(UpSampling,self).__init__(name=name,dtype=dtype)
        if up_sampling_method == "up_conv":
            self.l1_up = UpSampalingConv3D(filters=filters,
                                           kernel_size=[3,3,3],
                                           size=[2,2,2],
                                           strides=[1,1,1],
                                           padding="SAME",
                                           use_bias=use_bias,
                                           dtype=dtype)
        elif up_sampling_method == "sub_pixel_up":
            logging.warning("UpSampling has been replaced by UpSubpixelConv3D")
            self.l1_up = UpSubpixelConv3D(filters=filters,
                                          kernel_size=[3,3,3],
                                          strides=[2,2,2],
                                          padding="SAME",
                                          use_bias=use_bias,
                                          dtype=dtype)
        else:
            raise ValueError("Unsupported up_sampling_method {}!".format(up_sampling_method))
        self.l2_norm = InstanceNormalization(dtype=dtype)
        if specific_out_dtype is None:
            self.l3_activation = Activation("relu",dtype=dtype)
        else:
            self.l3_activation = Activation("relu",dtype=specific_out_dtype)
    def build(self,input_shape):
        super().build(input_shape)
        flow_shape=self.l1_up.compute_output_shape(input_shape)
        flow_shape=self.l2_norm.compute_output_shape(flow_shape)
        # flow_shape=self.l3_activation.compute_output_shape(flow_shape)
    def call(self,x,training=True):
        x = self.l1_up(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y
"""
CycleGAN Discriminator blocks
"""
#--------------------------------------------------------------------------------------------#
class Conv4S2(tf.keras.layers.Layer):#ck
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
                             dtype=dtype)
        if norm:
            self.l2_norm = InstanceNormalization(dtype=dtype)
        else:
            self.l2_norm = Activation("linear",dtype=dtype)
        if specific_out_dtype is None:
            self.l3_activation = Activation("leaky_relu",alpha=0.2,dtype=dtype)
        else:
            self.l3_activation = Activation("leaky_relu",alpha=0.2,dtype=specific_out_dtype)
    def build(self,input_shape):
        super().build(input_shape)
        flow_shape=self.l1_conv.compute_output_shape(input_shape)
        flow_shape=self.l2_norm.compute_output_shape(flow_shape)
        # flow_shape=self.l3_activation.compute_output_shape(flow_shape)
    def call(self,x,training=True):
        x = self.l1_conv(x,training=training)
        x = self.l2_norm(x,training=training)
        y = self.l3_activation(x,training=training)
        return y
#--------------------------------------------------------------------------------------------#
class Conv4S1(tf.keras.layers.Layer):#ck last
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
                               dtype=dtype)
        if use_sigmoid:
            tmp_activation_name = "sigmoid"
        else:
            tmp_activation_name = "linear"
        if specific_out_dtype is None:
            self.l2_activation = Activation(tmp_activation_name,dtype=dtype)
        else:
            self.l2_activation = Activation(tmp_activation_name,dtype=specific_out_dtype)
    def build(self,input_shape):
        super().build(input_shape)
        flow_shape=self.l1_conv.compute_output_shape(input_shape)
        # flow_shape=self.l2_activation.compute_output_shape(flow_shape)
    def call(self,x,training=True):
        x = self.l1_conv(x,training=training)
        y = self.l2_activation(x,training=training)
        return y    
if __name__=="__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    c7s1 = Conv7S1(32)
    c7s1.build(input_shape=[1,16,128,128,1])
    print(c7s1.l2_norm.trainable_variables)