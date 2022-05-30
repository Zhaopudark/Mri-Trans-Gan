import sys
import os
import tensorflow as tf
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)
from _build_helper import Reconstruction
from normalizations import SpectralNormalization as Sn
from activations import activation_slect
from initializers import initializer_slect
__all__ = [ 
    "Reshape",
    "Dense",
    "FlattenDense",
    "DenseVgg"
]
class Reshape(tf.keras.layers.Layer):
    def __init__(self,shape,name=None,dtype=None):
        """shape 不考虑batch维度的shape
        """
        super(Reshape,self).__init__(name=name,dtype=dtype)
        self.target_shape = shape
        self.reshape  = tf.keras.layers.Reshape(self.target_shape)
    def build(self,input_shape):
        super(Reshape,self).build(input_shape=input_shape)
        output_shape = [input_shape[0]]+self.target_shape
        return output_shape
    def call(self,x,training):
        return self.reshape(x)

class Dense(tf.keras.layers.Layer):
    def __init__(self,kernel_size,
                      use_bias=False,
                      activation=None,
                      kernel_initializer='glorot_uniform', 
                      bias_initializer='zeros',
                      spectral_normalization=False,
                      name=None,
                      dtype=None):
        super(Dense,self).__init__(name=name,dtype=dtype)
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.activation = activation_slect(activation)
        self.kernel_initializer = initializer_slect(kernel_initializer)
        self.bias_initializer = initializer_slect(bias_initializer)
        if spectral_normalization:
            self.Sn = Sn(dtype=dtype)
        self.spectral_normalization = spectral_normalization
    def build(self,input_shape):
        super(Dense,self).build(input_shape)
        self.w = self.add_weight('w',(input_shape[-1],self.kernel_size),initializer=self.kernel_initializer,trainable=True)
        if self.use_bias:
            self.b = self.add_weight('b',(self.kernel_size),initializer=self.bias_initializer,trainable=True)
        if self.spectral_normalization:
            self.Sn.build(input_shape=self.w.shape,w=self.w)
        output_shape = [input_shape[i] for i in range(len(input_shape)-1)]
        output_shape = output_shape+[self.kernel_size]
        return output_shape
    def call(self,x,training):
        if self.spectral_normalization:
            self.Sn(self.w,training)
        y = tf.matmul(x,self.w)
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return y
class FlattenDense(tf.keras.layers.Layer):
    def __init__(self,kernel_size,
                      use_bias=False,
                      activation=None,
                      kernel_initializer='glorot_uniform', 
                      bias_initializer='zeros',
                      spectral_normalization=False,
                      name=None,
                      dtype=None):
        super(FlattenDense,self).__init__(name=name,dtype=dtype)
        self.flatten = tf.keras.layers.Flatten(dtype=dtype)
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.activation = activation_slect(activation)
        self.kernel_initializer = initializer_slect(kernel_initializer)
        self.bias_initializer = initializer_slect(bias_initializer)
        if spectral_normalization:
            self.Sn = Sn(dtype=dtype)
        self.spectral_normalization = spectral_normalization
    def build(self,input_shape):
        super(FlattenDense,self).build(input_shape)
        buf = 1
        for i in range(1,len(input_shape),1): #Flatten_Dense 先进行flatten
            buf *= int(input_shape[i])
        input_shape = [input_shape[0],buf]
        self.w = self.add_weight('w',(input_shape[-1],self.kernel_size),initializer=self.kernel_initializer,trainable=True)
        if self.use_bias:
            self.b = self.add_weight('b',(self.kernel_size),initializer=self.bias_initializer,trainable=True)
        if self.spectral_normalization:
            self.Sn.build(input_shape=self.w.shape,w=self.w)
        output_shape = [input_shape[0]]+[self.kernel_size]
        return output_shape
    def call(self,x,training):
        x = self.flatten(x)
        if self.spectral_normalization:
            self.Sn(self.w,training)
        y = tf.matmul(x,self.w)
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return y
class DenseVgg(tf.keras.layers.Layer):
    def __init__(self,kernel_size,
                      use_bias=False,
                      activation=None,
                      kernel_initializer='glorot_uniform', 
                      bias_initializer='zeros',
                      name=None,
                      dtype=None):
        super(DenseVgg,self).__init__(name=name,dtype=dtype)
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.activation = activation_slect(activation)
        self.kernel_initializer = initializer_slect(kernel_initializer)
        self.bias_initializer = initializer_slect(bias_initializer)
    def build(self,input_shape):
        super(DenseVgg,self).build(input_shape)
        self.w = self.add_weight('w',(input_shape[-1],self.kernel_size),initializer=self.kernel_initializer,trainable=False)
        if self.use_bias:
            self.b = self.add_weight('b',(self.kernel_size),initializer=self.bias_initializer,trainable=False)
        output_shape = [input_shape[0]]+[self.kernel_size]
        return output_shape
    def call(self,x):
        y = tf.matmul(x,self.w)
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return tf.cast(y,dtype=tf.float32)  

if __name__ =="__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = Dense(10)
    x = tf.random.normal(shape=[128,100,10])
    y = model(x)
    print(y.shape,y.dtype)

    model = Reshape(shape=[50,20])
    out_shape = model.build(input_shape=[128,100,10])
    print(out_shape)
    y = model(x)
    print(y.shape,y.dtype)


