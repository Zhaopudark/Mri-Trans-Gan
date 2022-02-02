"""
以下卷积 反卷积层 的设计 因为keras源码自带conv不支持反射 padding 所以需要对此进行融合
先行padding 再valid卷积 保持等价
"""
import sys
import os
import tensorflow as tf
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)
sys.path.append(os.path.join(base,'../'))
from _build_helper import Reconstruction
from normalizations import SpectralNormalization as Sn
from activations import activation_slect
from initializers import initializer_slect

__all__ = [
    "Conv2D",
    "Conv2DVgg",
    "Conv2DTranspose",
    "UpSampalingConv2D",
    # "SeparableConv2D",
]


class ConvPad(tf.keras.layers.Wrapper):
    def __init__(self,layer,*args,**kwargs):
        if not isinstance(layer,tf.keras.layers.Layer):
            raise ValueError('Please initialize `Bidirectional` layer with a 'f'`tf.keras.layers.Layer` instance. Received: {layer}')
        if "padding" in kwargs.keys():
            if kwargs["padding"] is None:
                pass 
            elif kwargs["padding"].lower() == "valid":
                pass 
            elif kwargs["padding"].lower() == "same":
                pass 
            elif kwargs["padding"].lower() == "reflect": 
        super(ConvPad,self).__init__(layer,*args,**kwargs)# get self.layer


    def build(self, input_shape):
        """Build `Layer`"""
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])
        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(
                "{} object has no attribute 'kernel' nor "
                "'embeddings'".format(type(self.layer).__name__)
            )
        self.w_shape = self.w.shape.as_list()
        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="sn_u",
            dtype=self.w.dtype,
        )
    def call(self,inputs,training=None):
        """Call `Layer`"""
        if training is None:
            training = tf.keras.backend.learning_phase()
        if training:
            self.normalize_weights()
        tf.print(training)
        output = self.layer(inputs)
        return output
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())
    def normalize_weights(self):
        """Generate spectral normalized weights.
        This method will update the value of `self.w` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """
        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u
        with tf.name_scope("spectral_normalize"):
            for _ in range(self.power_iterations):
                v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
                u = tf.math.l2_normalize(tf.matmul(v, w))
            u = tf.stop_gradient(u)
            v = tf.stop_gradient(v)
            sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
            self.u.assign(tf.cast(u, self.u.dtype))
            self.w.assign(
                tf.cast(tf.reshape(self.w / sigma, self.w_shape), self.w.dtype)
            )
    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}

class Conv2D(tf.keras.layers.Conv2D):
    def __init__(self,*args,**kwargs):
        """
        Copy from the tf.keras.layers.Conv2D
        input:...see the reference
        output:...see the reference
        reference:https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv2D
        """
        super(Conv2D,self).__init__(*args,**kwargs)


class Conv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self,*args,**kwargs):
        """
        Copy from the tf.keras.layers.Conv2DTranspose
        input:...see the reference
        output:...see the reference
        reference:https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv2DTranspose
        """
        super(Conv2D,self).__init__(*args,**kwargs)
#------------------------------------------------------------------------------------------------------------------------------#
class Conv2DVgg(tf.keras.layers.Layer):
    def __init__(self,
                 filters=None,
                 kernel_size=None,
                 strides=None,
                 padding=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros',
                 activation=None,
                 name=None,
                 dtype=None,**kwargs):
        """
        接收VGG16 19 参数的2D卷积层
        不支持谱范数正则
        """
        super(Conv2DVgg,self).__init__(name=name,dtype=dtype)
        self.filters = filters#out put channels
        self.kernel_size = kernel_size#[3,3]
        if len(strides)!=2:
            raise ValueError("2D conv need 2 dimensions for strides")
        self.strides = [1]+strides+[1]
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.activation = activation_slect(activation)
        self.kernel_initializer = initializer_slect(kernel_initializer)
        self.bias_initializer = initializer_slect(bias_initializer)
    def build(self,input_shape):
        super(Conv2DVgg,self).build(input_shape=input_shape)
        if len(input_shape)!=4:
            raise ValueError("2D conv need 4 dimensions for input_shape")
        output_shape,padding,padding_vect = Reconstruction.ConvCalculation(input_shape=input_shape,
                                                                           filters=self.filters,
                                                                           kernel_size=self.kernel_size,
                                                                           strides=self.strides[1:-1],
                                                                           padding=self.padding)
        self.padding = padding
        self.padding_vect = padding_vect  
        self.w = self.add_weight('w',(self.kernel_size+[input_shape[-1]]+[self.filters]),initializer=self.kernel_initializer,trainable=False)
        if self.use_bias:
            self.b = self.add_weight('b',(output_shape[-1]),initializer=self.bias_initializer,trainable=False)
        else:
            pass 
        output_shape = output_shape[:]
        return output_shape
    def call(self,x,training):
        x = tf.pad(x,self.padding_vect,self.padding)
        y = tf.nn.conv2d(input=x,filters=self.w,strides=self.strides,padding="VALID")#接管padding方式后 用valid实现等效
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return tf.cast(y,dtype=tf.float32)  
#------------------------------------------------------------------------------------------------------------------------------#
class UpSampalingConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters=None,
                 kernel_size=None,
                 strides=None,
                 padding=None,
                 output_padding=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros',
                 activation=None,
                 spectral_normalization=False,
                 name=None,
                 dtype=None,**kwargs):
        """
        先将输入上采样到某个维度
        根据需要进行padding
        然后执行SAME卷积
        实现反卷积黑盒
        需要指定output_shape
        """
        super(UpSampalingConv2D,self).__init__(name=name,dtype=dtype)
        self.filters = filters #out put channels
        self.kernel_size = kernel_size #such as[3,3]
        if len(strides)!=2:
            raise ValueError("2D conv need 2 dimensions for strides")
        self.strides = [1]+strides+[1]
        self.up_size = self.strides[1:-1]
        self.up_op = tf.keras.layers.UpSampling2D(size=self.up_size,dtype=dtype)#上采样只对非通道和非batch进行 剔除冗余维度
        self.padding = padding.upper()
        self.output_padding = output_padding #不实现该逻辑
        self.use_bias = use_bias
        self.activation = activation_slect(activation)
        self.kernel_initializer = initializer_slect(kernel_initializer)
        self.bias_initializer = initializer_slect(bias_initializer)
        if spectral_normalization:
            self.Sn = Sn(dtype=dtype,**kwargs)
        self.spectral_normalization = spectral_normalization
    def build(self,input_shape,output_shape=None):
        super(UpSampalingConv2D,self).build(input_shape=input_shape)
        if len(input_shape)!=4:
            raise ValueError("2D conv need 4 dimensions for input_shape")
        if output_shape == None:# 手动计算输出shape
            output_shape = Reconstruction.ConvTransCal(input_shape=input_shape,filters=self.filters,kernel_size=self.kernel_size,strides=self.strides[1:-1],padding=self.padding)
        if len(output_shape)!=4:
            raise ValueError("2D conv need 4 dimensions for output_shape")
        self.padding,self.padding_vect,self.cut_flag = Reconstruction.Trans2UpsampleCal(input_shape=input_shape[1:-1],
                                                                                        output_shape=output_shape[1:-1],
                                                                                        filters=self.filters,
                                                                                        kernel_size=self.kernel_size,
                                                                                        strides=self.strides[1:-1],
                                                                                        padding=self.padding)
        # 不同于反卷积, 上采样+卷积替代的反卷积时正向卷积 回到正向卷积的角度 [input_shape[-1]s]是in_channels [self.filters]是out_channels
        self.w = self.add_weight('w',(self.kernel_size+[input_shape[-1]]+[self.filters]),initializer=self.kernel_initializer,trainable=True)
        if self.use_bias:
            self.b = self.add_weight('b',(output_shape[-1]),initializer=self.bias_initializer,trainable=True)
        else:
            pass 
        if self.spectral_normalization:
            self.Sn.build(input_shape=self.w.shape,w=self.w)
        output_shape = output_shape[:]
        return output_shape
    def call(self,x,training):
        x = self.up_op(x)
        if self.cut_flag:
            x = x[0-self.padding_vect[0][0]:int(x.shape[0])+self.padding_vect[0][1],
                  0-self.padding_vect[1][0]:int(x.shape[1])+self.padding_vect[1][1],
                  0-self.padding_vect[2][0]:int(x.shape[2])+self.padding_vect[2][1],
                  0-self.padding_vect[3][0]:int(x.shape[3])+self.padding_vect[3][1]]
        else:
            x = tf.pad(x,self.padding_vect,"CONSTANT") 
        if self.spectral_normalization:
            self.Sn(self.w,training)
        y = tf.nn.conv2d(input=x,filters=self.w,strides=[1,1,1,1],padding=self.padding)#接管stride恒定为1
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return y
#------------------------------------------------------------------------------------------------------------------------------#
class SeparableConv2D(tf.keras.layers.SeparableConv2D):
    """
        Copy from the tf.keras.layers.SeparableConv2D
        input:...see the reference
        output:...see the reference
        reference:https://tensorflow.google.cn/api_docs/python/tf/keras/layers/SeparableConv2D
    """
    def __init__(self,*args,**kwargs):
        super(SeparableConv2D,self).__init__(*args,**kwargs)