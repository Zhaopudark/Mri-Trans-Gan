"""
以下卷积 反卷积层 的设计 因为keras源码自带conv不支持反射 padding 所以需要对此进行融合
先行padding 再valid卷积 保持等价
"""
import sys
import os
import copy
import logging
from functools import partial,wraps
from typeguard import typechecked
from typing import Union,List,Tuple,Iterable
import tensorflow as tf
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)
sys.path.append(os.path.join(base,'../'))
sys.path.append(os.path.join(base,'../../'))
sys.path.append(os.path.join(base,'../../../'))
# from _build_helper import Reconstruction
# from normalizations import SpectralNormalization as Sn
# from activations import activation_slect
# from initializers import initializer_slect
from utils.convolution_helper import get_conv_paddings as get_conv_padding_vector
from utils.convolution_helper import get_padded_length_from_paddings as get_padded_length_from_padding_vector
from utils.convolution_helper import norm_paddings_by_data_format as norm_padding_vectors_by_data_format
from utils.convolution_helper import grab_length_by_data_format
__all__ = [
    "Conv2D",
    "Conv2DVgg",
    "Conv2DTranspose",
    "UpSampalingConv2D",
    # "SeparableConv2D",
]

def _normalize_padding(value):
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if padding not in {'valid', 'same', 'causal','full'}:
        raise ValueError('The `padding` argument must be a list/tuple or one of '
                        '"valid", "same", "full" (or "causal", only for `Conv1D). '
                        f'Received: {padding}')
    return padding
def _normalize_specific_padding_mode(value):
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if padding not in {'constant', 'reflect', 'symmetric'}:
        raise ValueError('The `padding` argument must be a list/tuple or one of '
                        '"constant", "reflect" or "symmetric"). '
                        f'Received: {padding}')
    return padding.upper()
# def _fix_call(call_func):
#     @wraps(call_func)
#     def fixd_call(self,inputs,**kwargs):
#         output = call_func(self,inputs,**kwargs)
#         return output
#     return fixd_call
class SpecificConvPad(tf.keras.layers.Wrapper):
    """
    Extend padding behavior of tf.keras.layers.Conv1D, tf.keras.layers.Conv2D 
    and tf.keras.layers.Conv3D. 
    Args:
      layer: A `tf.keras.layers.Conv1D`, `tf.keras.layers.Conv2D` or `tf.keras.layers.Conv3D` instance.
      padding_mode: Specific padding mode in `constant`, `reflect` or `symmetric`.
      padding_constant: Padding constant for `constant` padding_mode.
    Raises:
      ValueError: If not initialized with a `tf.keras.layers.Conv1D`, `tf.keras.layers.Conv2D` or `tf.keras.layers.Conv3D` instance.
      ValueError: If `padding_mode` does not in `constant`, `reflect` or `symmetric`.
      ValueError: If original `layers.padding` is `causal` but `padding_mode` is not `constant`.
    """
    @typechecked
    def __init__(self,
                layer:Union[tf.keras.layers.Conv1D,
                            tf.keras.layers.Conv2D,
                            tf.keras.layers.Conv3D,
                            ],
                padding_mode:Union[str,None]=None,
                padding_constant:int=0,
                **kwargs):
    
        self.padding_vectors = None
        self.padding_mode = padding_mode
        self.padding_constant = padding_constant

        if "name" not in kwargs.keys(): 
            kwargs["name"] = "specific_padded_"+layer.name
        super(SpecificConvPad,self).__init__(layer,**kwargs)
    
    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])
        # input_shape is needed 
        _kernel_size = self.layer.kernel_size
        _strides = self.layer.strides
        self._padding = _padding = self.layer.padding.lower() # save original padding for get_config and from_config
        _tf_data_format = self.layer._tf_data_format
        _dilation_rate = self.layer.dilation_rate
        _input_shape = grab_length_by_data_format(_tf_data_format,input_shape.as_list())

        if self.padding_mode is None:
            self.fused = True
        else: # padding_mode is str, confirmed by @typechecked
            if isinstance(_padding,(list,tuple)):
                self.fused = True
            else:
                if _padding in ["valid"]:
                    self.fused = True
                elif _padding in ["same","full","causal"]:
                    self.padding_mode = _normalize_specific_padding_mode(self.padding_mode)
                    if _padding == "causal":
                        if self.padding_mode != _normalize_specific_padding_mode("constant"):
                            raise ValueError("specific causal padding mode should only be CONSTANT not {}",format(self.padding_mode))
                    self.fused = False
                    _get_conv_padding_vector = partial(get_conv_padding_vector,padding=_padding)
                    _paddings = iter(list(map(_get_conv_padding_vector,_input_shape,_kernel_size,_strides,_dilation_rate)))
                    self.padding_vectors = norm_padding_vectors_by_data_format(_tf_data_format,_paddings)
                    self.layer.padding = _normalize_padding("valid")
                    self.layer._is_causal = self.layer.padding == _normalize_padding("causal") # causal is very special
        layer_input_shape = self._prefix_input_shape(input_shape)
        super().build(layer_input_shape)
    def _prefix_input_shape(self,input_shape):
        if not self.fused:
            input_shape = tf.TensorShape(input_shape).as_list()
            input_shape = copy.deepcopy(input_shape) 
            layer_input_shape = list(map(get_padded_length_from_padding_vector,input_shape,self.padding_vectors))
        else:
            layer_input_shape = input_shape
        return layer_input_shape
    def call(self,inputs,**kwargs):
        """Call `Layer`"""
        # For correct output when a layer have not been built, pad behavior must put here but not in build() func
        if not self.fused: 
            inputs = tf.pad(inputs,paddings=self.padding_vectors,mode=self.padding_mode,constant_values=self.padding_constant)
        output = self.layer(inputs,**kwargs)
        return output
    def get_config(self):
        config = {"padding_mode": self.padding_mode,
                  "padding_constant": self.padding_constant}
        base_config = super().get_config()
        if "layer" in base_config.keys():
            if "config" in base_config["layer"].keys():
                if "padding" in base_config["layer"]["config"].keys():
                    base_config["layer"]["config"]["padding"] = self._padding
        return dict(list(base_config.items())+list(config.items()))

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
# class Conv2DVgg(tf.keras.layers.Layer):
#     def __init__(self,
#                  filters=None,
#                  kernel_size=None,
#                  strides=None,
#                  padding=None,
#                  use_bias=False,
#                  kernel_initializer='glorot_uniform', 
#                  bias_initializer='zeros',
#                  activation=None,
#                  name=None,
#                  dtype=None,**kwargs):
#         """
#         接收VGG16 19 参数的2D卷积层
#         不支持谱范数正则
#         """
#         super(Conv2DVgg,self).__init__(name=name,dtype=dtype)
#         self.filters = filters#out put channels
#         self.kernel_size = kernel_size#[3,3]
#         if len(strides)!=2:
#             raise ValueError("2D conv need 2 dimensions for strides")
#         self.strides = [1]+strides+[1]
#         self.padding = padding.upper()
#         self.use_bias = use_bias
#         self.activation = activation_slect(activation)
#         self.kernel_initializer = initializer_slect(kernel_initializer)
#         self.bias_initializer = initializer_slect(bias_initializer)
#     def build(self,input_shape):
#         super(Conv2DVgg,self).build(input_shape=input_shape)
#         if len(input_shape)!=4:
#             raise ValueError("2D conv need 4 dimensions for input_shape")
#         output_shape,padding,padding_vect = Reconstruction.ConvCalculation(input_shape=input_shape,
#                                                                            filters=self.filters,
#                                                                            kernel_size=self.kernel_size,
#                                                                            strides=self.strides[1:-1],
#                                                                            padding=self.padding)
#         self.padding = padding
#         self.padding_vect = padding_vect  
#         self.w = self.add_weight('w',(self.kernel_size+[input_shape[-1]]+[self.filters]),initializer=self.kernel_initializer,trainable=False)
#         if self.use_bias:
#             self.b = self.add_weight('b',(output_shape[-1]),initializer=self.bias_initializer,trainable=False)
#         else:
#             pass 
#         output_shape = output_shape[:]
#         return output_shape
#     def call(self,x,training):
#         x = tf.pad(x,self.padding_vect,self.padding)
#         y = tf.nn.conv2d(input=x,filters=self.w,strides=self.strides,padding="VALID")#接管padding方式后 用valid实现等效
#         if self.use_bias:
#             y += self.b
#         y = self.activation(y)
#         return tf.cast(y,dtype=tf.float32)  
# #------------------------------------------------------------------------------------------------------------------------------#
# class UpSampalingConv2D(tf.keras.layers.Layer):
#     def __init__(self,
#                  filters=None,
#                  kernel_size=None,
#                  strides=None,
#                  padding=None,
#                  output_padding=None,
#                  use_bias=False,
#                  kernel_initializer='glorot_uniform', 
#                  bias_initializer='zeros',
#                  activation=None,
#                  spectral_normalization=False,
#                  name=None,
#                  dtype=None,**kwargs):
#         """
#         先将输入上采样到某个维度
#         根据需要进行padding
#         然后执行SAME卷积
#         实现反卷积黑盒
#         需要指定output_shape
#         """
#         super(UpSampalingConv2D,self).__init__(name=name,dtype=dtype)
#         self.filters = filters #out put channels
#         self.kernel_size = kernel_size #such as[3,3]
#         if len(strides)!=2:
#             raise ValueError("2D conv need 2 dimensions for strides")
#         self.strides = [1]+strides+[1]
#         self.up_size = self.strides[1:-1]
#         self.up_op = tf.keras.layers.UpSampling2D(size=self.up_size,dtype=dtype)#上采样只对非通道和非batch进行 剔除冗余维度
#         self.padding = padding.upper()
#         self.output_padding = output_padding #不实现该逻辑
#         self.use_bias = use_bias
#         self.activation = activation_slect(activation)
#         self.kernel_initializer = initializer_slect(kernel_initializer)
#         self.bias_initializer = initializer_slect(bias_initializer)
#         if spectral_normalization:
#             self.Sn = Sn(dtype=dtype,**kwargs)
#         self.spectral_normalization = spectral_normalization
#     def build(self,input_shape,output_shape=None):
#         super(UpSampalingConv2D,self).build(input_shape=input_shape)
#         if len(input_shape)!=4:
#             raise ValueError("2D conv need 4 dimensions for input_shape")
#         if output_shape == None:# 手动计算输出shape
#             output_shape = Reconstruction.ConvTransCal(input_shape=input_shape,filters=self.filters,kernel_size=self.kernel_size,strides=self.strides[1:-1],padding=self.padding)
#         if len(output_shape)!=4:
#             raise ValueError("2D conv need 4 dimensions for output_shape")
#         self.padding,self.padding_vect,self.cut_flag = Reconstruction.Trans2UpsampleCal(input_shape=input_shape[1:-1],
#                                                                                         output_shape=output_shape[1:-1],
#                                                                                         filters=self.filters,
#                                                                                         kernel_size=self.kernel_size,
#                                                                                         strides=self.strides[1:-1],
#                                                                                         padding=self.padding)
#         # 不同于反卷积, 上采样+卷积替代的反卷积时正向卷积 回到正向卷积的角度 [input_shape[-1]s]是in_channels [self.filters]是out_channels
#         self.w = self.add_weight('w',(self.kernel_size+[input_shape[-1]]+[self.filters]),initializer=self.kernel_initializer,trainable=True)
#         if self.use_bias:
#             self.b = self.add_weight('b',(output_shape[-1]),initializer=self.bias_initializer,trainable=True)
#         else:
#             pass 
#         if self.spectral_normalization:
#             self.Sn.build(input_shape=self.w.shape,w=self.w)
#         output_shape = output_shape[:]
#         return output_shape
#     def call(self,x,training):
#         x = self.up_op(x)
#         if self.cut_flag:
#             x = x[0-self.padding_vect[0][0]:int(x.shape[0])+self.padding_vect[0][1],
#                   0-self.padding_vect[1][0]:int(x.shape[1])+self.padding_vect[1][1],
#                   0-self.padding_vect[2][0]:int(x.shape[2])+self.padding_vect[2][1],
#                   0-self.padding_vect[3][0]:int(x.shape[3])+self.padding_vect[3][1]]
#         else:
#             x = tf.pad(x,self.padding_vect,"CONSTANT") 
#         if self.spectral_normalization:
#             self.Sn(self.w,training)
#         y = tf.nn.conv2d(input=x,filters=self.w,strides=[1,1,1,1],padding=self.padding)#接管stride恒定为1
#         if self.use_bias:
#             y += self.b
#         y = self.activation(y)
#         return y
# #------------------------------------------------------------------------------------------------------------------------------#
# class SeparableConv2D(tf.keras.layers.SeparableConv2D):
#     """
#         Copy from the tf.keras.layers.SeparableConv2D
#         input:...see the reference
#         output:...see the reference
#         reference:https://tensorflow.google.cn/api_docs/python/tf/keras/layers/SeparableConv2D
#     """
#     def __init__(self,*args,**kwargs):
#         super(SeparableConv2D,self).__init__(*args,**kwargs)
if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.experimental.enable_op_determinism()

    input_shape = 15
    dtype = "float32"
    kernel_size = 1
    stride = 3
    padding = "same"
    dilation_rate = 1
    dim = 3
    input_shape = [8]+[input_shape,]*dim+[3]
    tf.keras.utils.set_random_seed(1000)
    x = tf.random.normal(shape=input_shape,seed=1000,dtype=dtype)

    tf.keras.utils.set_random_seed(1000)
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
    conv3d = tf.keras.layers.Conv3D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
    y = conv3d(x)

    tf.keras.utils.set_random_seed(1000)
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
    conv3d_ = tf.keras.layers.Conv3D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
    conv3d_2 = SpecificConvPad(conv3d_,padding_mode="constant",padding_constant=0)
    y_ = conv3d_2(x)

    assert y.shape==y_.shape
    computed = tf.reduce_mean(y-y_)
    print(computed)