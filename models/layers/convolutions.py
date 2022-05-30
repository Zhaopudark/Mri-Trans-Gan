"""
以下卷积 反卷积层 的设计 因为keras源码自带conv不支持反射 padding 所以需要对此进行融合
先行padding 再valid卷积 保持等价
"""
import copy
import logging
from functools import partial,wraps
from typeguard import typechecked

import tensorflow as tf

from models.layers._build_helper import Reconstruction
# from normalizations import SpectralNormalization as Sn
from models.layers.activations import activation_slect
from models.layers.initializers import initializer_slect
from utils.convolution_helper import get_conv_paddings as get_conv_padding_vector
from utils.convolution_helper import get_padded_length_from_paddings as get_padded_length_from_padding_vector
from utils.convolution_helper import normalize_paddings_by_data_format as normalize_padding_vectors_by_data_format
from utils.convolution_helper import grab_length_by_data_format
from utils.convolution_helper import normalize_padding
from utils.convolution_helper import normalize_specific_padding_mode
from utils.convolution_helper import normalize_tuple


__all__ = [
    'Conv2D',
    'Conv2DVgg',
    'Conv2DTranspose',
    'UpSampalingConv2D',
    # 'SeparableConv2D',
]

# def _fix_call(call_func):
#     @wraps(call_func)
#     def fixd_call(self,inputs,**kwargs):
#         output = call_func(self,inputs,**kwargs)
#         return output
#     return fixd_call
#------------------------------------------------------------------------------------------------------------------------------#
class ConvPadConcretization(tf.keras.layers.Wrapper):
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
                layer:tf.keras.layers.Conv1D|tf.keras.layers.Conv2D|tf.keras.layers.Conv3D,
                padding_mode:str|None=None,
                padding_constant:int=0,
                **kwargs):
        self.padding_vectors = None
        self.padding_mode = padding_mode
        self.padding_constant = padding_constant

        if 'name' not in kwargs.keys(): 
            kwargs['name'] = f"specific_padded_{layer.name}"
        super(ConvPadConcretization,self).__init__(layer,**kwargs)
    
    def build(self, input_shape):
        """Build `Layer`"""
        self._build_input_shape = input_shape # if superclass is not tf.keras.layers.Layer directly
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
                if _padding in ['valid']:
                    self.fused = True
                elif _padding in ['same','full','causal']:
                    self.padding_mode = normalize_specific_padding_mode(self.padding_mode)
                    if _padding == 'causal':
                        if self.padding_mode != normalize_specific_padding_mode('constant'):
                            raise ValueError("specific causal padding mode should only be CONSTANT not {}.",format(self.padding_mode))
                    self.fused = False
                    _get_conv_padding_vector = partial(get_conv_padding_vector,padding=_padding)
                    _padding_vectors = iter(list(map(_get_conv_padding_vector,_input_shape,_kernel_size,_strides,_dilation_rate)))
                    self.padding_vectors = normalize_padding_vectors_by_data_format(_tf_data_format,_padding_vectors)
                    self.layer.padding = normalize_padding('valid')
                    self.layer._is_causal = self.layer.padding == normalize_padding('causal') # causal is very special
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
    def call(self,inputs,*args,**kwargs):
        """Call `Layer`"""
        # For correct output when a layer have not been built, pad behavior must put here but not in build() func
        if not self.fused: 
            inputs = tf.pad(inputs,paddings=self.padding_vectors,mode=self.padding_mode,constant_values=self.padding_constant)
        return self.layer(inputs,*args,**kwargs)
    def get_config(self):
        config = {'padding_mode': self.padding_mode,
                  'padding_constant': self.padding_constant}
        base_config = super().get_config()
        if 'layer' in base_config.keys():
            if 'config' in base_config['layer'].keys():
                if 'padding' in base_config['layer']['config'].keys():
                    base_config['layer']['config']['padding'] = self._padding
        return base_config|config
#------------------------------------------------------------------------------------------------------------------------------#
class Conv1D(tf.keras.layers.Conv1D):
    def __init__(self,*args,**kwargs):
        """
        Simply wrap the tf.keras.layers.ConvXD, where ConvXD is Conv1D, Conv2D or Conv3D.
        input:...see the reference
        output:...see the reference
        reference:https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv1D
        """
        super(Conv1D,self).__init__(*args,**kwargs)
#------------------------------------------------------------------------------------------------------------------------------#
class Conv1DTranspose(tf.keras.layers.Conv1DTranspose):
    def __init__(self,*args,**kwargs):
        """
        Simply wrap the tf.keras.layers.ConvXDTranspose, where ConvXDTranspose is Conv1DTranspose, Conv2DTranspose or Conv3DTranspose.
        input:...see the reference
        output:...see the reference
        reference:https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv1DTranspose
        """
        super(Conv1DTranspose,self).__init__(*args,**kwargs)
#------------------------------------------------------------------------------------------------------------------------------#
class Conv2D(tf.keras.layers.Conv2D):
    def __init__(self,*args,**kwargs):
        """
        Simply wrap the tf.keras.layers.ConvXD, where ConvXD is Conv1D, Conv2D or Conv3D.
        input:...see the reference
        output:...see the reference
        reference:https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv2D
        """
        super(Conv2D,self).__init__(*args,**kwargs)
#------------------------------------------------------------------------------------------------------------------------------#
class Conv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self,*args,**kwargs):
        """
        Simply wrap the tf.keras.layers.ConvXDTranspose, where ConvXDTranspose is Conv1DTranspose, Conv2DTranspose or Conv3DTranspose.
        input:...see the reference
        output:...see the reference
        reference:https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv2DTranspose
        """
        super(Conv2DTranspose,self).__init__(*args,**kwargs)
#------------------------------------------------------------------------------------------------------------------------------#
class Conv3D(tf.keras.layers.Conv3D):
    def __init__(self,*args,**kwargs):
        """
        Simply wrap the tf.keras.layers.ConvXD, where ConvXD is Conv1D, Conv2D or Conv3D.
        input:...see the reference
        output:...see the reference
        reference:https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv3D
        """
        super(Conv3D,self).__init__(*args,**kwargs)
#------------------------------------------------------------------------------------------------------------------------------#
class Conv3DTranspose(tf.keras.layers.Conv3DTranspose):
    def __init__(self,*args,**kwargs):
        """
        Simply wrap the tf.keras.layers.ConvXDTranspose, where ConvXDTranspose is Conv1DTranspose, Conv2DTranspose or Conv3DTranspose.
        input:...see the reference
        output:...see the reference
        reference:https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv3DTranspose
        """
        super(Conv3DTranspose,self).__init__(*args,**kwargs)
#------------------------------------------------------------------------------------------------------------------------------#
class GroupConv1D(tf.keras.layers.Conv1D):
    @typechecked
    def __init__(self,*args,groups:int,**kwargs):
        """
        Group Convolution has been supported in tf.keras.layers.Conv, where Conv is Conv1D, Conv2D or Conv3D.
        This function is a simpler wrapper around tf.keras.layers.ConvXD
        """
        super(GroupConv1D,self).__init__(*args,groups=groups,**kwargs)
#------------------------------------------------------------------------------------------------------------------------------#
class GroupConv2D(tf.keras.layers.Conv2D):
    @typechecked
    def __init__(self,*args,groups:int,**kwargs):
        """
        Group Convolution has been supported in tf.keras.layers.Conv, where Conv is Conv1D, Conv2D or Conv3D.
        This function is a simpler wrapper around tf.keras.layers.ConvXD
        """
        super(GroupConv2D,self).__init__(*args,groups=groups,**kwargs)
#------------------------------------------------------------------------------------------------------------------------------#
class GroupConv3D(tf.keras.layers.Conv3D):
    @typechecked
    def __init__(self,*args,groups:int,**kwargs):
        """
        Group Convolution has been supported in tf.keras.layers.Conv, where Conv is Conv1D, Conv2D or Conv3D.
        This function is a simpler wrapper around tf.keras.layers.ConvXD
        """
        super(GroupConv3D,self).__init__(*args,groups=groups,**kwargs)
#------------------------------------------------------------------------------------------------------------------------------#
class UpSampalingConv(tf.keras.layers.Layer):
    @typechecked
    def __init__(self,
                 rank:int,
                 filters:int,
                 kernel_size:tuple|list|int,
                 size:tuple|list|int,
                 interpolation:str='nearest',
                 strides:tuple|list|int=1,
                 padding:str='same',
                 data_format:str='channels_last',
                 dilation_rate:tuple|list|int=1,
                 groups:int=1,
                 activation=None,
                 use_bias:bool=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        This layer combine UpSampling and Conv together, to to mimic ConvTranspose in shape-wise.
        It's a good idea to  avoid `Checkerboard Artifacts` of traditional deconvolution method.

        See the https://distill.pub/2016/deconv-checkerboard/
        This layer's operation can be divedie into 2 procedure, i.e., `UpSampling+Conv`
        We should make Conv's stride "=1", and give wranings if dilation_rate ">1" 
        Here's the explanation:
        Frist, if we make Conv's stride ">1", the former `UpSampling` will make no sense.
        Second, if we make Conv's dilation_rate ">1", consider the following 3 procedures:
            procedure 1, conv1d_transpose on input [a,b,c] with kernel [x,y,z], strides = 2,
                equivalent matrix representation is :
                    [a,b,c]@[[x,y,z,0,0,0,],
                             [0,0,x,y,z,0,],   
                             [0,0,0,0,x,y,],]
                    = [[ax,by,az+bx,by,bz+cx,cy]]
                finally got a output with elements different from each other.
            procedure 2: UpSampalingConv1D on input [a,b,c] with kernel [x,y,z], dilation_rate=1
            'nearest' interpolation and size=(2,)
                equivalent matrix representation is :
                    [a,a,b,b,c,c]@[[x,0,0,0,0,0,],
                                   [y,x,0,0,0,0,],   
                                   [z,y,x,0,0,0,],
                                   [0,z,y,x,0,0,],
                                   [0,0,z,y,x,0,],
                                   [0,0,0,z,y,x,],]
                    = [[ax+ay+bz,ax+by+bz,bx+by+cz,bx+cy+cz,cx+cy,cz]]
                finally got a output with elements different from each other.
            procedure 3: UpSampalingConv1D on input [a,b,c] with kernel [x,y,z], dilation_rate=2
            'nearest' interpolation and size=(2,)
                equivalent matrix representation is :
                    [a,a,b,b,c,c]@[[x,0,0,0,0,0,],
                                   [0,x,0,0,0,0,],   
                                   [y,0,x,0,0,0,],
                                   [0,y,0,x,0,0,],
                                   [z,0,y,0,x,0,],
                                   [0,z,0,y,0,x,],]
                    = [[ax+by+cz,ax+by+cz,bx+cy,bx+cy,cx,cx]]
                finally got a output with same elements in each two columns.
                So, this output has the same amount of information as input, far from 
                original purpose of deconv(ConvTranspose) or UpSampalingConv(ResizeConv)
            These 3 procedures show that, in some cases of certain parameters, if dilation_rate ">1",
            UpSampalingConv will make no sense. But in other cases, such as with 'bilinear' interpolation 
            and dilation_rate ">1", UpSampalingConv can also output a desired tensor with elements different from each other.
            So, it is better to give out a warning, instead of set mandatory restriction.
        Args: 
            size: represent `UpSampling` operation's up size.

            interpolation: Specify interpolation method for `UpSampling` operation. 
            NOTE Currently, `UpSampling1D` and  `UpSampling3D` only support 'nearest' 
            interpolation `UpSampling2D` only  support 'nearest' or 'bilinear' interpolation

            output_padding: This arg works as same as in tf.keras.layer.Conv1DTranspose, since 
            this layer should mimic ConvTranspose in shape-wise.

            Other args are same with tf.keras.layer.Conv1D, will control the `Conv` operation. 
            NOTE Stride in each dim should be 1.
            If dilation_rate is set ">1", a warning will give out to remind user to be careful
            on the process.
        Raises:
            ValueError: If output_length of 'UpSampling and padding Conv' are not equal to the mimicked
            ConvTranspose's output_length
        References:
            - [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)
            - [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1)
            - [Deconvolutional Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
        """
        super(UpSampalingConv,self).__init__(**kwargs)
        self._rank = _rank = rank
        _upsampling_kwargs = {} # collect kwargs for `UpSampling`, here is for get_config()
        # if _rank <= 0 or _rank>3:
        #     raise ValueError(f"rank should be a integer of 1 or 2 or 3, not {_rank}.") #NOTE there is no need because this Class is not open to user
        _upsampling_kwargs['size'] = normalize_tuple(size,_rank,'size')
        _upsampling_kwargs['data_format'] = data_format # upsampling and convolution share a data_format
        _interpolation = interpolation.lower()
        if (_interpolation not in ['nearest','bilinear'])or(_rank!=2 and _interpolation != 'nearest'):
            raise ValueError(" Currently, `UpSampling1D` and  `UpSampling3D` only support 'nearest' "
            f"interpolation, `UpSampling2D` only  support `nearest` or `bilinear` interpolation, but not`{interpolation}`")           
        _upsampling_kwargs['interpolation'] = _interpolation
        self._upsampling_kwargs = copy.deepcopy(_upsampling_kwargs)
        
        _convolution_kwargs={} # collect kwargs for `Conv`, here is for get_config()
        _convolution_kwargs['filters'] = filters
        _convolution_kwargs['kernel_size'] = kernel_size
        _strides = normalize_tuple(strides,_rank,'strides')
        if any([x!=1 for x in _strides]):
            raise ValueError(f"UpSampalingConv's strides should be 1, not {strides}.")
        _convolution_kwargs['strides'] = _strides
        _padding = padding.lower()
        if _padding != 'same':
            raise ValueError(f"UpSampalingConv's conv padding should be `same`, not `{padding}`.")
        _convolution_kwargs['padding'] = _padding
        _convolution_kwargs['data_format'] = data_format # upsampling and convolution share a data_format
        dilation_rate = normalize_tuple(dilation_rate,_rank,'dilation_rate')
        if any([x!=1 for x in dilation_rate]):
            logging.getLogger(__name__).warning("UpSampalingConv ususally runs with dilation_rate==1, please be careful whether "
                f"the operation is your desired one when dilation_rate={dilation_rate}.")
        _convolution_kwargs['dilation_rate'] = dilation_rate
        _convolution_kwargs['groups'] = groups
        _convolution_kwargs['activation'] = activation
        _convolution_kwargs['use_bias'] = use_bias
        _convolution_kwargs['kernel_initializer'] = kernel_initializer
        _convolution_kwargs['bias_initializer'] = bias_initializer
        _convolution_kwargs['kernel_regularizer'] = kernel_regularizer
        _convolution_kwargs['bias_regularizer'] = bias_regularizer
        _convolution_kwargs['activity_regularizer'] = activity_regularizer
        _convolution_kwargs['kernel_constraint'] = kernel_constraint
        _convolution_kwargs['bias_constraint'] = bias_constraint

        self._convolution_kwargs = copy.deepcopy(_convolution_kwargs)
        _convolution_kwargs['strides'] = 1
        _convolution_kwargs['padding'] = 'same'
        if _rank == 1:
            _upsampling_kwargs['size'] = _upsampling_kwargs['size'][0] # Since UpSampling1D's bug, it can not receive any type of 'size', except integer
            _upsampling_kwargs.pop('interpolation')
            _upsampling_kwargs.pop('data_format')

            self.up_sampling = tf.keras.layers.UpSampling1D(**_upsampling_kwargs,**kwargs) 
            self.convolution = tf.keras.layers.Conv1D(**_convolution_kwargs,**kwargs)
        elif _rank == 2:
            self.up_sampling = tf.keras.layers.UpSampling2D(**_upsampling_kwargs,**kwargs) 
            self.convolution = tf.keras.layers.Conv2D(**_convolution_kwargs,**kwargs)
        elif _rank == 3:
            _upsampling_kwargs.pop('interpolation')
            self.up_sampling = tf.keras.layers.UpSampling3D(**_upsampling_kwargs,**kwargs) 
            self.convolution = tf.keras.layers.Conv3D(**_convolution_kwargs,**kwargs)
        else:
            raise ValueError(f"rank should be one of 1, 2 or 3, not {_rank}.")

    def build(self,input_shape): # here we mandatorily specify input_spec, so we should override base class's build() indeed
        """Build `Layer`"""
        super(UpSampalingConv,self).build(input_shape) # will save raw input_shape in _build_input_shape
        input_shape = tf.TensorShape(input_shape)
        # self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:]) 
        temp_output_shape = self.up_sampling.compute_output_shape(input_shape)
        self.convolution.compute_output_shape(temp_output_shape)
    def call(self,inputs,*args,**kwargs):
        temp_output = self.up_sampling(inputs,*args,**kwargs)
        output = self.convolution(temp_output)
        return output
    def get_config(self):
        config = {**self._upsampling_kwargs,**self._convolution_kwargs}
        base_config = super().get_config()
        return base_config|config
class UpSampalingConv1D(UpSampalingConv):
    def __init__(self,**kwargs):
        super().__init__(rank=1,**kwargs)
class UpSampalingConv2D(UpSampalingConv):
    def __init__(self,**kwargs):
        super().__init__(rank=2,**kwargs)
class UpSampalingConv3D(UpSampalingConv):
    def __init__(self,**kwargs):
        super().__init__(rank=3,**kwargs)
#------------------------------------------------------------------------------------------------------------------------------#NOTE 新 
class UpSubpixelConv3D(tf.keras.layers.Layer):
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
        先将输入进行同维度卷积(中间层),扩展通道维度位原来的2**3倍(假定最终要进行系数位2的上采样)
        根据需要进行padding
        然后执行亚像素卷积
        实现反卷积黑盒
        需要指定output_shape
        同时要同步本层 upsampling层和_Sn层dtype
        """
        super(UpSubpixelConv3D,self).__init__(name=name,dtype=dtype)
        self.filters = filters #out put channels 并非中间层的filters
        self.kernel_size = kernel_size #such as[3,3,3]
        if len(strides)!=3:
            raise ValueError("3D conv need 3 dimensions for strides")
        self.inner_filters = filters  #中间层filters的特殊计算
        for stride in strides:
            self.inner_filters *= stride
        self.strides = [1]+strides+[1]
        self.inner_strides = [1]+[1,1,1]+[1]
        self.padding = padding.upper()
        self.output_padding = output_padding #不实现该逻辑
        self.use_bias = use_bias
        self.activation = activation_slect(activation)
        self.kernel_initializer = initializer_slect(kernel_initializer)
        self.bias_initializer = initializer_slect(bias_initializer)
        if spectral_normalization:
            self.Sn = Sn(dtype=dtype,**kwargs)
        self.spectral_normalization = spectral_normalization
        self.subpixel_conv3d = SubpixelConv3D(scale=strides)
        
    def build(self,input_shape,output_shape=None):
        super(UpSubpixelConv3D,self).build(input_shape=input_shape)
        #output_shape会重复指定输出深度 此处忽略该深度
        if len(input_shape)!=5:
            raise ValueError("3D conv need 5 dimensions for input_shape")
        if output_shape == None:# 手动计算输出shape
            output_shape = Reconstruction.ConvTransCal(input_shape=input_shape,filters=self.filters,kernel_size=self.kernel_size,strides=self.strides[1:-1],padding=self.padding)
        if len(output_shape)!=5:
            raise ValueError("3D conv need 5 dimensions for output_shape")
        #进行中间的卷积计算(通道扩展)
        inner_output_shape,padding,padding_vect = Reconstruction.ConvCalculation(input_shape=input_shape,
                                                                filters=self.inner_filters,
                                                                kernel_size=self.kernel_size,
                                                                strides=self.inner_strides[1:-1],
                                                                padding=self.padding)
        self.padding = padding
        self.padding_vect = padding_vect 
        # 不同于反卷积, 上采样+卷积替代的反卷积时正向卷积 回到正向卷积的角度 [input_shape[-1]s]是in_channels [self.filters]是out_channels
        self.w = self.add_weight('w',(self.kernel_size+[input_shape[-1]]+[self.inner_filters]),initializer=self.kernel_initializer,trainable=True)
        if self.use_bias:
            self.b = self.add_weight('b',(inner_output_shape[-1]),initializer=self.bias_initializer,trainable=True)
        else:
            pass 
        if self.spectral_normalization:
            self.Sn.build(input_shape=self.w.shape,w=self.w)
        output_shape = self.subpixel_conv3d.build(input_shape=inner_output_shape)
        output_shape = output_shape[:]
        return output_shape
    def call(self,x,training):
        x = tf.pad(x,self.padding_vect,'CONSTANT')  
        if self.spectral_normalization:
            self.Sn(self.w,training)
        y = tf.nn.conv3d(input=x,filters=self.w,strides=[1,1,1,1,1],padding='VALID')# 接管 stride恒定为1 padding为VALID
        if self.use_bias:
            y += self.b
        y = self.subpixel_conv3d(y)
        y = self.activation(y)
        return y
#------------------------------------------------------------------------------------------------------------------------------#
class Vgg2Conv3D(tf.keras.layers.Layer):
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
        super(Vgg2Conv3D,self).__init__(name=name,dtype=dtype)
        self.filters = filters # out put channels
        self.kernel_size = kernel_size # [3,3,3]
        if len(strides)!=3:
            raise ValueError("3D conv need 3 dimensions for strides")
        self.strides = [1]+strides+[1]
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.activation = activation_slect(activation)
        self.kernel_initializer = initializer_slect(kernel_initializer)
        self.bias_initializer = initializer_slect(bias_initializer)
    def build(self,input_shape):
        super(Vgg2Conv3D,self).build(input_shape=input_shape)
        if len(input_shape)!=5:
            raise ValueError("3D conv need 5 dimensions for input_shape") # padding 接入的预处理
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
        y = tf.nn.conv3d(input=x,filters=self.w,strides=self.strides,padding='VALID') # 接管padding方式后 用valid实现等效
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return tf.cast(y,dtype=tf.float32)  



# #------------------------------------------------------------------------------------------------------------------------------#
# class SeparableConv2D(tf.keras.layers.SeparableConv2D):
#     """
#         Simply wrap the tf.keras.layers.SeparableConv2D
#         input:...see the reference
#         output:...see the reference
#         reference:https://tensorflow.google.cn/api_docs/python/tf/keras/layers/SeparableConv2D
#     """
#     def __init__(self,*args,**kwargs):
#         super(SeparableConv2D,self).__init__(*args,**kwargs)
if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.experimental.enable_op_determinism()
    def default(method):
        """Decorates a method to detect overrides in subclasses."""
        method._is_default = True  # pylint: disable=protected-access
        return method
    class father():
        def __init__(self) -> None:
            pass
            self._build_input_shape = None
        @default
        def build(self,input_shape):
            if not hasattr(self.build, '_is_default'):
                self._build_input_shape = input_shape
            self.built = True
    class sun(father):
        def __init__(self) -> None:
            super().__init__()
        def build(self, input_shape):
            # super().build(input_shape)
            self.built=True
    y = sun()
    print(y._build_input_shape)
    y.build([None,3,1])
    print(y._build_input_shape)
    x = tf.constant([1.,2.,3.,4.,5.],shape=[1,5,1])
    conv1d_ = tf.keras.layers.Conv1D(filters=1, kernel_size=[2,]*1, strides=(1,)*1, padding='same',dilation_rate=1,kernel_initializer=tf.initializers.Ones(),use_bias=False)
    conv1d = ConvPadConcretization(conv1d_, padding_mode='constant',padding_constant=1)
    y = conv1d(x)
    print(tf.squeeze(y))
    conv1d_ = tf.keras.layers.Conv1D(filters=1, kernel_size=[2,]*1, strides=(1,)*1, padding='same',dilation_rate=1,kernel_initializer=tf.initializers.Ones(),use_bias=False)
    conv1d = ConvPadConcretization(conv1d_, padding_mode='reflect')
    y = conv1d(x)
    print(tf.squeeze(y))
    conv1d_ = tf.keras.layers.Conv1D(filters=1, kernel_size=[2,]*1, strides=(1,)*1, padding='same',dilation_rate=1,kernel_initializer=tf.initializers.Ones(),use_bias=False)
    conv1d = ConvPadConcretization(conv1d_, padding_mode='symmetric')
    y = conv1d(x)
    # x --> padded_x [1.,2.,3.,4.,5.,5.], zero padding x from right side
    # kernel = [1,1]
    # padded_x --> y [3.,7.,5.,9.,10.], conv padded x by kernel, and do not need extral 'padding'
    print(tf.squeeze(y))

    import numpy as np
    x = tf.constant([1.,2.,3.,4.,5.],shape=[1,5,1])
    print(tf.squeeze(x))
    conv1d = tf.keras.layers.Conv1D(filters=1, kernel_size=[3,]*1, strides=(2,)*1, padding='same',dilation_rate=1,kernel_initializer=tf.initializers.Ones(),use_bias=False)
    conv1d_ = tf.keras.layers.Conv1D(filters=1, kernel_size=[3,]*1, strides=(2,)*1, padding='same',dilation_rate=1,kernel_initializer=tf.initializers.Ones(),use_bias=False)
    conv1d_2 = ConvPadConcretization(conv1d_, padding_mode='constant',padding_constant=0)
    # 'constant' padding with constant 0, wrappered layer should have the same behavior than original one. So:
    y = conv1d(x)
    print(tf.squeeze(y))
    y_2 = conv1d_2(x)
    print(tf.squeeze(y_2))
    print(np.isclose(tf.reduce_mean(y-y_2),0.0))

    
 
 


   



    

    
    
            


