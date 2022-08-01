"""
以下卷积 反卷积层 的设计 因为keras源码自带conv不支持反射 padding 所以需要对此进行融合
先行padding 再valid卷积 保持等价
"""
import tensorflow as tf
from .._build_helper import Reconstruction
from ..normalizations import SpectralNormalization as Sn
from ..activations import activation_slect
from ..initializers import initializer_slect

__all__ = [
    "Conv2D",
    "Conv2DVgg",
    "Conv2DTranspose",
    "UpSampalingConv2D",
    # "SeparableConv2D",
]
class Conv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters=None,
                 kernel_size=None,
                 strides=None,
                 padding=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros',
                 activation=None,
                 spectral_normalization=False,
                 name=None,
                 dtype=None,**kwargs):
        """
        参照官方API
        增加_Sn
        filters = 8
        kernel_size = (10,10) 
        strides = (1,1)
        padding = 'valid'
        use_bias = False
        
        valid 时 不进行padding 正常卷积
        SAME 时 默认zero padding
        CONSTANT 时 默认 zero padding 
        REFLECT  时 REFLECT padding
        SYMMETRIC 时 SYMMETRIC padding
        """
        super(Conv2D,self).__init__(name=name,dtype=dtype)
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
        if spectral_normalization:
            self.Sn = Sn(dtype=dtype,**kwargs)
        self.spectral_normalization = spectral_normalization
    def build(self,input_shape):
        super(Conv2D,self).build(input_shape=input_shape)
        #padding 接入的预处理
        if len(input_shape)!=4:
            raise ValueError("2D conv need 4 dimensions for input_shape")
        output_shape,padding,padding_vect = Reconstruction.ConvCalculation(input_shape=input_shape,
                                                                           filters=self.filters,
                                                                           kernel_size=self.kernel_size,
                                                                           strides=self.strides[1:-1],
                                                                           padding=self.padding)
        self.padding = padding
        self.padding_vect = padding_vect  
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
        x = tf.pad(x,self.padding_vect,self.padding)
        if self.spectral_normalization:
            self.Sn(self.w,training)#适应计算类型
        y = tf.nn.conv2d(input=x,filters=self.w,strides=self.strides,padding="VALID")#接管padding方式后 用valid实现等效
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return y  
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
class Conv2DTranspose(tf.keras.layers.Layer):
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
        为了精确控制反卷积的输出shape
        需要指定output_shape
        """
        super(Conv2DTranspose,self).__init__(name=name,dtype=dtype)
        self.filters = filters #out put channels
        self.kernel_size = kernel_size #[3,3]
        if len(strides)!=2:
            raise ValueError("2D conv need 2 dimensions for strides")
        self.strides = [1]+strides+[1]
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
        super(Conv2DTranspose,self).build(input_shape=input_shape)
        #output_shape会重复指定输出深度 此处忽略该深度
        if len(input_shape)!=4:
            raise ValueError("2D conv need 4 dimensions for input_shape")
        if output_shape == None:# 手动计算输出shape
            output_shape = Reconstruction.ConvTransCal(input_shape=input_shape,filters=self.filters,kernel_size=self.kernel_size,strides=self.strides[1:-1],padding=self.padding)
        if len(output_shape)!=4:
            raise ValueError("2D conv need 4 dimensions for output_shape")
        Reconstruction.ConvTransCheck(input_shape=input_shape[1:-1],output_shape=output_shape[1:-1],filters=self.filters,kernel_size=self.kernel_size,strides=self.strides[1:-1],padding=self.padding)
        # 从正向卷积的角度 [self.filters]依旧是in_channels [input_shape[-1]]依旧是out_channels
        self.w = self.add_weight('w',(self.kernel_size+[self.filters]+[input_shape[-1]]),initializer=self.kernel_initializer,trainable=True)
        if self.use_bias:
            self.b = self.add_weight('b',(output_shape[-1]),initializer=self.bias_initializer,trainable=True)
        else:
            pass 
        self.tmp_output_shape = output_shape
        if self.spectral_normalization:
            self.Sn.build(input_shape=self.w.shape,w=self.w)
        output_shape = output_shape[:]
        return output_shape
    def output_shape_tweaker(self,x):
        B = int(x.shape[0])
        if B != self.tmp_output_shape[0]:
            current_output_shape = [B]+self.tmp_output_shape[1::]
        else:
            current_output_shape =self.tmp_output_shape[:]
        return current_output_shape
    def call(self,x,training):
        tmp_output_shape = self.output_shape_tweaker(x)
        if self.spectral_normalization:
            self.Sn(self.w,training)#适应计算类型
        y = tf.nn.conv2d_transpose(input=x,filters=self.w,output_shape=tmp_output_shape,strides=self.strides,padding=self.padding)
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return y
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
class SeparableConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters=None,
                 kernel_size=None,
                 strides=None,
                 padding=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros',
                 activation=None,
                 spectral_normalization=False,
                 name=None,
                 dtype=None,**kwargs):
        """
        深度可分离卷积
        由深度卷积 1x1卷积组成
        对深度卷积部分单独计算谱范数只是理论上可行 实践意义上 没有硬件 驱动与底层软件优化 所以不是并行计算 时间复杂度高
        """
        super(SeparableConv2D,self).__init__(name=name,dtype=dtype)
        self.filters = filters#out put channels
        self.kernel_size = kernel_size#[3,3]
        if len(strides)!=2:
            raise ValueError("2D conv need 2 dimensions for strides")
        self.strides = [1]+strides+[1]
        self.strides_depth_wise = [1]+strides+[1]
        self.strides_point_wise = [1,1,1,1]
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.activation = activation_slect(activation)
        self.kernel_initializer = initializer_slect(kernel_initializer)
        self.bias_initializer = initializer_slect(bias_initializer)

        if spectral_normalization:
            self._Sn_depth_wise = Sn(dtype=dtype,**kwargs)
            self._Sn_point_wise = Sn(dtype=dtype,**kwargs)
 
        self.spectral_normalization = spectral_normalization
    def build(self,input_shape):
        super(SeparableConv2D,self).build(input_shape=input_shape)
        #padding 接入的预处理
        if len(input_shape)!=4:
            raise ValueError("2D conv need 4 dimensions for input_shape")
        output_shape,padding,padding_vect = Reconstruction.ConvCalculation(input_shape=input_shape,
                                                                           filters=self.filters,
                                                                           kernel_size=self.kernel_size,
                                                                           strides=self.strides[1:-1],
                                                                           padding=self.padding)
        self.padding = padding
        self.padding_vect = padding_vect

        depth = int(input_shape[-1])
        self.depth_wise_w = self.add_weight('depth_wise_w',(self.kernel_size+[depth]+[1]),initializer=self.kernel_initializer,trainable=True)
        self.w = self.add_weight('w',([1,1]+[input_shape[-1]]+[self.filters]),initializer=self.kernel_initializer,trainable=True)
        if self.use_bias:
            self.b = self.add_weight('b',(output_shape[-1]),initializer=self.bias_initializer,trainable=True)
        else:
            pass 
        if self.spectral_normalization:
            self._Sn_depth_wise.build(input_shape=self.depth_wise_w.shape,w=self.depth_wise_w)
            self._Sn_point_wise.build(input_shape=self.w.shape,w=self.w)
        output_shape = output_shape[:]
        return output_shape
    def call(self,x,training):
        x = tf.pad(x,self.padding_vect,self.padding)
        if self.spectral_normalization:
            self._Sn_depth_wise(self.depth_wise_w,training)#适应计算类型
            self._Sn_point_wise(self.w,training)#适应计算类型

        y = tf.nn.separable_conv2d(input=x,depthwise_filter=self.depth_wise_w,pointwise_filter=self.w,strides=self.strides,padding="VALID")#接管padding方式后 用valid实现等效
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return y