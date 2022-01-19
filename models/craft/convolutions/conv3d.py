"""
以下卷积 反卷积层 的设计 因为keras源码自带conv不支持反射 padding 所以需要对此进行融合
先行padding 再valid卷积 保持等价
"""
from re import template
import sys
import os
import warnings
import tensorflow as tf
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)
sys.path.append(os.path.join(base,'../'))
from _build_helper import Reconstruction
from normalizations import SpectralNormalization as Sn
from activations import activation_slect
from initializers import initializer_slect
from super_resolution import SubpixelConv3D

__all__ = [
    "Conv3D",
    "Conv3DTranspose",
    "UpSampalingConv3D",
    "UpSubpixelConv3D",
    "Vgg2Conv3D",
    # "GroupConv3D",
]
class Conv3D(tf.keras.layers.Layer):
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
        kernel_size = (10,10,10) 
        strides = (1,1,1)
        padding = 'valid'
        use_bias = False
        
        valid 时 不进行padding 正常卷积
        SAME 时 默认zero padding
        CONSTANT 时 默认 zero padding 
        REFLECT  时 REFLECT padding
        SYMMETRIC 时 SYMMETRIC padding
        """
        super(Conv3D,self).__init__(name=name,dtype=dtype)
        self.filters = filters#out put channels
        self.kernel_size = kernel_size#[3,3,3]
        if len(strides)!=3:
            raise ValueError("3D conv need 3 dimensions for strides")
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
        super(Conv3D,self).build(input_shape=input_shape)
        if len(input_shape)!=5:
            raise ValueError("3D conv need 5 dimensions for input_shape")#padding 接入的预处理
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
            self.Sn(self.w,training)
            tf.print("SN is enabled!")
        y = tf.nn.conv3d(input=x,filters=self.w,strides=self.strides,padding="VALID")#接管padding方式后 用valid实现等效
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return y 
#------------------------------------------------------------------------------------------------------------------------------#
class Conv3DTranspose(tf.keras.layers.Layer):
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
        super(Conv3DTranspose,self).__init__(name=name,dtype=dtype)
        self.filters = filters #out put channels
        self.kernel_size = kernel_size #[3,3,3]
        if len(strides)!=3:
            raise ValueError("3D conv need 3 dimensions for strides")
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
        super(Conv3DTranspose,self).build(input_shape=input_shape)
        if len(input_shape)!=5:
            raise ValueError("3D conv need 5 dimensions for input_shape")
        if output_shape == None:# 手动计算输出shape
            output_shape = Reconstruction.ConvTransCal(input_shape=input_shape,filters=self.filters,kernel_size=self.kernel_size,strides=self.strides[1:-1],padding=self.padding)
        if len(output_shape)!=5:
            raise ValueError("3D conv need 5 dimensions for output_shape")
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
            self.Sn(self.w,training)
        y = tf.nn.conv3d_transpose(input=x,filters=self.w,output_shape=tmp_output_shape,strides=self.strides,padding=self.padding)
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return y 
#------------------------------------------------------------------------------------------------------------------------------#
class UpSampalingConv3D(tf.keras.layers.Layer):
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
        同时要同步本层 upsampling层和_Sn层dtype
        """
        super(UpSampalingConv3D,self).__init__(name=name,dtype=dtype)
        self.filters = filters #out put channels
        self.kernel_size = kernel_size #such as[3,3,3]
        if len(strides)!=3:
            raise ValueError("3D conv need 3 dimensions for strides")
        self.strides = [1]+strides+[1]
        self.up_size = self.strides[1:-1]
        self.up_op = tf.keras.layers.UpSampling3D(size=self.up_size,dtype=dtype)#上采样只对非通道和非batch进行 剔除冗余维度
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
        super(UpSampalingConv3D,self).build(input_shape=input_shape)
        #output_shape会重复指定输出深度 此处忽略该深度
        if len(input_shape)!=5:
            raise ValueError("3D conv need 5 dimensions for input_shape")
        if output_shape == None:# 手动计算输出shape
            output_shape = Reconstruction.ConvTransCal(input_shape=input_shape,filters=self.filters,kernel_size=self.kernel_size,strides=self.strides[1:-1],padding=self.padding)
        if len(output_shape)!=5:
            raise ValueError("3D conv need 5 dimensions for output_shape")
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
                  0-self.padding_vect[3][0]:int(x.shape[3])+self.padding_vect[3][1],
                  0-self.padding_vect[4][0]:int(x.shape[4])+self.padding_vect[4][1]]
        else:
            x = tf.pad(x,self.padding_vect,"CONSTANT")  
        if self.spectral_normalization:
            self.Sn(self.w,training)
        y = tf.nn.conv3d(input=x,filters=self.w,strides=[1,1,1,1,1],padding=self.padding)#接管stride恒定为1
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return y
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
        x = tf.pad(x,self.padding_vect,"CONSTANT")  
        if self.spectral_normalization:
            self.Sn(self.w,training)
        y = tf.nn.conv3d(input=x,filters=self.w,strides=[1,1,1,1,1],padding="VALID")# 接管 stride恒定为1 padding为VALID
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
        y = tf.nn.conv3d(input=x,filters=self.w,strides=self.strides,padding="VALID") # 接管padding方式后 用valid实现等效
        if self.use_bias:
            y += self.b
        y = self.activation(y)
        return tf.cast(y,dtype=tf.float32)  
#------------------------------------------------------------------------------------------------------------------------------#
class GroupConv3D(tf.keras.Model):
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
        kernel_size = (10,10,10) 
        strides = (1,1,1)
        padding = 'valid'
        use_bias = False
        
        valid 时 不进行padding 正常卷积
        SAME 时 默认zero padding
        CONSTANT 时 默认 zero padding 
        REFLECT  时 REFLECT padding
        SYMMETRIC 时 SYMMETRIC padding
        """
        super(GroupConv3D,self).__init__(name=name,dtype=dtype)
        if filters%2 != 0:
            raise ValueError("Group CNN needs even filters!")
        single_filters=filters//2
        self.conv3D_0 = Conv3D(filters=single_filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer, 
                               bias_initializer=bias_initializer,
                               activation=activation,
                               spectral_normalization=spectral_normalization,
                               dtype=dtype)
        self.conv3D_1 = Conv3D(filters=single_filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer, 
                               bias_initializer=bias_initializer,
                               activation=activation,
                               spectral_normalization=spectral_normalization,
                               dtype=dtype)
    def build(self,input_shape):
        self.half_index = input_shape[-1]//2
        sigle_input_shape = input_shape[0:-1]+[self.half_index]
        flow_shape_0 = self.conv3D_0.build(input_shape=sigle_input_shape)
        flow_shape_1 = self.conv3D_1.build(input_shape=sigle_input_shape)
        if flow_shape_0!=flow_shape_1:
            raise ValueError("Group Conv Error")
        flow_shape = flow_shape_0[:]
        flow_shape[-1] +=  flow_shape_1[-1]
        self.built = True
        output_shape = flow_shape[:]
        return output_shape
    def call(self,x,training):
        x0 = x[:,:,:,:,0:self.half_index]
        x1 = x[:,:,:,:,self.half_index::]
        y0 = self.conv3D_0(x0,training)
        y1 = self.conv3D_1(x1,training)
        y = tf.concat([y0,y1],-1)
        return y
if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    up3d = UpSubpixelConv3D(filters=128,kernel_size=[3,3,3],strides=[2,2,2],padding="same")
    x = tf.random.normal([1,16,16,16,8])
    up3d.build(input_shape=[None,16,16,16,8])
    y = up3d(x)
    print(y.shape,y.dtype)
    up3d = UpSubpixelConv3D(filters=128,kernel_size=[3,3,3],strides=[1,2,2],padding="same")
    x = tf.random.normal([1,16,16,16,8])
    up3d.build(input_shape=[None,16,16,16,8])
    y = up3d(x)
    print(y.shape,y.dtype)
    up3d = tf.keras.conv3d_transpose(filters=128,kernel_size=[3,3,3],strides=[1,2,2],padding="same")
    x = tf.random.normal([1,16,16,16,8])
    up3d.build(input_shape=[None,16,16,16,8])
    y = up3d(x)
    print(up3d.get_config())

