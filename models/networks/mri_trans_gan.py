import sys
import os
from numpy import float32
import tensorflow as tf
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)
sys.path.append(os.path.join(base,'../'))
from blocks import mri_trans_gan_2d,mri_trans_gan_3d,mri_trans_gan_3d_special 
from _gan_helper import GeneratorHelper,DiscriminatorHelper
__all__ = [ 
    "Generator",
    "Discriminator",
]
###################################################################
class Generator(tf.keras.Model):
    def __init__(self,args,name=None,dtype=None):
        super(Generator,self).__init__(name=name,dtype=dtype)
        #--------------------------------------------------------#
        up_sampling_method = args.up_sampling_method
        capacity_vector = args.capacity_vector
        res_blocks_num = args.res_blocks_num
        self_attention = args.self_attention_G
        dimensions_type = args.dimensions_type
        #--------------------------------------------------------#
        G_helper = GeneratorHelper(name=name,args=args)
        last_activation_name = G_helper.output_activation_name()
        #--------------------------------------------------------#
        if dimensions_type=="2D":
            self.block_list = []
            self.block_list.append(mri_trans_gan_2d.Conv7S1(filters=capacity_vector,dtype=dtype))
            self.block_list.append(mri_trans_gan_2d.DownSampling(filters=capacity_vector*2,dtype=dtype))
            self.block_list.append(mri_trans_gan_2d.DownSampling(filters=capacity_vector*4,dtype=dtype))
            self.block_list.append(mri_trans_gan_2d.ResBlocks(filters=capacity_vector*4,n=res_blocks_num,dtype=dtype))
            self.block_list.append(mri_trans_gan_2d.UpSampling(filters=capacity_vector*2,dtype=dtype,up_sampling_method=up_sampling_method))
            self.block_list.append(mri_trans_gan_2d.UpSampling(filters=capacity_vector,dtype=dtype,up_sampling_method=up_sampling_method))
            tmp_policy = tf.keras.mixed_precision.Policy('float32')
            self.block_list.append(mri_trans_gan_2d.Conv7S1(filters=1,activation=last_activation_name,dtype=dtype,specific_out_dtype=tmp_policy))
        elif dimensions_type == "3D":
            self.block_list = []
            self.block_list.append(mri_trans_gan_3d.Conv7S1(filters=capacity_vector,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d.DownSampling(filters=capacity_vector*2,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d.DownSampling(filters=capacity_vector*4,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d.ResBlocks(filters=capacity_vector*4,n=res_blocks_num,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d.UpSampling(filters=capacity_vector*2,dtype=dtype,up_sampling_method=up_sampling_method))
            self.block_list.append(mri_trans_gan_3d.UpSampling(filters=capacity_vector,dtype=dtype,up_sampling_method=up_sampling_method))
            tmp_policy = tf.keras.mixed_precision.Policy('float32')
            self.block_list.append(mri_trans_gan_3d.Conv7S1(filters=1,activation=last_activation_name,dtype=dtype,specific_out_dtype=tmp_policy))
        elif dimensions_type == "3D_special":
            self.block_list = []
            self.block_list.append(mri_trans_gan_3d_special.Conv7S1(filters=capacity_vector,dtype=dtype,activation="relu"))
            self.block_list.append(mri_trans_gan_3d_special.DownSampling(filters=capacity_vector*2,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d_special.DownSampling(filters=capacity_vector*4,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d_special.ResBlocks(filters=capacity_vector*4,n=res_blocks_num,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d_special.UpSampling(filters=capacity_vector*2,dtype=dtype,up_sampling_method=up_sampling_method))
            self.block_list.append(mri_trans_gan_3d_special.UpSampling(filters=capacity_vector,dtype=dtype,up_sampling_method=up_sampling_method))
            tmp_policy = tf.keras.mixed_precision.Policy('float32')
            self.block_list.append(mri_trans_gan_3d_special.Conv7S1(filters=1,activation=last_activation_name,dtype=dtype,specific_out_dtype=tmp_policy))
        else:
            raise ValueError("dimension type is not supported: "+dimensions_type)
    def build(self,input_shape):
        flow_shape=input_shape[:]
        for item in self.block_list:
            flow_shape=item.build(input_shape=flow_shape)
        self.built = True
    def call(self,in_put,training=True,step=None,epoch=None):
        # in_put=(x,mask)生成的样本(生成器输出)自带mask 要求真实样本你必须带有mask
        x,mask = in_put
        for item in self.block_list:
            y = item(x,training=training)
            x = y
        mask = tf.cast(mask,x.dtype)
        y = tf.multiply(x,mask)
        return y

class Discriminator(tf.keras.Model):
    def __init__(self,args,name=None,dtype=None):
        super(Discriminator,self).__init__(name=name,dtype=dtype)
        #--------------------------------------------------------#
        capacity_vector = int(args.capacity_vector)
        dimensions_type = args.dimensions_type
        self_attention = args.self_attention_D
        sn_flag = bool(args.spectral_normalization)
        sn_iter_k = int(args.sn_iter_k)
        sn_clip_flag = bool(args.sn_clip_flag)
        sn_clip_range = float(args.sn_clip_range)
        #--------------------------------------------------------#
        D_helper = DiscriminatorHelper(name=name,args=args)
        use_sigmoid = D_helper.output_sigmoid_flag()
        #--------------------------------------------------------#
        if dimensions_type=="2D":
            self.block_list=[]
            self.block_list.append(mri_trans_gan_2d.Conv4S2(filters=capacity_vector,norm=False,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
            self.block_list.append(mri_trans_gan_2d.Conv4S2(filters=capacity_vector*2,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
            self.block_list.append(mri_trans_gan_2d.Conv4S2(filters=capacity_vector*4,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
            self.block_list.append(mri_trans_gan_2d.Conv4S2(filters=capacity_vector*8,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
            tmp_policy = tf.keras.mixed_precision.Policy('float32')
            self.block_list.append(mri_trans_gan_2d.Conv4S1(use_sigmoid=use_sigmoid,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype,specific_out_dtype=tmp_policy))
        elif dimensions_type == "3D":
            self.block_list=[]
            self.block_list.append(mri_trans_gan_3d.Conv4S2(filters=capacity_vector,norm=False,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d.Conv4S2(filters=capacity_vector*2,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d.Conv4S2(filters=capacity_vector*4,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d.Conv4S2(filters=capacity_vector*8,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
            tmp_policy = tf.keras.mixed_precision.Policy('float32')
            self.block_list.append(mri_trans_gan_3d.Conv4S1(use_sigmoid=use_sigmoid,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype,specific_out_dtype=tmp_policy))
        elif dimensions_type == "3D_special":
            self.block_list=[]
            # paper
            self.block_list.append(mri_trans_gan_3d_special.Conv4S2(filters=capacity_vector,norm=False,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d_special.Conv4S2(filters=capacity_vector*2,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d_special.Conv4S2(filters=capacity_vector*4,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d_special.Conv4S2(filters=capacity_vector*8,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
            tmp_policy = tf.keras.mixed_precision.Policy('float32')
            self.block_list.append(mri_trans_gan_3d_special.Conv4S1(use_sigmoid=use_sigmoid,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype,specific_out_dtype=tmp_policy))
        else:
            raise ValueError("dimension type is not supported: "+dimensions_type)
    def build(self,input_shape):
        flow_shape=input_shape[:]
        for item in self.block_list:
            flow_shape=item.build(input_shape=flow_shape)
        self.built = True
    def call(self,in_put,buf_flag=False,training=True,step=None,epoch=None): # 额外多一个buf_flag
        x,*_ = in_put
        buf = []
        for item in self.block_list:
            y = item(x,training=training)
            buf.append(y)
            x = y
        if buf_flag:
            return y,buf
        else:
            return y
if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    x = tf.random.normal(shape=[1,3,128,128,1])
    x_m = tf.random.normal(shape=[1,16,240,240,1])
    m = tf.random.normal(shape=[1,3,128,128,1])
    class tmp_args():
        def __init__(self) -> None:
            pass
    args = tmp_args()
    args.capacity_vector = 32
    args.up_sampling_method = "up_conv"
    args.res_blocks_num = 9
    args.self_attention_G = None
    args.dimensions_type = "3D_special"
    args.self_attention_D = None
    args.spectral_normalization = False
    args.sn_iter_k = 1
    args.sn_clip_flag = True
    args.sn_clip_range = 128.0
    args.domain = [0.0,1.0]
    args.gan_loss_name = "WGAN-GP"
    g = Generator(args)
    d = Discriminator(args)
    g.build(input_shape=[1,3,128,128,1])
    d.build(input_shape=[1,3,128,128,1])
    input_ = [x,m]
    y = g(input_)
    print(y.shape,y.dtype)
    y = d(input_)
    print(y.shape,y.dtype)
