import tensorflow as tf

from ..blocks import mri_trans_gan_2d,mri_trans_gan_3d_special,mri_trans_gan_3d
from ._gan_helper import GeneratorHelper,DiscriminatorHelper

"""
做模型结构的探究 
开辟额外支路

生成器 输入x              128*128--128*128--64*64--32*32--64*64--128*128--128*128
      输入的patch位置信息  240*240
判别器
"""
BLOCKS_DICT = {'2D':mri_trans_gan_2d,'3D_special':mri_trans_gan_3d_special,'3D':mri_trans_gan_3d}
###################################################################
class Generator(tf.keras.Model):
    def __init__(self,args,name=None,dtype=None):
        super(Generator,self).__init__(name=name,dtype=dtype)
        #--------------------------------------------------------#
        up_sampling_method = args['up_sampling_method']
        capacity_vector = args['capacity_vector']
        res_blocks_num = args['res_blocks_num']
        self_attention = args['self_attention_G']
        dimensions_type = args['dimensions_type']
        #--------------------------------------------------------#
        G_helper = GeneratorHelper(name=name,args=args)
        last_activation_name = G_helper.output_activation_name()
        output_domain = G_helper.domain
        #--------------------------------------------------------#
        self.broad_cast_list = []
        #--------------------------------------------------------#
        try:
            blocks = BLOCKS_DICT[dimensions_type]
        except KeyError:
            raise ValueError("dimension type is not supported: "+dimensions_type)
        self.block_list = []
        self.block_list.append(blocks.Conv7S1(filters=capacity_vector,dtype=dtype,activation='relu'))
        self.block_list.append(blocks.DownSampling(filters=capacity_vector*2,dtype=dtype))
        self.block_list.append(blocks.DownSampling(filters=capacity_vector*4,dtype=dtype))
        self.block_list.append(blocks.ResBlocks(filters=capacity_vector*4,n=res_blocks_num,dtype=dtype))
        self.block_list.append(blocks.UpSampling(filters=capacity_vector*2,dtype=dtype,up_sampling_method=up_sampling_method))
        self.block_list.append(blocks.UpSampling(filters=capacity_vector,dtype=dtype,up_sampling_method=up_sampling_method))
        tmp_policy = tf.keras.mixed_precision.Policy('float32')
        self.block_list.append(blocks.Conv7S1(filters=1,activation=last_activation_name,dtype=dtype,specific_out_dtype=tmp_policy,output_domain=output_domain))
    def build(self,input_shape):
        flow_shape=input_shape[:]
        for item in self.block_list:
            flow_shape=item.compute_output_shape(input_shape=flow_shape).as_list()
        self.built = True
    def call(self,in_put,training=True,step=None,epoch=None):
        # in_put=(x,mask)生成的样本(生成器输出)自带mask 要求真实样本你必须带有mask
        # logging.getLogger(__name__).debug("Debug hhhhhhhhhhhhh")
        x,mask = in_put
        for item in self.block_list:
            y = item(x,training=training)
            x = y
        mask = tf.cast(mask,x.dtype)
        
        y = (x*mask*0.999+0.001)*mask
        return y

class Discriminator(tf.keras.Model):
    def __init__(self,args,name=None,dtype=None):
        super(Discriminator,self).__init__(name=name,dtype=dtype)
        #--------------------------------------------------------#
        capacity_vector = int(args['capacity_vector'])
        dimensions_type = args['dimensions_type']
        self_attention = args['self_attention_D']
        sn_flag = bool(args['spectral_normalization'])
        sn_iter_k = int(args['sn_iter_k'])
        sn_clip_flag = bool(args['sn_clip_flag'])
        sn_clip_range = float(args['sn_clip_range'])
        #--------------------------------------------------------#
        D_helper = DiscriminatorHelper(name=name,args=args)
        use_sigmoid = D_helper.output_sigmoid_flag()
        #--------------------------------------------------------#
        self.broad_cast_list = []
        #--------------------------------------------------------#
        try:
            blocks = BLOCKS_DICT[dimensions_type]
        except KeyError:
            raise ValueError("dimension type is not supported: "+dimensions_type)
        self.block_list:list[tf.keras.layers.Layer]=[]
        self.block_list.append(blocks.Conv4S2(filters=capacity_vector,norm=False,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
        self.block_list.append(blocks.Conv4S2(filters=capacity_vector*2,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
        self.block_list.append(blocks.Conv4S2(filters=capacity_vector*4,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
        self.block_list.append(blocks.Conv4S2(filters=capacity_vector*8,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype))
        tmp_policy = tf.keras.mixed_precision.Policy('float32')
        self.block_list.append(blocks.Conv4S1(use_sigmoid=use_sigmoid,spectral_normalization=sn_flag,iter_k=sn_iter_k,clip_flag=sn_clip_flag,clip_range=sn_clip_range,dtype=dtype,specific_out_dtype=tmp_policy))
    def build(self,input_shape):
        flow_shape=input_shape[:]
        for item in self.block_list:
            flow_shape=item.compute_output_shape(input_shape=flow_shape).as_list()
        self.built = True
    def call(self,in_put,buf_flag=False,training=True,step=None,epoch=None): # 额外多一个buf_flag
        # logging.getLogger(__name__).debug("Debug hhhhhhhhhhhhh")
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


