import sys
import os
from numpy import broadcast, float32
import tensorflow as tf
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)
sys.path.append(os.path.join(base,'../'))
from blocks import mri_trans_gan_2d,mri_trans_gan_3d_special 
from blocks import mri_trans_gan_3d_bak as mri_trans_gan_3d
from _gan_helper import GeneratorHelper,DiscriminatorHelper
"""
做模型结构的探究 
开辟额外支路

生成器 输入x              128*128--128*128--64*64--32*32--64*64--128*128--128*128
      输入的patch位置信息  240*240
判别器
"""
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
        output_domain = G_helper.domain
        #--------------------------------------------------------#
        self.broad_cast_list = []
        #--------------------------------------------------------#
        if dimensions_type=="2D":
            self.block_list = []
            self.block_list.append(mri_trans_gan_2d.Conv7S1(filters=capacity_vector,dtype=dtype))
            self.block_list.append(mri_trans_gan_2d.DownSampling(filters=capacity_vector*2,dtype=dtype))
            self.block_list.append(mri_trans_gan_2d.DownSampling(filters=capacity_vector*4,dtype=dtype))
            self.block_list.append(mri_trans_gan_2d.ResBlocks(filters=capacity_vector*4+capacity_vector*4//8,n=res_blocks_num,dtype=dtype))
            self.block_list.append(mri_trans_gan_2d.UpSampling(filters=capacity_vector*2,dtype=dtype,up_sampling_method=up_sampling_method))
            self.block_list.append(mri_trans_gan_2d.UpSampling(filters=capacity_vector,dtype=dtype,up_sampling_method=up_sampling_method))
            tmp_policy = tf.keras.mixed_precision.Policy('float32')
            self.block_list.append(mri_trans_gan_2d.Conv7S1(filters=1,activation=last_activation_name,dtype=dtype,specific_out_dtype=tmp_policy,output_domain=output_domain))
        elif dimensions_type == "3D":
            self.block_list = []
            self.block_list.append(mri_trans_gan_3d.Conv7S1(filters=capacity_vector,dtype=dtype,activation="relu"))
            self.block_list.append(mri_trans_gan_3d.DownSampling(filters=capacity_vector*2,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d.DownSampling(filters=capacity_vector*4,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d.ResBlocks(filters=capacity_vector*4+capacity_vector*4//8,n=res_blocks_num,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d.UpSampling(filters=capacity_vector*2,dtype=dtype,up_sampling_method=up_sampling_method))
            self.block_list.append(mri_trans_gan_3d.UpSampling(filters=capacity_vector,dtype=dtype,up_sampling_method=up_sampling_method))
            tmp_policy = tf.keras.mixed_precision.Policy('mixed_float16')
            self.block_list.append(mri_trans_gan_3d.Conv7S1(filters=1,activation=last_activation_name,dtype=dtype,specific_out_dtype=tmp_policy,output_domain=output_domain))
        elif dimensions_type == "3D_special":
            self.block_list = []
            self.block_list.append(mri_trans_gan_3d_special.Conv7S1(filters=capacity_vector,dtype=dtype,activation="relu"))
            self.block_list.append(mri_trans_gan_3d_special.DownSampling(filters=capacity_vector*2,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d_special.DownSampling(filters=capacity_vector*4,dtype=dtype))
            self.block_list.append(mri_trans_gan_3d_special.ResBlocks(filters=capacity_vector*4+capacity_vector*4//8,n=res_blocks_num,dtype=dtype)) # TODO 为了迎合patch_mask添加到通道维度的操作 临时方案 并不合理
            self.block_list.append(mri_trans_gan_3d_special.UpSampling(filters=capacity_vector*2,dtype=dtype,up_sampling_method=up_sampling_method))
            self.block_list.append(mri_trans_gan_3d_special.UpSampling(filters=capacity_vector,dtype=dtype,up_sampling_method=up_sampling_method))
            tmp_policy = tf.keras.mixed_precision.Policy('float32')
            self.block_list.append(mri_trans_gan_3d_special.Conv7S1(filters=1,activation=last_activation_name,dtype=dtype,specific_out_dtype=tmp_policy,output_domain=output_domain))
        else:
            raise ValueError("dimension type is not supported: "+dimensions_type)
    def patch_mask_wrapper(self,patch,patch_mask):
        tmp_indexs = []
        for i in range(len(patch.shape)-1):#此处不考察最后的通道维度C
            if patch.shape[i]!=patch_mask.shape[i]:
                tmp_indexs.append(i)
        tmp_mask = patch_mask
        for index in tmp_indexs:
            indicator = round(patch_mask.shape[index]/patch.shape[index])
            target = [x*indicator for x in range(patch_mask.shape[index]//indicator)]
            tmp_mask = tf.gather(tmp_mask,axis=index,indices=target)
        padding_vector = [[0,0] for _ in range(len(patch.shape))]
        for index in tmp_indexs:
            pad_nums = max(0,patch.shape[index]-tmp_mask.shape[index])
            padding_vector[index] = [pad_nums//2,pad_nums-pad_nums//2]
        return tf.pad(tmp_mask,padding_vector)
    def input_shape_warpper(self,input_shape): # 为融合进输入的patch mask 信息计算新的各层输入输出shape
        tmp_input_shape = input_shape[:]
        for item in [8,1]:
            if tmp_input_shape[-1]//item >=1:
                self.broad_cast_list.append(tmp_input_shape[-1]//item)
                tmp_input_shape[-1] = tmp_input_shape[-1]+self.broad_cast_list[-1]
                break
            else:
                self.broad_cast_list.append(1)
                tmp_input_shape[-1] = tmp_input_shape[-1]+self.broad_cast_list[-1]
                break
        return tmp_input_shape
    def build(self,input_shape):
        flow_shape=input_shape[:]
        for item in self.block_list:
            flow_shape=self.input_shape_warpper(flow_shape)
            flow_shape=item.build(input_shape=flow_shape)
        self.built = True
    def call(self,in_put,training=True,step=None,epoch=None):
        # in_put=(x,mask)生成的样本(生成器输出)自带mask 要求真实样本你必须带有mask
        # print("Debug hhhhhhhhhhhhh")
        x,x_m,mask = in_put
        for i,(item,broadcast_indicator) in enumerate(zip(self.block_list,self.broad_cast_list)):
            tmp_x_m = self.patch_mask_wrapper(x,x_m) # 处理非C维度
            tmp_x_m = tf.broadcast_to(tmp_x_m,shape=tmp_x_m.shape[0:-1]+[broadcast_indicator])# 处理C维度
            x = tf.concat([x,tmp_x_m],axis=-1)
            y = item(x,training=training)
            x = y
        mask = tf.cast(mask,x.dtype)  
        # x = x*30.0-10.0 # TODO domain shift
        y = x*mask
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
        self.broad_cast_list = []
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
    def patch_mask_wrapper(self,patch,patch_mask):
        tmp_indexs = []
        for i in range(len(patch.shape)-1):#此处不考察最后的通道维度C
            if patch.shape[i]!=patch_mask.shape[i]:
                tmp_indexs.append(i)
        tmp_mask = patch_mask
        for index in tmp_indexs:
            indicator = round(patch_mask.shape[index]/patch.shape[index])
            target = [x*indicator for x in range(patch_mask.shape[index]//indicator)]
            tmp_mask = tf.gather(tmp_mask,axis=index,indices=target)
        padding_vector = [[0,0] for _ in range(len(patch.shape))]
        for index in tmp_indexs:
            pad_nums = max(0,patch.shape[index]-tmp_mask.shape[index])
            padding_vector[index] = [pad_nums//2,pad_nums-pad_nums//2]
        return tf.pad(tmp_mask,padding_vector)
    def input_shape_warpper(self,input_shape): # 为融合进输入的patch mask 信息计算新的各层输入输出shape
        tmp_input_shape = input_shape[:]
        for item in [8,1]:
            if tmp_input_shape[-1]//item >=1:
                self.broad_cast_list.append(tmp_input_shape[-1]//item)
                tmp_input_shape[-1] = tmp_input_shape[-1]+self.broad_cast_list[-1]
                break
            else:
                self.broad_cast_list.append(1)
                tmp_input_shape[-1] = tmp_input_shape[-1]+self.broad_cast_list[-1]
                break
        return tmp_input_shape
    def build(self,input_shape):
        flow_shape=input_shape[:]
        for item in self.block_list:
            flow_shape=self.input_shape_warpper(flow_shape)
            flow_shape=item.build(input_shape=flow_shape)
        self.built = True
    def call(self,in_put,buf_flag=False,training=True,step=None,epoch=None): # 额外多一个buf_flag
        # print("Debug hhhhhhhhhhhhh")
        x,x_m,*_ = in_put
        buf = []
        for item,broadcast_indicator in zip(self.block_list,self.broad_cast_list):
            tmp_x_m = self.patch_mask_wrapper(x,x_m) # 处理非C维度
            tmp_x_m = tf.broadcast_to(tmp_x_m,shape=tmp_x_m.shape[0:-1]+[broadcast_indicator]) # 处理C维度
            x = tf.concat([x,tmp_x_m],axis=-1)
            tf.keras.utils.set_random_seed(1000)
            y = item(x,training=training)
            buf.append(y)
            x = y
        if buf_flag:
            return y,buf
        else:
            return y