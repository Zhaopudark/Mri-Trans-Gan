"""
一个好的loss计算模块
计算时对batch 维度独立
返回时按照需求返回 默认在batch维度取均值
一般的 求导永远是对无维度的单值loss求导
"""

"""
构建若干装饰器 输入为rec_loss x x_ y y_ 
返回对应的
"""
import sys
import os
import tensorflow as tf
import numpy as np 
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base,'../../'))
# from blocks.vgg import Vgg16LayerBuf_V3 as Fg
from models.blocks.vgg import Vgg16LayerBuf_V4 as Fg #V4 适应3x128x128
from models.blocks.vgg import Vgg16LayerBuf_V2 as Fg2D
from models.blocks.vgg import Vgg16LayerBuf_V2 as StyFg
__all__ = [
    "CycleConsistencyLoss",
    "DualGanReconstructionLoss",
]
class CycleConsistencyLoss():
    def __init__(self,args):
        self.cc_flag = bool(args.CC)
        self.cc_l = float(args.CC_l)
        #--------------------------------------------#
        if self.cc_flag:
            self.call = self.cc_wrapper(self.call)
    #--------------------------------------------#
    def call(self,x,x__,y,y__):
        return tf.constant(0.0,dtype=tf.float32)
    def cc_wrapper(self,func):
        def call(x,x__,y,y__):
            out = func(x=x,x__=x__,y=y,y__=y__)
            out += self.cc(x=x,x__=x__,y=y,y__=y__)
            return out
        return call
    #------------------------------------------------------------------#
    def cc(self,x,x__,y,y__):
        return self.cc_l*(mae(x,x__)+mae(y,y__))
class DualGanReconstructionLoss():
    def __init__(self,args): #MAE MSE MGD Per Sty
        self.mae_flag = bool(args.MAE)
        self.mse_flag = bool(args.MSE)
        self.mgd_flag = bool(args.MGD)
        #--------------------------------------------#
        self.per_d_flag = bool(args.Per_Reuse_D)
        self.transfer_learning_model = args.transfer_learning_model.lower() # vgg16 vgg19 mobile-net
        self.per_flag = bool(args.Per)
        self.per_2d_flag = bool(args.Per_2D)
        self.sty_flag = bool(args.Sty)
        #--------------------------------------------#
        if self.mae_flag:
            self.call = self.mae_wrapper(self.call)
        if self.mse_flag:
            self.call = self.mse_wrapper(self.call)
        if self.mgd_flag:
            self.call = self.mgd_wrapper(self.call)
        #--------------------------------------------#
        if self.per_d_flag:
            self.call = self.per_d_wrapper(self.call)
        #--------------------------------------------#
        self.mixed_precision = bool(args.mixed_precision)
        if self.mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
        else:
            policy = None
        if self.transfer_learning_model =="vgg16":
            self.Fg = Fg(dtype=policy)
            self.Fg2D = Fg2D(dtype=policy)
            self.StyFg = StyFg(dtype=policy)
            self.Fg.build(input_shape=None)
            self.Fg2D.build(input_shape=None)
            self.StyFg.build(input_shape=None)
        else:
            raise ValueError("Unsupported transfer learning model:{}".format(self.transfer_learning_model))
        if self.per_flag:
            self.call = self.per_wrapper(self.call)
        if self.per_2d_flag:
            self.call = self.per_2d_wrapper(self.call)
        if self.sty_flag:
            self.call = self.sty_wrapper(self.call)
    #--------------------------------------------#
    def call(self,x,x_,y,y_,xd,x_d,yd,y_d):
        return tf.constant(0.0,dtype=tf.float32)
    def mae_wrapper(self,func):
        def call(x,x_,y,y_,xd,x_d,yd,y_d):
            out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
            out += self.mae(x=x,x_=x_,y=y,y_=y_)
            return out
        return call
    def mse_wrapper(self,func):
        def call(x,x_,y,y_,xd,x_d,yd,y_d):
            out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
            out += self.mse(x=x,x_=x_,y=y,y_=y_)
            return out
        return call
    def mgd_wrapper(self,func):
        def call(x,x_,y,y_,xd,x_d,yd,y_d):
            out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
            out += self.mgd(x=x,x_=x_,y=y,y_=y_)
            return out
        return call
    def per_d_wrapper(self,func):
        def call(x,x_,y,y_,xd,x_d,yd,y_d):
            out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
            out += self.per_d(x=xd,x_=x_d,y=yd,y_=y_d)
            return out
        return call
    def per_wrapper(self,func):
        def call(x,x_,y,y_,xd,x_d,yd,y_d):
            out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
            out += self.per(x=x,x_=x_,y=y,y_=y_)
            return out
        return call
    def per_2d_wrapper(self,func):
        def call(x,x_,y,y_,xd,x_d,yd,y_d):
            out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
            out += self.per_2d(x=x,x_=x_,y=y,y_=y_)
            return out
        return call
    def sty_wrapper(self,func):
        def call(x,x_,y,y_,xd,x_d,yd,y_d):
            out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
            out += self.sty(x=x,x_=x_,y=y,y_=y_)
            return out
        return call
    #------------------------------------------------------------------#
    def mae(self,x,x_,y,y_):
        return mae(x,x_)+mae(y,y_)
    def mse(self,x,x_,y,y_):
        return mse(x,x_)+mse(y,y_)
    def mgd(self,x,x_,y,y_):
        return mgd(x,x_)+mgd(y,y_)
    def per_d(self,x,x_,y,y_):
        buf_0,buf_1 = dual_feature_difference_list(x=x,x_=x_,y=y,y_=y_,index_begin=0,index_end=3,ords=1)#同med GAN的L1
        return buf_0+buf_1
    def per(self,x,x_,y,y_):
        per_loss_0 = 0.0
        per_loss_1 = 0.0

        feature_list_fake_0 = self.Fg(y_,training=True,scale=4) # [0,1,2,3,4]
        feature_list_real_0 = self.Fg(y ,training=True,scale=4) # [0,1,2,3,4]
        feature_list_fake_1 = self.Fg(x_,training=True,scale=4) # [0,1,2,3,4]
        feature_list_real_1 = self.Fg(x ,training=True,scale=4) # [0,1,2,3,4]
        buf_0,buf_1 = dual_feature_difference_list(x=feature_list_real_0,x_=feature_list_fake_0,y=feature_list_real_1,y_=feature_list_fake_1,index_begin=1,index_end=4,ords=2)
        per_loss_0 += (1/3)*buf_0
        per_loss_1 += (1/3)*buf_1

        feature_list_fake_0 = self.Fg(tf.transpose(y_,perm=[0,2,1,3,4]),training=True,scale=4)
        feature_list_real_0 = self.Fg(tf.transpose(y ,perm=[0,2,1,3,4]),training=True,scale=4)
        feature_list_fake_1 = self.Fg(tf.transpose(x_,perm=[0,2,1,3,4]),training=True,scale=4)
        feature_list_real_1 = self.Fg(tf.transpose(x ,perm=[0,2,1,3,4]),training=True,scale=4)
        buf_0,buf_1 = dual_feature_difference_list(x=feature_list_real_0,x_=feature_list_fake_0,y=feature_list_real_1,y_=feature_list_fake_1,index_begin=1,index_end=4,ords=2)
        per_loss_0 += (1/3)*buf_0
        per_loss_1 += (1/3)*buf_1

        feature_list_fake_0 = self.Fg(tf.transpose(y_,perm=[0,3,1,2,4]),training=True,scale=4)
        feature_list_real_0 = self.Fg(tf.transpose(y ,perm=[0,3,1,2,4]),training=True,scale=4)
        feature_list_fake_1 = self.Fg(tf.transpose(x_,perm=[0,3,1,2,4]),training=True,scale=4)
        feature_list_real_1 = self.Fg(tf.transpose(x ,perm=[0,3,1,2,4]),training=True,scale=4)
        buf_0,buf_1 = dual_feature_difference_list(x=feature_list_real_0,x_=feature_list_fake_0,y=feature_list_real_1,y_=feature_list_fake_1,index_begin=1,index_end=4,ords=2)
        per_loss_0 += (1/3)*buf_0
        per_loss_1 += (1/3)*buf_1
        return per_loss_0+per_loss_1
    def per_2d(self,x,x_,y,y_):
        per_loss_0 = 0.0
        per_loss_1 = 0.0
        slice_num = x.shape[1]
        for slice_index in range(slice_num):
            feature_list_fake_0 = self.Fg2D(y_[:,slice_index,:,:,:],training=True,scale=4)
            feature_list_real_0 = self.Fg2D( y[:,slice_index,:,:,:],training=True,scale=4)
            feature_list_fake_1 = self.Fg2D(x_[:,slice_index,:,:,:],training=True,scale=4)
            feature_list_real_1 = self.Fg2D( x[:,slice_index,:,:,:],training=True,scale=4)
            tmp_l = len(feature_list_fake_0)
            assert tmp_l==5
            for index in range(1,tmp_l,1):
                buf_0,buf_1 = dual_feature_difference_list(x=feature_list_real_0,x_=feature_list_fake_0,y=feature_list_real_1,y_=feature_list_fake_1,index_begin=1,index_end=4,ords=2)
                per_loss_0 += (1/slice_num)*buf_0
                per_loss_1 += (1/slice_num)*buf_1
        return per_loss_0+per_loss_1

    def sty(self,x,x_,y,y_):
        style_loss_0 = 0.0
        style_loss_1 = 0.0
        slice_num = x.shape[1]
        for slice_index in range(slice_num):
            feature_list_fake_0 = self.StyFg(y_[:,slice_index,:,:,:],training=True,scale=4)
            feature_list_real_0 = self.StyFg( y[:,slice_index,:,:,:],training=True,scale=4)
            feature_list_fake_1 = self.StyFg(x_[:,slice_index,:,:,:],training=True,scale=4)
            feature_list_real_1 = self.StyFg( x[:,slice_index,:,:,:],training=True,scale=4)
            tmp_l = len(feature_list_fake_0)
            assert tmp_l==5
            for index in range(1,tmp_l,1):
                style_loss_0 += (1/(tmp_l-1)/slice_num)*style_diff_2D(feature_list_real_0[index],feature_list_fake_0[index])
                style_loss_1 += (1/(tmp_l-1)/slice_num)*style_diff_2D(feature_list_real_1[index],feature_list_fake_1[index])
        # tf.print(style_loss_0.shape)
        # tf.print(style_loss_1)
        return style_loss_0+style_loss_1
#---------------------------------------------------------------------------------------------------------------------------------#
def mae(x,y):
    # tf.print("mae")
    return tf.reduce_mean(tf.abs(x-y))
def mae2(x,y): #但是为了计算速度 不采用此方法
    b = x.shape[0]
    M = 1
    for i in range(1,len(x.shape),1):
        M *= x.shape[i]
    norm = tf.norm(tf.reshape(x-y, shape=[b,-1]),ord=1,axis=-1)/M
    return tf.reduce_mean(norm)
def mse(x,y):
    return tf.reduce_mean(tf.math.square(x-y))
def mgd(x,y):
    if len(x.shape)==5:
        dz1,dy1,dx1 = pix_gradient_3D(x)
        dz2,dy2,dx2 = pix_gradient_3D(y)
        return tf.reduce_mean(tf.abs(dz1-dz2))/3 + tf.reduce_mean(tf.abs(dy1-dy2))/3 + tf.reduce_mean(tf.abs(dx1-dx2))/3
    elif len(x.shape)==4:
        dy1,dx1 = pix_gradient_2D(x)
        dy2,dx2 = pix_gradient_2D(y)
        return tf.reduce_mean(tf.abs(dy1-dy2))/2 + tf.reduce_mean(tf.abs(dx1-dx2))/2
    else:
        raise ValueError("mgd only support for 4 dims or 5 dims.")
def pix_gradient_2D(img): #shape=[b,h(y),w(x),c] 计算(x, y)点dx为[I(x+1,y)-I(x, y)] 末端pad 0 
    dx = img[:,:,1::,:]-img[:,:,0:-1,:]
    dy = img[:,1::,:,:]-img[:,0:-1,:,:]
    dx = tf.pad(dx,paddings=[[0,0],[0,0],[0,1],[0,0]]) # 末端pad 0
    dy = tf.pad(dy,paddings=[[0,0],[0,1],[0,0],[0,0]]) # 末端pad 0 
    return dy,dx
def pix_gradient_3D(img): #shape=[b,d(z),h(y),w(x),c] 计算(x,y,z)点dx为[I(x+1,y,z)-I(x,y,z)] 末端pad 0 
    dx = img[:,:,:,1::,:]-img[:,:,:,0:-1,:]
    dy = img[:,:,1::,:,:]-img[:,:,0:-1,:,:]
    dz = img[:,1::,:,:,:]-img[:,0:-1,:,:,:]
    dx = tf.pad(dx,paddings=[[0,0],[0,0],[0,0],[0,1],[0,0]]) # 末端pad 0
    dy = tf.pad(dy,paddings=[[0,0],[0,0],[0,1],[0,0],[0,0]]) # 末端pad 0 
    dz = tf.pad(dz,paddings=[[0,0],[0,1],[0,0],[0,0],[0,0]]) # 末端pad 0
    return dz,dy,dx
def feature_difference(x,y,ords=1):#计算2D或3D情况下特征图x,y的差 
    # 涉及运算效率，不进行输入检查
    # tf.print("Vper",ords)
    if ords==1:
        return tf.reduce_mean(tf.abs(x-y)) # 在包括batch的维度取均值
    elif ords==2:
        return tf.reduce_mean(tf.square(x-y))# 在包括batch的维度取均值
    else:
        raise ValueError("Unsupported ords")
def feature_difference_list(x,x_,index_begin=1,ords=1):#计算2D或3D情况下一系列特征图x,y的差 并保持权值总和1 index_begin 默认为1 跳过第0特征图
    buf = 0.0
    l = len(x)-index_begin
    for index in range(index_begin,len(x),1):
        buf += (1/l)*feature_difference(x[index],x_[index],ords=ords)
    return buf
def dual_feature_difference_list(x,x_,y,y_,index_begin,index_end,ords=1):#计算2D或3D情况下一系列特征图x,y的差 并保持权值总和1 index_begin 默认为1 跳过第0特征图
    buf_0,buf_1 = 0.0,0.0
    l = index_end-index_begin+1
    for index in range(index_begin,index_end+1,1):
        buf_0 += (1/l)*feature_difference(x[index],x_[index],ords=ords)
        buf_1 += (1/l)*feature_difference(y[index],y_[index],ords=ords)
    return buf_0,buf_1
def grma_2D(x):
    b,h,w,c = x.shape
    m = tf.reshape(x,[b,-1,c])
    m_T = tf.transpose(m,perm=[0,2,1])
    g = (1.0/(h*w*c))*tf.matmul(m_T,m)
    # tf.print(tf.reduce_mean(g))
    return g # [B,C,C]
def grma_3D(x):
    b,d,h,w,c = x.shape
    m = tf.reshape(x,[b,-1,c])
    m_T = tf.transpose(m,perm=[0,2,1])
    g = (1.0/(d*h*w*c))*tf.matmul(m_T,m)
    return g # [B,C,C]
def style_diff_2D(x,y):
    style_diff = tf.reduce_mean(tf.square(tf.norm(grma_2D(x)-grma_2D(y),ord="fro",axis=[1,2]))) # 在batch 维度取均值
    return style_diff
def style_diff_3D(x,y):
    style_diff = tf.reduce_mean(tf.square(tf.norm(grma_3D(x)-grma_3D(y),ord="fro",axis=[1,2]))) # 在batch 维度取均值
    return style_diff
    