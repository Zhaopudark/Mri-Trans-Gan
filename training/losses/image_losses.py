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
from models.blocks.vgg import PerceptualLossExtractor
from training.losses._image_losses import MeanVolumeGradientError
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
        self.per_flag = bool(args.Per)
        self.per_2d_flag = bool(args.Per_2D)
        self.sty_flag = bool(args.Sty)
        self.transfer_learning_model = PerceptualLossExtractor(
                model_name=args.transfer_learning_model.lower(),
                use_feature_reco_loss=bool(self.per_flag or self.per_2d_flag),
                use_style_reco_loss=bool(self.sty_flag),
                ) # vgg16 vgg19 mobile-net
        
        #--------------------------------------------#
        if self.mae_flag:
            self.call = self.mae_wrapper(self.call)
        if self.mse_flag:
            self.call = self.mse_wrapper(self.call)
        if self.mgd_flag:
            self._mgd = MeanVolumeGradientError()
            self.call = self.mgd_wrapper(self.call)
        #--------------------------------------------#
        if self.per_d_flag:
            self.call = self.per_d_wrapper(self.call)
        #--------------------------------------------#
        if (self.per_flag)or(self.per_2d_flag)or(self.sty_flag):
            self.call = self.per_wrapper(self.call)
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
    # def per_wrapper(self,func):
    #     def call(x,x_,y,y_,xd,x_d,yd,y_d):
    #         out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
    #         out += self.per(x=x,x_=x_,y=y,y_=y_)
    #         return out
    #     return call
    # def per_2d_wrapper(self,func):
    #     def call(x,x_,y,y_,xd,x_d,yd,y_d):
    #         out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
    #         out += self.per_2d(x=x,x_=x_,y=y,y_=y_)
    #         return out
    #     return call
    # def sty_wrapper(self,func):
    #     def call(x,x_,y,y_,xd,x_d,yd,y_d):
    #         out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
    #         out += self.sty(x=x,x_=x_,y=y,y_=y_)
    #         return out
    #     return call
    #------------------------------------------------------------------#
    def mae(self,x,x_,y,y_):
        return (mae(x,x_)+mae(y,y_))/2
    def mse(self,x,x_,y,y_):
        return (mse(x,x_)+mse(y,y_))/2
    def mgd(self,x,x_,y,y_):
        return (self._mgd(x,x_)+self._mgd(y,y_))/2
    def per_d(self,x,x_,y,y_):
        buf_0,buf_1 = dual_feature_difference_list(x=x,x_=x_,y=y,y_=y_,index_begin=0,index_end=3,ords=1)#同med GAN的L1
        return (buf_0+buf_1)/2
    def per(self,x,x_,y,y_):
        per_loss_x = self.transfer_learning_model(inputs=[x,x_])
        per_loss_y = self.transfer_learning_model(inputs=[y,y_])
        return (per_loss_x+per_loss_y)/2
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

    