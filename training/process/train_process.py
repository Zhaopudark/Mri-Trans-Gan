"""
将GAN的训练过程 也进行封装 以便于调试
"""
import sys
import os
import tensorflow as tf
# base = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(base,'../'))
# sys.path.append(os.path.join(base,'../../'))
import functools
from collections.abc import Iterable
__all__ = [
    'TrainProcess',
]
class TrainProcess():
    """
    抽象出训练测试的等方法
    """
    def __init__(self,args):
        """
        models,optimizers 用于普遍的通用的初始化
        """
        if hasattr(args,'xla'):
            self.xla = bool(args.xla)
        else:
            self.xla = False 
        if hasattr(args,'mixed_precision'):
            self.mixed_precision = bool(args.mixed_precision)
        else:
            self.mixed_precision = False 
    #----------------------------------------------------------------------------------------------------------#
    def _apply_gradients(self,optmizer,gradient,variable):
        optmizer.apply_gradients(zip(gradient,variable))
    def _get_scaled_loss(self,optmizer,loss):
        return optmizer.get_scaled_loss(loss)
    def _get_unscaled_gradients(self,optmizer,scared_gradient):#
        return optmizer.get_unscaled_gradients(scared_gradient)
    def train_wrapper(self,loss_func,optimizer_list=None,variable_list=None):# 只有训练过程需要计算梯度
        """
        train_step返回loss 
        """
        if optimizer_list is not None:
            assert len(optimizer_list)==len(variable_list)
        def mixed_wrappered_func(*args,**kwargs):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(variable_list)
                loss_list = loss_func(*args,**kwargs)
                scared_loss_list = list(map(self._get_scaled_loss,optimizer_list,loss_list))
            scared_gradient_list = list(map(tape.gradient,scared_loss_list,variable_list))
            gradient_list = list(map(self._get_unscaled_gradients,optimizer_list,scared_gradient_list))
            _ = list(map(self._apply_gradients,optimizer_list,gradient_list,variable_list))
            return loss_list
        def unmixed_wrappered_func(*args,**kwargs):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(variable_list)
                loss_list = loss_func(*args,**kwargs)
            gradient_list = list(map(tape.gradient,loss_list,variable_list))
            _ = list(map(self._apply_gradients,optimizer_list,gradient_list,variable_list))
            return loss_list
        if self.mixed_precision:
            print("Mixed Precision used!")
            wrappered_func = mixed_wrappered_func
        else:
            wrappered_func = unmixed_wrappered_func
        if self.xla:
            wrappered_func = tf.function(wrappered_func,jit_compile=True)
            print("XLA used in training!")
        else:
            wrappered_func = tf.function(wrappered_func)
        return wrappered_func
    def predict_wrapper(self,predict_func):
        if self.xla:
            wrappered_func = tf.function(predict_func,jit_compile=True)
            print("XLA used in predicit!")
        else:
            wrappered_func = tf.function(predict_func)
        return wrappered_func

if __name__ =='__main__':
    def fun1(x):
        print('func1')
        return x 
    def fun2(x,y):
        print('func2')
        return x+y 
    def fun3(x,y,z):
        print('func3')
        return x+y+z
    model_list = [1,2]
    optimizer_list = [4,5]
    variable_list = map(fun1,model_list)
    with tf.GradientTape(persistent=True) as tape:
        loss_list = [1,2]
    gradient_list = map(fun2,loss_list,variable_list)
    _ = map(fun3,optimizer_list,gradient_list,variable_list)
    print(_)
    
    _ = list(map(fun3,optimizer_list,gradient_list,variable_list))
    print(_)
    

    
    


