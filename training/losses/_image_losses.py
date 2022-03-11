"""
一个好的loss计算模块
计算时对batch 维度独立
返回时按照需求返回 默认在batch维度取均值
一般的 求导永远是对无维度的单值loss求导
 So, in this class, it is not appropriate to consider rightmost first. The broadcast method should be:
            Starting with the heading (i.e. leftmost) dimensions and works its way right. In the backend of keras.losses.Loss,
            the broadcast behavior of sample_weight follows:
                1. Squeezes or expands `last` dim of `sample_weight` if its rank differs by 1
                from the new rank of temp_loss.(`last` dim should be 1 if squeeze it)
                2. If `sample_weight` is scalar, it is kept scalar.

"""

"""
构建若干装饰器 输入为rec_loss x x_ y y_ 
返回对应的
"""
import sys
import os
import logging
import copy
from typeguard import typechecked
import numpy as np 
import tensorflow as tf
__all__ = [
    "MeanAbsoluteError",
    "MeanSquaredError",
    "MeanVolumeGradientError",
    "MeanFeatureReconstructionError",
    "MeanStyleReconstructionError",
]

class LossAcrossListWrapper(tf.keras.losses.Loss):
    """Abstract wrapper base loss.
    Wrappers take another layer and augment it in various ways.
    """
    @typechecked
    def __init__(self,loss:tf.keras.losses.Loss,**kwargs):
        self.inner_loss = loss
        kwargs["reduction"] = tf.keras.losses.Reduction.AUTO
        super().__init__(**kwargs)
    def call(self,y_true,y_pred):
        buf = []
        for s_y_true,s_y_pred in zip(y_true,y_pred):
            buf.append(tf.reduce_mean(self.inner_loss.call(s_y_true,s_y_pred)))
        return buf

    def get_config(self):
        config = {'inner_loss': tf.keras.losses.serialize(self.inner_loss)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = copy.deepcopy(config)
        loss = tf.keras.losses.deserialize(config.pop('inner_loss'),custom_objects=custom_objects)
        return cls(loss, **config)


class MeanAbsoluteError(tf.keras.losses.MeanAbsoluteError):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)

class MeanSquaredError(tf.keras.losses.MeanSquaredError):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)
    
class MeanVolumeGradientError(tf.keras.losses.Loss):
    """
    Calculation Loss(MeanVolumeGradientError) between 2 tensors, y_true and y_pred
    y_true and y_pred should in the same shape
    consider they are N dimensions tensors in shape [B,D1,D2,...,D{N-2},C] (channels_last)
    or [B,C,D1,D2,...,D{N-2}] (channels_first)
    B is the real batch dimension (not broad sense batch dimension)
    C is the channels dimension
    D1 to D{N-2} is the meaningful dimension. 
        If y_true is a 2D image tensor, N==4, then "D1,D2" represent the "H(height),W(weight)" dimensions.
        If y_true is a 3D image tensor, N==5, then "D1,D2,D3" represent the "D(depth),H(height),W(weight)" dimensions.
        ...
    This loss func will work as the following 3 steps:
        1. Calculate the volume_gradient on each meaningful dimension of y_true and y_pred respectively, got 2 list of tensor, volume_gradient_true and volume_gradient_pred.
        2. Through backend loss function, calculate the loss between 2 elements from volume_gradient_true and volume_gradient_pred's corresponding position item by item, got a list of tensor, volume_gradient_error.
            NOTE  here we give a choice to users whether use 1-norm or 2-norm as backend loss function
        3. If given sample_weight, apply sample_weight on volume_gradient_error first. Then, reduce on volume_gradient_error.
    
    About shape,
    p.s., list/tuple can be regarded as broad sense tensor.
    Here, we specify 3 behavior about shape and shape's reduction:
        if y_true in shape of [B,D1,D2,...,D{N-2},C] (channels_last)
        1. prefix behaior:
            from y_true and y_pred to get volume_gradient_true and volume_gradient_pred, in shape of [{N-2},B,D1,D2,...,D{N-2},C]
            transpose to  [B,{N-2},D1,D2,...,D{N-2},C]
        2. backend loss function's reduction behavior
            by `VolumeGradient` definition, reduce meaningful dimensions, i.e., `D1,D2,...,D{N-2}` dimensions.
            volume_gradient_error in shape of [B,{N-2},C]
        3. sample_weight application behavior
            sample_weight in this class represtents the weight over "B, C or {N-2}" dimensions,
            it means giving loss results differents weighs on different batches, channels or meaningful dimensions (which dimension has larger proportion and which has less)

            Sample_weight will multiply with volume_gradient_error.

            Since sample_weight's broadcast behavior in traditional `tf.keras.losses.Loss` in very special:
                1. In typical application, loss function's sample_weight acts as a coefficient for the loss, and losses usually conduct in 'axis=-1'. 
                2. Squeezes or expands `last` dim of `sample_weight` if its rank differs by 1
                from the new rank of temp_loss(volume_gradient_error).(`last` dim should be 1 if squeeze it)
                3. If `sample_weight` is scalar, it is kept scalar.
            Additionally a general multiply broadcast behavior follows: 
                https://numpy.org/doc/stable/user/basics.broadcasting.html
                It starts with the trailing (i.e. rightmost) dimensions and works its way left. Two dimensions are compatible when
                    1. they are equal, or
                    2. one of them is 1
            So, if we do nothing,  sample_weight's broadcast behavior will follow `tf.keras.losses.Loss`'s constraint, it may leads problems 
            If we give a sample_weight in shape [N-2] and wants it work on [B,{N-2},C] results. i.e., exceptions will be encountered.

            If we  prevent orginal `tf.keras.losses.Loss`'s sample_weight's broadcast behavior, manually broadcast sample_weight to [B,{N-2},C] first,
            the  potential risk will be eliminated.

            So, supported sample_weight shape is 
                []
                [C],[1],[N-2],  --> we additional make it support [N-2], beacuse MeanVolumeGradientError actually concern about the meaningful dimension
                [N-2,C],[N-2,1],[1,C],[1,1]
                [B,N-2,C],[B,N-2,1],[B,1,C],[1,N-2,C],[B,1,1],[1,N-2,1],[1,1,C],[1,1,1]

    >>> loss = MeanVolumeGradientError()
    >>> y_true = tf.random.normal(shape=[4,5,6])
    >>> y_pred = tf.random.normal(shape=[4,5,6])
    >>> y = loss(y_true,y_pred)
    >>> print(y.shape)
    (4, 1, 6)

    >>> loss = MeanVolumeGradientError()
    >>> y_true = tf.random.normal(shape=[4,5,7,9,6])
    >>> y_pred = tf.random.normal(shape=[4,5,7,9,6])
    >>> sample_weight = tf.random.uniform(shape=[4,3,6])
    >>> y = loss(y_true,y_pred,sample_weight)
    >>> print(y.shape)
    (4, 3, 6)
    
    """
    @typechecked
    def __init__(self,mode:str="L1",name:str="mean_volume_gradient_error",data_format:str="channels_last",**kwargs) -> None:
        if "reduction" in kwargs.keys():
            if kwargs["reduction"] != tf.keras.losses.Reduction.NONE:
                logging.warning("""
                    MeanVolumeGradientError will reduce output by its own reduction. 
                    Setting `reduction` is ineffective.
                    Output will be reduce to [B,{N-2},C] shape whether the input_shape is 
                    [B,D1,D2,...,D{N-2},C] (channels last) or [B,C,D1,D2,...,D{N-2}] (channels first)
                    """)
        kwargs["reduction"] = tf.keras.losses.Reduction.NONE
        super().__init__(name=name,**kwargs)
        self.loss_kwargs = {}
        self.loss_kwargs["mode"] = mode
        self.loss_kwargs["data_format"] = data_format
        if mode.upper() == "L1":
            self.loss_func = self._l1_loss
        elif mode.upper() == "L2":
            self.loss_func = self._l2_loss
        else:
            raise ValueError("MeanVolumeGradientError's inner loss mode only support in 'L1' or 'L2', not{}".format(mode))
        self.data_format = data_format.lower()
        if self.data_format not in ["channels_last","channels_first"]:
            raise ValueError("data_format should be one in 'channels_last' or'channels_first', not {}.".format(data_format))
    def _l1_loss(self,x1,x2):
        _axis = self._get_reduce_axis(x1)
        return tf.reduce_mean(tf.abs(x1-x2),axis=_axis)
    def _l2_loss(self,x1,x2): 
        _axis = self._get_reduce_axis(x1)
        return tf.reduce_mean(tf.square(x1-x2),axis=_axis)
    def _get_volume_gradient(self,input):
        out_buf = []
        valid_indexs = self._get_meaningful_axes(input)
        begin = [0,]*len(input.shape)
        size = input.shape.as_list() # total volume index 0<->N-1
        paddings = [[0,0],]*len(input.shape)
        for index in valid_indexs:
            _begin = begin[:]
            _size = size[:]
            _size[index] -= 1 
            base = tf.slice(input,begin=_begin,size=_size) # base volume index 0<->N-2

            _begin = begin[:]
            _begin[index] += 1
            _size = size[:]
            _size[index] -= 1 
            base_move_1 = tf.slice(input,begin=_begin,size=_size) #base_move volume index 1<->N-1
            diff = base_move_1-base
            _paddings = paddings[:]
            _paddings[index] = [0,1]
        
            diff = tf.pad(diff,paddings=_paddings)
            out_buf.append(diff) # out volume == (volume index 1<->N-1) - (volume index 0<->N-2)
        return tf.stack(out_buf,axis=1)
    def _get_meaningful_axes(self,x): # indicate `D1,D2,...,D{N-2}`
        rank = len(x.shape)-2
        assert rank>=0
        if self.data_format == "channels_last":
            return list(range(len(x.shape)))[1:-1:]
        else:
            return list(range(len(x.shape)))[2::]
    def _get_reduce_axis(self,x): # indicate `{N-2},D1,D2,...,D{N-2}`
        rank = len(x.shape)-2
        assert rank>=0
        if self.data_format == "channels_last":
            return list(range(len(x.shape)))[2:-1:] # indicate`D1,D2,...,D{N-2}` in [B,{N-2},D1,D2,...,D{N-2},C]
        else:
            return list(range(len(x.shape)))[3::] # indicate`D1,D2,...,D{N-2}` in [B,{N-2},C,D1,D2,...,D{N-2}]
    def _expand_or_squeeze_sample_weight(self,y_true,y_pred,sample_weight=None):
        if sample_weight is not None:
            sample_weight = tf.convert_to_tensor(sample_weight)
            if self.data_format == "channels_last":
                loss_shape = [y_true.shape[0],len(y_true.shape)-2,y_true.shape[-1]]
            else:
                loss_shape = [y_true.shape[0],len(y_true.shape)-2,y_true.shape[1]]
            if (len(sample_weight.shape)==1) and (sample_weight.shape[0]==loss_shape[1]): # suppose it indicate `N-2` dimension
                    sample_weight = tf.expand_dims(sample_weight,axis=0) # [1,N-2]
                    sample_weight = tf.expand_dims(sample_weight,axis=-1) # [1,N-2,1]
            sample_weight = tf.broadcast_to(sample_weight,shape=loss_shape) # if shape un-matched, raise error
        else:
            return sample_weight
    def call(self,y_true,y_pred):
        y_true_gradient = self._get_volume_gradient(y_true)
        y_pred_gradient = self._get_volume_gradient(y_pred)      
        return self.loss_func(y_true_gradient,y_pred_gradient)
    def __call__(self,y_true,y_pred,sample_weight=None):
        sample_weight = self._expand_or_squeeze_sample_weight(y_true,y_pred,sample_weight)
        return super().__call__(y_true,y_pred,sample_weight)
    def get_config(self):
        config = {}
        if hasattr(self,"loss_kwargs"):
            config = {**self.loss_kwargs}     
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MeanFeatureReconstructionError(tf.keras.losses.Loss):
    """
    see: https://arxiv.org/pdf/1603.08155.pdf
    Calculation Loss(MeanFeatureReconstructionError) between 2 tensors (feature maps), y_true and y_pred
    y_true and y_pred should in the same shape
    consider they are N dimensions tensors in shape [B,D1,D2,...,D{N-1}] 
    B is the real batch dimension (not broad sense batch dimension)
    D1 to D{N-1} is the meaningful dimension. 
        e.g.,
        If in channels_last data_format
            If y_true is a 2D image tensor, N==4, then "D1,D2,D3" represent the "H(height),W(weight),C(channels)" dimensions.
            If y_true is a 3D image tensor, N==5, then "D1,D2,D3,D4" represent the "D(depth),H(height),W(weight),C(channels)" dimensions.
            ...
    Since calculate the mean over batch dimension may leads to problems, here 
    we will maintain the batch dimension
    define a loss func:
        axis = [1,2,...,N-1]
        loss_func(y_true,y_pred)=tf.reduce_mean(tf.square(y_true-y_pred),axis=axis) # shape = [B]
        NOTE some papers use 1-norm when practice, so here we give a choice to users.
            such as https://ieeexplore.ieee.org/abstract/document/8653423

    This loss func will work as the following 2 steps:
        1. Calculate the difference of y_true and y_pred, representing their "Feature Difference":
        2. Through backend loss function, calculate the loss between "Feature Difference".
            NOTE here we give a choice to users whether use 1-norm or 2-norm as backend loss function
        3. If given sample_weight, apply sample_weight on loss results first. Then, reduce on loss results.
    About shape,
    p.s., list/tuple can be regarded as broad sense tensor.
    Here, we specify 3 behavior about shape and shape's reduction:
        if y_true in shape of [B,D1,D2,...,D{N-1}]
        1. from y_true and y_pred to get tf.square(y_true-y_pred) or tf.abs(y_true-y_pred)
           matain the shape [B,D1,D2,...,D{N-1}]
        2. tf.reduce_mean(), reduce and mean on  `D1,D2,...,D{N-1}` dimensions, got a [B] shape tensor 
        3. sample_weight application behavior
            the same as MeanVolumeGradientError
            sample_weight in this class represtents the weight over "B" dimensions,
            it means giving loss results differents weighs on different batches
            So, supported sample_weight shape is 
                []
                [B],[1],  
            There is no need to broadcast  sample_weight manually.

    >>> loss = MeanFeatureReconstructionError()
    >>> y_true = tf.random.normal(shape=[4,5,6])
    >>> y_pred = tf.random.normal(shape=[4,5,6])
    >>> y = loss(y_true,y_pred)
    >>> print(y.shape)
    (4,)

    >>> loss = MeanFeatureReconstructionError()
    >>> y_true = tf.random.normal(shape=[4,5,7,9,6])
    >>> y_pred = tf.random.normal(shape=[4,5,7,9,6])
    >>> sample_weight = tf.random.uniform(shape=[4])
    >>> y = loss(y_true,y_pred,sample_weight)
    >>> print(y.shape)
    (4,)
    
    """
    @typechecked
    def __init__(self,mode:str="L1",name:str="mean_feat_reco_error",**kwargs) -> None:
        if "reduction" in kwargs.keys():
            if kwargs["reduction"] != tf.keras.losses.Reduction.NONE:
                logging.warning(
                    """
                    MeanFeatureReconstructionError will reduce output by its own reduction. 
                    Setting `reduction` is ineffective.
                    Output will be reduce to [B] shape always.               
                    """)
        kwargs["reduction"] = tf.keras.losses.Reduction.NONE
        super().__init__(name=name,**kwargs)
        self.loss_kwargs = {}
        self.loss_kwargs["mode"] = mode
        if mode.upper() == "L1":
            self.loss_func = self._l1_loss
        elif mode.upper() == "L2":
            self.loss_func = self._l2_loss
        else:
            raise ValueError("MeanFeatureReconstructionError's inner loss mode only support in 'L1' or 'L2', not{}".format(mode))
    def _get_reduce_or_norm_axis(self,x):
        return list(range(len(x.shape)))[1::]
    def _l1_loss(self,x1,x2): # [B,x,x,...]
        _axis = self._get_reduce_or_norm_axis(x1)
        return tf.reduce_mean(tf.abs(x1-x2),axis=_axis)
    def _l2_loss(self,x1,x2): # [B,x,x,...]
        _axis = self._get_reduce_or_norm_axis(x1)
        return tf.reduce_mean(tf.square(x1-x2),axis=_axis)
    def call(self,y_true,y_pred):    
        return self.loss_func(y_true,y_pred)
    def get_config(self):
        config = {}
        if hasattr(self,"loss_kwargs"):
            config = {**self.loss_kwargs}     
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MeanStyleReconstructionError(tf.keras.losses.Loss):
    """
    see: https://arxiv.org/pdf/1603.08155.pdf
    Calculation Loss(MeanStyleReconstructionError) between 2 tensors (feature maps), y_true and y_pred
    y_true and y_pred should in the same shape

    Consider they are N dimensions tensors in shape [B,D1,D2,...,D{N-2},C] (channels_last) or [B,C,D1,D2,...,D{N-2}] (channels_first)
    B is the real batch dimension (not broad sense batch dimension)
    C is the channels dimension
    D1 to D{N-2} is the meaningful dimension. 
        If y_true is a 2D image tensor, N==4, then "D1,D2" represent the "H(height),W(weight)" dimensions.
        If y_true is a 3D image tensor, N==5, then "D1,D2,D3" represent the "D(depth),H(height),W(weight)" dimensions.
        ...

    Since calculate the mean over batch dimension may leads to problems, here 
    we will maintain the batch dimension
    define a loss func:
        loss_func(y_true,y_pred)=tf.suqare(tf.norm(get_gram_matrix(y_ture)-get_gram_matrix(y_pred),ord="fro",axis=[-2,-1])) # [B]
    It is difficult to understand, so here is the explaination:

    This loss func will work as the following 3 steps:
        1. Calculate the Gram Matrix of y_true and y_pred by get_gram_matrix() function, representing their 'style':
            for example: consider y_true in [B,D1,D3,D4,...,D{N-1},C] or [B,C,D1,D3,D4,...,D{N-1}] shape
                1. reshape y_true to [B,-1,C] or [B,C,-1]
                2. matmul y_true and y_true.T in last 2 dimension got shape [B,C,C]
                3. divive the matmuled result by total volume nums over all dimensions, except batch dimension
        2. Through backend loss function, calculate the loss of Gram Matrix difference of y_ture and y_pred.
            tf.suqare(tf.norm()) is the backend loss function, i.e., square of `Matrix F-norm`
        3. If given sample_weight, apply sample_weight on loss results first. Then, reduce on loss results.
        
        NOTE if we consider y_true a `Ones` tensor in shape [B,D1,D3,D4,...,D{N-1},C] or [B,C,D1,D3,D4,...,D{N-1}]
        Then, get_gram_matrix(y_true) will give out a tensor in [B,C,C], each element is 1/C 
        and then, tf.suqare(tf.norm(·)) will give out a tensor in [B],each element is (1/c**2+1/C**2+...+1/C**2) == 1
        So the mean `energy` of each point isn't changed after style difference calculation, i.e., mean `energy` is not affected by the number of volumes

    About shape,
    p.s., list/tuple can be regarded as broad sense tensor.
    Here, we specify 3 behavior about shape and shape's reduction:
        if y_true in shape of [B,D1,D2,...,D{N-1},C] or [B,C,D1,D3,D4,...,D{N-1}]
        1. get the Gram Matrix
           make the results in shape [B,C,C]
        2. square of `Matrix F-norm`
           reduce and nrom on  `C,C` dimensions, got a [B] shape tensor 
        3. sample_weight application behavior
            the same as MeanVolumeGradientError
            sample_weight in this class represtents the weight over "B" dimensions,
            it means giving loss results differents weighs on different batches
            So, supported sample_weight shape is 
                []
                [B],[1],  
            There is no need to broadcast  sample_weight manually.

    >>> loss = MeanStyleReconstructionError()
    >>> y_true = tf.random.normal(shape=[4,5,6])
    >>> y_pred = tf.random.normal(shape=[4,5,6])
    >>> y = loss(y_true,y_pred)
    >>> print(y.shape)
    (4,)

    >>> loss = MeanStyleReconstructionError()
    >>> y_true = tf.random.normal(shape=[4,5,7,9,6])
    >>> y_pred = tf.random.normal(shape=[4,5,7,9,6])
    >>> sample_weight = tf.random.uniform(shape=[4])
    >>> y = loss(y_true,y_pred,sample_weight)
    >>> print(y.shape)
    (4,)

    """
    @typechecked
    def __init__(self,name:str="mean_style_reco_error",data_format:str="channels_last",**kwargs) -> None:
        if "reduction" in kwargs.keys():
            if kwargs["reduction"] != tf.keras.losses.Reduction.NONE:
                logging.warning(
                    """
                    MeanStyleReconstructionError will reduce output by its own reduction. 
                    Setting `reduction` is ineffective.
                    Output will be reduce to [B] shape always.               
                    """)
        kwargs["reduction"] = tf.keras.losses.Reduction.NONE
        super().__init__(name=name,**kwargs)
        self.loss_kwargs = {}
        self.loss_kwargs["data_format"] = data_format
        self.data_format = data_format.lower()
        if self.data_format not in ["channels_last","channels_first"]:
            raise ValueError("data_format should be one in 'channels_last' or'channels_first', not {}.".format(data_format))
    def _get_gram_matrix(self,x):
        _shape = x.shape
        _total_volume_num = tf.cast(tf.math.reduce_prod(_shape[1::]),x.dtype)
        if self.data_format == "channels_last":
            _shape = [_shape[0]]+[-1]+[_shape[-1]] # [B,-1,C]
            _transpose_a = True # [B,C,-1]
            _transpose_b = False # [B,-1,C]
            #  [B,C,-1] @ [B,-1,C]  = [B,C,C]
        elif self.data_format == "channels_first":
            _shape = [_shape[0]]+[_shape[1]]+[-1] # [B,C,-1]
            _transpose_a = False # [B,C,-1]
            _transpose_b = True  # [B,-1,C]
            # [B,C,-1] @ [B,-1,C]  = [B,C,C]
        else:
            raise ValueError("data_format should be one in 'channels_last' or'channels_first', not {}.".format(self.data_format))
        x = tf.reshape(x,_shape,name="flattened_tensor_for_gram_matrix")
        return tf.matmul(x,x,transpose_a=_transpose_a,transpose_b=_transpose_b)/_total_volume_num
    def call(self,x1,x2):
        x1 = self._get_gram_matrix(x1) # [B,C,C]
        x2 = self._get_gram_matrix(x2) # [B,C,C]
        return tf.square(tf.norm(x1-x2,ord="fro",axis=[-2,-1])) # [B]
    def get_config(self):
        config = {}
        if hasattr(self,"loss_kwargs"):
            config = {**self.loss_kwargs}     
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)  

    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    loss1 = MeanVolumeGradientError()
    _loss = MeanVolumeGradientError()
    loss2 = LossAcrossListWrapper(_loss)

    feature_1 = tf.random.normal(shape=[7,4,5,6,3])
    feature_2 = tf.random.normal(shape=[7,4,5,6,4])
    feature_3 = tf.random.normal(shape=[7,4,5,6,5])
    feature_4 = tf.random.normal(shape=[7,4,5,6,3])
    feature_5 = tf.random.normal(shape=[7,4,5,6,4])
    feature_6 = tf.random.normal(shape=[7,4,5,6,5])
    y_true = [feature_1,feature_2,feature_3]
    y_pred = [feature_4,feature_5,feature_6]
    y1 = 1/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 1/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 1/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y2 = loss2(y_true,y_pred)
    computed = tf.reduce_mean(y1-y2)
    print(computed)
    y1 = 2/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 1/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 1/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y2 = loss2(y_true,y_pred,[2,1,1])
    computed = tf.reduce_mean(y1-y2)
    print(computed)

    # y = loss(y_true,y_pred)
    # print(y)
    # print(feature_1.shape)
    # feature_1 = tf.stack([feature_1,feature_1,feature_1,feature_1],axis=1)
    # print(feature_1.shape)




    
# class DualGanReconstructionLoss():
#     def __init__(self,args): #MAE MSE MGD Per Sty
#         self.mae_flag = bool(args.MAE)
#         self.mse_flag = bool(args.MSE)
#         self.mgd_flag = bool(args.MGD)
#         #--------------------------------------------#
#         self.per_d_flag = bool(args.Per_Reuse_D)
#         self.transfer_learning_model = args.transfer_learning_model.lower() # vgg16 vgg19 mobile-net
#         self.per_flag = bool(args.Per)
#         self.per_2d_flag = bool(args.Per_2D)
#         self.sty_flag = bool(args.Sty)
#         #--------------------------------------------#
#         if self.mae_flag:
#             self.call = self.mae_wrapper(self.call)
#         if self.mse_flag:
#             self.call = self.mse_wrapper(self.call)
#         if self.mgd_flag:
#             self.call = self.mgd_wrapper(self.call)
#         #--------------------------------------------#
#         if self.per_d_flag:
#             self.call = self.per_d_wrapper(self.call)
#         #--------------------------------------------#
#         self.mixed_precision = bool(args.mixed_precision)
#         if self.mixed_precision:
#             policy = tf.keras.mixed_precision.Policy('mixed_float16')
#         else:
#             policy = None
#         if self.transfer_learning_model =="vgg16":
#             self.Fg = Fg(dtype=policy)
#             self.Fg2D = Fg2D(dtype=policy)
#             self.StyFg = StyFg(dtype=policy)
#             self.Fg.build(input_shape=None)
#             self.Fg2D.build(input_shape=None)
#             self.StyFg.build(input_shape=None)
#         else:
#             raise ValueError("Unsupported transfer learning model:{}".format(self.transfer_learning_model))
#         if self.per_flag:
#             self.call = self.per_wrapper(self.call)
#         if self.per_2d_flag:
#             self.call = self.per_2d_wrapper(self.call)
#         if self.sty_flag:
#             self.call = self.sty_wrapper(self.call)
#     #--------------------------------------------#
#     def call(self,x,x_,y,y_,xd,x_d,yd,y_d):
#         return tf.constant(0.0,dtype=tf.float32)
#     def mae_wrapper(self,func):
#         def call(x,x_,y,y_,xd,x_d,yd,y_d):
#             out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
#             out += self.mae(x=x,x_=x_,y=y,y_=y_)
#             return out
#         return call
#     def mse_wrapper(self,func):
#         def call(x,x_,y,y_,xd,x_d,yd,y_d):
#             out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
#             out += self.mse(x=x,x_=x_,y=y,y_=y_)
#             return out
#         return call
#     def mgd_wrapper(self,func):
#         def call(x,x_,y,y_,xd,x_d,yd,y_d):
#             out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
#             out += self.mgd(x=x,x_=x_,y=y,y_=y_)
#             return out
#         return call
#     def per_d_wrapper(self,func):
#         def call(x,x_,y,y_,xd,x_d,yd,y_d):
#             out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
#             out += self.per_d(x=xd,x_=x_d,y=yd,y_=y_d)
#             return out
#         return call
#     def per_wrapper(self,func):
#         def call(x,x_,y,y_,xd,x_d,yd,y_d):
#             out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
#             out += self.per(x=x,x_=x_,y=y,y_=y_)
#             return out
#         return call
#     def per_2d_wrapper(self,func):
#         def call(x,x_,y,y_,xd,x_d,yd,y_d):
#             out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
#             out += self.per_2d(x=x,x_=x_,y=y,y_=y_)
#             return out
#         return call
#     def sty_wrapper(self,func):
#         def call(x,x_,y,y_,xd,x_d,yd,y_d):
#             out = func(x=x,x_=x_,y=y,y_=y_,xd=xd,x_d=x_d,yd=yd,y_d=y_d)
#             out += self.sty(x=x,x_=x_,y=y,y_=y_)
#             return out
#         return call
#     #------------------------------------------------------------------#
#     def mae(self,x,x_,y,y_):
#         return mae(x,x_)+mae(y,y_)
#     def mse(self,x,x_,y,y_):
#         return mse(x,x_)+mse(y,y_)
#     def mgd(self,x,x_,y,y_):
#         return mgd(x,x_)+mgd(y,y_)
#     def per_d(self,x,x_,y,y_):
#         buf_0,buf_1 = dual_feature_difference_list(x=x,x_=x_,y=y,y_=y_,index_begin=0,index_end=3,ords=1)#同med GAN的L1
#         return buf_0+buf_1
#     def per(self,x,x_,y,y_):
#         per_loss_0 = 0.0
#         per_loss_1 = 0.0

#         feature_list_fake_0 = self.Fg(y_,training=True,scale=4) # [0,1,2,3,4]
#         feature_list_real_0 = self.Fg(y ,training=True,scale=4) # [0,1,2,3,4]
#         feature_list_fake_1 = self.Fg(x_,training=True,scale=4) # [0,1,2,3,4]
#         feature_list_real_1 = self.Fg(x ,training=True,scale=4) # [0,1,2,3,4]
#         buf_0,buf_1 = dual_feature_difference_list(x=feature_list_real_0,x_=feature_list_fake_0,y=feature_list_real_1,y_=feature_list_fake_1,index_begin=1,index_end=4,ords=2)
#         per_loss_0 += (1/3)*buf_0
#         per_loss_1 += (1/3)*buf_1

#         feature_list_fake_0 = self.Fg(tf.transpose(y_,perm=[0,2,1,3,4]),training=True,scale=4)
#         feature_list_real_0 = self.Fg(tf.transpose(y ,perm=[0,2,1,3,4]),training=True,scale=4)
#         feature_list_fake_1 = self.Fg(tf.transpose(x_,perm=[0,2,1,3,4]),training=True,scale=4)
#         feature_list_real_1 = self.Fg(tf.transpose(x ,perm=[0,2,1,3,4]),training=True,scale=4)
#         buf_0,buf_1 = dual_feature_difference_list(x=feature_list_real_0,x_=feature_list_fake_0,y=feature_list_real_1,y_=feature_list_fake_1,index_begin=1,index_end=4,ords=2)
#         per_loss_0 += (1/3)*buf_0
#         per_loss_1 += (1/3)*buf_1

#         feature_list_fake_0 = self.Fg(tf.transpose(y_,perm=[0,3,1,2,4]),training=True,scale=4)
#         feature_list_real_0 = self.Fg(tf.transpose(y ,perm=[0,3,1,2,4]),training=True,scale=4)
#         feature_list_fake_1 = self.Fg(tf.transpose(x_,perm=[0,3,1,2,4]),training=True,scale=4)
#         feature_list_real_1 = self.Fg(tf.transpose(x ,perm=[0,3,1,2,4]),training=True,scale=4)
#         buf_0,buf_1 = dual_feature_difference_list(x=feature_list_real_0,x_=feature_list_fake_0,y=feature_list_real_1,y_=feature_list_fake_1,index_begin=1,index_end=4,ords=2)
#         per_loss_0 += (1/3)*buf_0
#         per_loss_1 += (1/3)*buf_1
#         return per_loss_0+per_loss_1
#     def per_2d(self,x,x_,y,y_):
#         per_loss_0 = 0.0
#         per_loss_1 = 0.0
#         slice_num = x.shape[1]
#         for slice_index in range(slice_num):
#             feature_list_fake_0 = self.Fg2D(y_[:,slice_index,:,:,:],training=True,scale=4)
#             feature_list_real_0 = self.Fg2D( y[:,slice_index,:,:,:],training=True,scale=4)
#             feature_list_fake_1 = self.Fg2D(x_[:,slice_index,:,:,:],training=True,scale=4)
#             feature_list_real_1 = self.Fg2D( x[:,slice_index,:,:,:],training=True,scale=4)
#             tmp_l = len(feature_list_fake_0)
#             assert tmp_l==5
#             for index in range(1,tmp_l,1):
#                 buf_0,buf_1 = dual_feature_difference_list(x=feature_list_real_0,x_=feature_list_fake_0,y=feature_list_real_1,y_=feature_list_fake_1,index_begin=1,index_end=4,ords=2)
#                 per_loss_0 += (1/slice_num)*buf_0
#                 per_loss_1 += (1/slice_num)*buf_1
#         return per_loss_0+per_loss_1

#     def sty(self,x,x_,y,y_):
#         style_loss_0 = 0.0
#         style_loss_1 = 0.0
#         slice_num = x.shape[1]
#         for slice_index in range(slice_num):
#             feature_list_fake_0 = self.StyFg(y_[:,slice_index,:,:,:],training=True,scale=4)
#             feature_list_real_0 = self.StyFg( y[:,slice_index,:,:,:],training=True,scale=4)
#             feature_list_fake_1 = self.StyFg(x_[:,slice_index,:,:,:],training=True,scale=4)
#             feature_list_real_1 = self.StyFg( x[:,slice_index,:,:,:],training=True,scale=4)
#             tmp_l = len(feature_list_fake_0)
#             assert tmp_l==5
#             for index in range(1,tmp_l,1):
#                 style_loss_0 += (1/(tmp_l-1)/slice_num)*style_diff_2D(feature_list_real_0[index],feature_list_fake_0[index])
#                 style_loss_1 += (1/(tmp_l-1)/slice_num)*style_diff_2D(feature_list_real_1[index],feature_list_fake_1[index])
#         # tf.print(style_loss_0.shape)
#         # tf.print(style_loss_1)
#         return style_loss_0+style_loss_1
# #---------------------------------------------------------------------------------------------------------------------------------#
# def mae(x,y):
#     # tf.print("mae")
#     return tf.reduce_mean(tf.abs(x-y))
# def mae2(x,y): #但是为了计算速度 不采用此方法
#     b = x.shape[0]
#     M = 1
#     for i in range(1,len(x.shape),1):
#         M *= x.shape[i]
#     norm = tf.norm(tf.reshape(x-y, shape=[b,-1]),ord=1,axis=-1)/M
#     return tf.reduce_mean(norm)
# def mse(x,y):
#     return tf.reduce_mean(tf.math.square(x-y))

# def feature_difference(x,y,ords=1):#计算2D或3D情况下特征图x,y的差 
#     # 涉及运算效率，不进行输入检查
#     # tf.print("Vper",ords)
#     if ords==1:
#         return tf.reduce_mean(tf.abs(x-y)) # 在包括batch的维度取均值
#     elif ords==2:
#         return tf.reduce_mean(tf.square(x-y))# 在包括batch的维度取均值
#     else:
#         raise ValueError("Unsupported ords")
# def feature_difference_list(x,x_,index_begin=1,ords=1):#计算2D或3D情况下一系列特征图x,y的差 并保持权值总和1 index_begin 默认为1 跳过第0特征图
#     buf = 0.0
#     l = len(x)-index_begin
#     for index in range(index_begin,len(x),1):
#         buf += (1/l)*feature_difference(x[index],x_[index],ords=ords)
#     return buf
# def dual_feature_difference_list(x,x_,y,y_,index_begin,index_end,ords=1):#计算2D或3D情况下一系列特征图x,y的差 并保持权值总和1 index_begin 默认为1 跳过第0特征图
#     buf_0,buf_1 = 0.0,0.0
#     l = index_end-index_begin+1
#     for index in range(index_begin,index_end+1,1):
#         buf_0 += (1/l)*feature_difference(x[index],x_[index],ords=ords)
#         buf_1 += (1/l)*feature_difference(y[index],y_[index],ords=ords)
#     return buf_0,buf_1
# def grma_2D(x):
#     b,h,w,c = x.shape
#     m = tf.reshape(x,[b,-1,c])
#     m_T = tf.transpose(m,perm=[0,2,1])
#     g = (1.0/(h*w*c))*tf.matmul(m_T,m)
#     # tf.print(tf.reduce_mean(g))
#     return g # [B,C,C]
# def grma_3D(x):
#     b,d,h,w,c = x.shape
#     m = tf.reshape(x,[b,-1,c])
#     m_T = tf.transpose(m,perm=[0,2,1])
#     g = (1.0/(d*h*w*c))*tf.matmul(m_T,m)
#     return g # [B,C,C]
# def style_diff_2D(x,y):
#     style_diff = tf.reduce_mean(tf.square(tf.norm(grma_2D(x)-grma_2D(y),ord="fro",axis=[1,2]))) # 在batch 维度取均值
#     return style_diff
# def style_diff_3D(x,y):
#     style_diff = tf.reduce_mean(tf.square(tf.norm(grma_3D(x)-grma_3D(y),ord="fro",axis=[1,2]))) # 在batch 维度取均值
#     return style_diff
    