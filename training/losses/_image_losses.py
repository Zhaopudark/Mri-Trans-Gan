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
import logging
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
class MeanAbsoluteError(tf.keras.losses.MeanAbsoluteError):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)

class MeanSquaredError(tf.keras.losses.MeanSquaredError):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)

class LossCrossListWrapper(tf.keras.losses.Loss):
    """
    Wraps a loss function in the `Loss` class.
    TODO 彻底对loss 解耦 拆分成两个任务 在list上的loss 以及list中的每个的tensor loss
    """
    @typechecked
    def __init__(self,mean_over_batch:bool=False,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.loss_kwargs = {}
        self.loss_kwargs["mean_over_batch"] = mean_over_batch # TODO @property
        self.mean_over_batch = mean_over_batch
    def loss_func(self,x1,x2):
        NotImplementedError('Must be implemented in subclasses.')
    def call(self,y_true,y_pred):
        buf = []
        for x1,x2 in zip(y_true,y_pred):
            loss_func_output = self.loss_func(x1,x2)
            if self.mean_over_batch:
                buf.append(tf.reduce_mean(loss_func_output)) # TODO 
            else:
                buf.append(loss_func_output)
        return buf
    def get_config(self):
        config = {}
        if hasattr(self,"loss_kwargs"):
            config = {**self.loss_kwargs}     
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class MeanVolumeGradientError(LossCrossListWrapper):
    """
    MeanVolumeGradientError 
    calculate the volume gradient error between 2 tensors  
    give out a mean scaler 
    sample_weight can be used to set weights.

    Consider 
    y_true is a tensor in [B,D2,D3,D4,...,DN-1,C] or [B,C,D2,D3,D4,...,DN-1] shape
    y_pred is the same shape as y_true

    First
        define 'get_volumn_gradient' to calculate volumn gradient of each dim, except batch dimension and channel dimension.
        for example: consider y_true in [B,D2,D3,D4,...,DN-1,C] or [B,C,D2,D3,D4,...,DN-1] shape
            1. slice y_true to get y_true_1 on `D2` dimension, [::,0:-1:,::,...,::]
            2. slice y_true to get y_true_2 on `D2` dimension, [::,1::,::,...,::]
            3. y_true_2 minus y_true_1 got a tensor in shape [B,D2 - 1,D3,D4,...,D{N-1},C] or [B,C,D2 - 1,D3,D4,...,D{N-1}]
            4. pading the tensor to shape [B,D2,D3,D4,...,DN-1,C] or [B,C,D2,D3,D4,...,DN-1], as volumn_gradient_D2
            5. repeat above procedure on `D3`,`D4`,...,`DN-1` dimension
            6. gather all results to a list volumn_gradient=[volumn_gradient_D2,volumn_gradient_D3,...,volumn_gradient_D{N-1}]

        axis = [1,2,...,N-2] if y_true in [B,D2,D3,D4,...,DN-1,C]  or [2,3,...,N-1] if y_true in [B,C,D2,D3,D4,...,DN-1]
        loss_func(volumn_gradient_D2_ture,volumn_gradient_D2_pred)
            = tf.reduce_mean(tf.square(volumn_gradient_D2_ture-volumn_gradient_D2_pred),axis=axis) # [B,C]
        NOTE  here we give a choice to users whether use 1-norm or 2-norm
        temp_loss = [loss_func(volumn_gradient_D2_ture,volumn_gradient_D2_pred),
                     loss_func(volumn_gradient_D3_ture,volumn_gradient_D3_pred),
                     ...,
                     loss_func(volumn_gradient_D{N-1}_ture,volumn_gradient_D{N-1}_pred)]  # shape = [N-2,B,C]
    Second 
        Use sample_weight to give each temp_loss's volume a specific weight
        The same as MeanFeatureReconstructionError
        So in this class, since temp_loss has shape [N-2,B,C]
            sample_weight' right shape is:
                [], 
                [N-2,B],[N-2,1],[1,B],[1,1]
                [N-2,B,C],[N-2,1,C],[N-2,B,1],[1,B,C],[N-2,1,1],[1,B,1],[1,1,C],[1,1,1]
                [N-2,B,C,1],[N-2,1,C,1],[N-2,B,1,1],[1,B,C,1],[N-2,1,1,1],[1,B,1,1],[1,1,C,1],[1,1,1,1]
            sample_weight' wrong shape is
                [N-2],[B],[C] since their rank(1) differs by 2 not 1 from the new rank of temp_loss(3), and not suit temp_loss' shape [N-2,B,C] when matmul()
                others
        For general usage, we want to use a list of `N-2` elements to give weights for different dimension's volume gradients.
        Unfortunately, shape [N-2] is not suit exactly since keras.losses.Loss's broadcast mechanism.
        So, we should modify the behavior, if user give a sample_weight in rank 1, we add a dimension to it.
    Third: 
        Reduce weighted_loss by reduction.
        The same as MeanFeatureReconstructionError

    Deprecated:
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
    """
    @typechecked
    def __init__(self,mode:str="L1",name:str="mean_volume_gradient_error",data_format:str="channels_last",**kwargs) -> None:
        super().__init__(name=name,**kwargs)
        self.loss_kwargs["mode"] = mode
        self.loss_kwargs["data_format"] = data_format
        if mode.upper() == "L1":
            self.inner_loss = self.l1_loss
        elif mode.upper() == "L2":
            self.inner_loss = self.l2_loss
        else:
            raise ValueError("MeanVolumeGradientError's inner loss mode only support in 'L1' or 'L2', not{}".format(mode))
        self.data_format = data_format.lower()
        if self.data_format not in ["channels_last","channels_first"]:
            raise ValueError("data_format should be one in 'channels_last' or'channels_first', not {}.".format(data_format))
    def _get_volume_gradient(self,input):
        out_buf = []
        valid_indexs = self._get_reduce_or_norm_axis(input)
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
        return out_buf
    def _get_reduce_or_norm_axis(self,x):
        rank = len(x.shape)-2
        assert rank>=0
        if self.data_format == "channels_last":
            return list(range(len(x.shape)))[1:-1:]
        elif self.data_format == "channels_first":
            return list(range(len(x.shape)))[2::]
        else:
            raise ValueError("data_format should be one in 'channels_last' or'channels_first', not {}.".format(self.data_format))
    def l1_loss(self,x1,x2):
        _axis = self._get_reduce_or_norm_axis(x1)
        return tf.reduce_mean(tf.abs(x1-x2),axis=_axis)
    def l2_loss(self,x1,x2): 
        _axis = self._get_reduce_or_norm_axis(x1)
        return tf.reduce_mean(tf.square(x1-x2),axis=_axis)
    def loss_func(self,x1,x2):
        return self.inner_loss(x1,x2)
    def call(self,y_true,y_pred):
        y_true_gradient = self._get_volume_gradient(y_true)
        y_pred_gradient = self._get_volume_gradient(y_pred)      
        return super().call(y_true_gradient,y_pred_gradient)
    def __call__(self,y_true,y_pred,sample_weight=None):
        if sample_weight is not None:
            sample_weight = tf.convert_to_tensor(sample_weight)
            if len(sample_weight.shape)==1:
                sample_weight = tf.expand_dims(sample_weight,[-1])
        return super().__call__(y_true,y_pred,sample_weight)
class MeanFeatureReconstructionError(LossCrossListWrapper):
    """
    see: https://arxiv.org/pdf/1603.08155.pdf
    MeanFeatureReconstructionError 
    calculate the error between 2 lists of feature maps 
    give out a mean scaler 
    sample_weight can be used to set weights.

    More detailedly:
    Consider f1, f2, ..., fn are n feature maps, they are 
    not necessarily have same shape with each other.
    y_true = [f1_ture,f2_ture,...,fn_ture]
    y_pred = [f1_pred,f2_pred,...,fn_pred]

    Consider f1_ture,f1_pred have N dims, i.e., the shape [B,D2,D3,D4,...,DN].
    Since calculate the mean over batch dimension may leads to problems, here 
    we will maintain the batch dimension
    define a loss func:
        axis = [1,2,...,N-1]
        loss_func(f1_ture,f1_pred)=tf.reduce_mean(tf.square(f1_ture-f1_pred),axis=axis) # shape = [B]
        NOTE some papers use 1-norm when practice, so here we give a choice to users.
            such as https://ieeexplore.ieee.org/abstract/document/8653423
    First:
        temp_loss = [loss_func(f1_ture,f1_pred),loss_func(f2_ture,f2_pred),...,loss_func(fn_ture,fn_pred)]  # shape = [n,B]
    Second:
        Use sample_weight to give each temp_loss's volume a specific weight
        sample_weight shold be a list, tuple or tensor and have the shape that can be broadcast to temp_loss's shape,
            NOTE 1. In typical application, loss function's sample_weight acts as a coefficient for the loss, and losses usually
            conduct in 'axis=-1'. 
            2. Ususally, broadcast follows 
            https://numpy.org/doc/stable/user/basics.broadcasting.html
            It starts with the trailing (i.e. rightmost) dimensions and works its way left. Two dimensions are compatible when
                1. they are equal, or
                2. one of them is 1
            3. So, in loss computation, it is not appropriate to consider rightmost first. The broadcast method should be:
            Starting with the heading (i.e. leftmost) dimensions and works its way right. In the backend of keras.losses.Loss,
            the broadcast behavior of sample_weight follows:
                1. Squeezes or expands `last` dim of `sample_weight` if its rank differs by 1
                from the new rank of temp_loss.(`last` dim should be 1 if squeeze it)
                2. If `sample_weight` is scalar, it is kept scalar.
            So in this class, since  temp_loss has shape [n,B]
                sample_weight' right shape is:
                    [], 
                    [n], [1], 
                    [n,1], [1,B], [1,1], [n,B]
                    [n,1,1], [1,B,1], [1,1,1], [n,B,1]
                sample_weight' wrong shape is
                    [B] since it will expanded to [B,1] but not suit temp_loss' shape [n,B] when matmul()
                    others
            For general usage, we want to use a list of `n` elements to give weights for different feature maps.
            Luckly, with the help of keras.losses.Loss's sample_weight broadcast behavior, we can exactly use the 
            the `n` elements list to achieve our goal, without any tweaking, even though this mechanism 
            is designed for "BATCH" originally.
        if give sample_weight: 
            weighted_loss = temp_loss*sample_weight,axis=0 # [n,B]
        if no sample_weight:
            weighted_loss = temp_loss,axis=0 # [B]
            or consider sample_weight is '1' in each dim
            weighted_loss = temp_loss*tf.ones_like(temp_loss) # [n,B]
    Third: 
        Reduce weighted_loss by reduction. This procedure can not be totally controlled by user.
        NOTE In general practice, weighted_loss is not "[n,B]" in shape, but "[B,D2,D3,D4,...,DN-1]".
            Then, if reduction is None,
                    reduced_weighted_loss will in shape "[B,D2,D3,D4,...,DN-1]"
                if reduction is AUTO/SUM/SUM_OVER_BATCH_SIZE
                    reduced_weighted_loss will in shape "[]"
        So, if weighted_loss is  "[n,B]" in shape:
            if reduction is None,
                reduced_weighted_loss will in shape "[n,B]"
            if reduction is AUTO/SUM/SUM_OVER_BATCH_SIZE
                reduced_weighted_loss will in shape "[]"
        So, in this class, we finally cannot got ideal output with shape "[B]", if just inherit from tf.keras.losses.Loss.
        TODO  Make this class maintain batch dimension and leave the decision of reduction to users. This problem does not influence traditional usage, since we usually
        consider "Feature Reconstruction Loss" when batch size set to 1
    a target feature map X of shape [H,W,C]
    a current feature map X_ of shape [H,W,C]
    
    """
    @typechecked
    def __init__(self,mode:str="L1",name:str="mean_feat_reco_error",**kwargs) -> None:
        super().__init__(name=name,**kwargs)
        self.loss_kwargs["mode"] = mode
        if mode.upper() == "L1":
            self.inner_loss = self.l1_loss
        elif mode.upper() == "L2":
            self.inner_loss = self.l2_loss
        else:
            raise ValueError("MeanFeatureReconstructionError's inner loss mode only support in 'L1' or 'L2', not{}".format(mode))
    def _get_reduce_or_norm_axis(self,x):
        return list(range(len(x.shape)))[1::]
    def l1_loss(self,x1,x2): # [B,x,x,x,x]
        _axis = self._get_reduce_or_norm_axis(x1)
        return tf.reduce_mean(tf.abs(x1-x2),axis=_axis)
    def l2_loss(self,x1,x2): # [B,x,x,x,x]
        _axis = self._get_reduce_or_norm_axis(x1)
        return tf.reduce_mean(tf.square(x1-x2),axis=_axis)
    def loss_func(self,x1,x2):
        return self.inner_loss(x1,x2)

class MeanStyleReconstructionError(LossCrossListWrapper):
    """
    see: https://arxiv.org/pdf/1603.08155.pdf
    MeanStyleReconstructionError 
    calculate the style error between 2 lists of feature maps 
    give out a mean scaler 
    sample_weight can be used to set weights

    More detailedly:
    Consider f1, f2, ..., fn are n feature maps, they are 
    not necessarily have same shape with each other.
    y_true = [f1_ture,f2_ture,...,fn_ture]
    y_pred = [f1_pred,f2_pred,...,fn_pred]

    Consider f1_ture,f1_pred have N dims, i.e., the shape 
    [B,D2,D3,D4,...,DN-1,C]  # data_format = channels_last 
    or [B,C,D2,D3,D4,...,DN-1]  # data_format = channels_first
    
    Since calculate the mean over batch dimension may leads to problems, here 
    we will maintain the batch dimension.
    First
        define 'get_gram_matrix' to calculate Gram Matrix of each feature, representing 'style'
        for example: consider f1_ture in [B,D2,D3,D4,...,DN-1,C] or [B,C,D2,D3,D4,...,DN-1] shape
            1. reshape f1_ture to [B,-1,C] or [B,C,-1]
            2. matmul f1_ture and f1_ture.T in last 2 dimension got shape [B,C,C]
            3. divive the matmuled result by total volume nums over all dims, except batch dim
        loss_func(f1_ture,f1_pred)=
            tf.suqare(
                tf.norm(
                    get_gram_matrix(f1_ture)-get_gram_matrix(f1_pred),
                    ord="fro",
                    axis=[-2,-1])) # [B]
        NOTE if we consider f1_ture a 'ones' tensor the shape [B,D2,D3,D4,...,DN-1,C] or [B,C,D2,D3,D4,...,DN-1]
        Then, get_gram_matrix(f1_ture) will give out a tensor in [B,C,C], each element is 1/C 
        and then, tf.suqare(tf.norm(·)) will give out a tensor in [B],each element is (1/c**2+1/C**2+...+1/C**2) == 1
        So the mean `energy` of each point isn't changed after style difference calculation, i.e., mean `energy` is not affected by the number of volumes
        temp_loss = [loss_func(f1_ture,f1_pred),loss_func(f2_ture,f2_pred),...,loss_func(fn_ture,fn_pred)]  # shape = [n,B] 
    Second 
        Use sample_weight to give each temp_loss's volume a specific weight
        The same as MeanFeatureReconstructionError
        So in this class, since  temp_loss has shape [n,B]
            sample_weight' right shape is:
                [], 
                [n], [1], 
                [n,1], [1,B], [1,1], [n,B]
                [n,1,1], [1,B,1], [1,1,1], [n,B,1]
            sample_weight' wrong shape is
                [B] since it will expanded to [B,1] but not suit temp_loss' shape [n,B] when matmul()
                others
        For general usage, we want to use a list of `n` elements to give weights for different feature maps (gram matrices).
        Luckly, with the help of keras.losses.Loss's sample_weight broadcast behavior, we can exactly use the 
        the `n` elements list to achieve our goal, without any tweaking, even though this mechanism 
        is designed for "BATCH" originally.
    Third: 
        Reduce weighted_loss by reduction.
        The same as MeanFeatureReconstructionError
    """
    @typechecked
    def __init__(self,name:str="mean_style_reco_error",data_format:str="channels_last",**kwargs) -> None:
        super().__init__(name=name,**kwargs)
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
    def loss_func(self,x1,x2):
        x1 = self._get_gram_matrix(x1) # [B,C,C]
        x2 = self._get_gram_matrix(x2) # [B,C,C]
        return tf.square(tf.norm(x1-x2,ord="fro",axis=[-2,-1])) # [B]

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)  



    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    loss = MeanStyleReconstructionError()
    feature_1 = tf.random.normal(shape=[2,8,8,3])
    feature_2 = tf.random.normal(shape=[2,8,8,4])
    feature_3 = tf.random.normal(shape=[2,8,8,5])
    feature_4 = tf.random.normal(shape=[2,8,8,3])
    feature_5 = tf.random.normal(shape=[2,8,8,4])
    feature_6 = tf.random.normal(shape=[2,8,8,5])
    y_true = [feature_1,feature_2,feature_3]
    y_pred = [feature_4,feature_5,feature_6]
    y = loss(y_true,y_pred)
    print(y)

    y_true = [feature_1,feature_2,feature_3,feature_1,feature_2,feature_3]
    y_pred = [feature_4,feature_5,feature_6,feature_4,feature_5,feature_6]
    y = loss(y_true,y_pred)
    print(y)

    # feature_1 = tf.stack([feature_1,feature_1],axis=1)
    # feature_2 = tf.stack([feature_2,feature_2],axis=1)
    # feature_3 = tf.stack([feature_3,feature_3],axis=1)
    # feature_4 = tf.stack([feature_4,feature_4],axis=1)
    # feature_5 = tf.stack([feature_5,feature_5],axis=1)
    # feature_6 = tf.stack([feature_6,feature_6],axis=1)
    # y_true = [feature_1,feature_2,feature_3]
    # y_pred = [feature_4,feature_5,feature_6]
    # y = loss(y_true,y_pred)
    # print(y)

    # feature_1 = tf.stack([feature_1,feature_1],axis=1)
    # feature_2 = tf.stack([feature_2,feature_2],axis=1)
    # feature_3 = tf.stack([feature_3,feature_3],axis=1)
    # feature_4 = tf.stack([feature_4,feature_4],axis=1)
    # feature_5 = tf.stack([feature_5,feature_5],axis=1)
    # feature_6 = tf.stack([feature_6,feature_6],axis=1)
    # # feature_1 = tf.transpose(feature_1,perm=[1,0,2,3,4,5])
    # # feature_2 = tf.transpose(feature_2,perm=[1,0,2,3,4,5])
    # # feature_3 = tf.transpose(feature_3,perm=[1,0,2,3,4,5])
    # # feature_4 = tf.transpose(feature_4,perm=[1,0,2,3,4,5])
    # # feature_5 = tf.transpose(feature_5,perm=[1,0,2,3,4,5])
    # # feature_6 = tf.transpose(feature_6,perm=[1,0,2,3,4,5])
    # y_true = [feature_1,feature_2,feature_3]
    # y_pred = [feature_4,feature_5,feature_6]
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
    