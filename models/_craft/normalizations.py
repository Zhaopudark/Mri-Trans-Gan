import tensorflow as tf
import numpy as np
__all__ = [
    "BatchNormalization",
    "LayerNormalization",
    "InstanceNormalization",
    "SpectralNormalization",
]
def _norm_wrapper_mixed(call):#对norm的操作不作混合精度 
    def call_mixed(x,training):
        x = tf.cast(x,tf.float32)
        y = call(x,training)
        return tf.cast(y,tf.float16)
    return call_mixed
#------------------------------------------------------------------------------------------------------------------------------#
class BatchNormalization(tf.keras.layers.Layer):
    """BatchNormalization
    B-X-C
    B-H-W-C
    B-D-H-W-C
    在不同的C维度 对B-(...)求norm 最小要求维度为2
    moments后维度[1,...,C]
    变量shape与其一致
    """
    def __init__(self,epsilon=1e-3,momentum=0.99,name=None,dtype=None):
        super(BatchNormalization,self).__init__(name=name,dtype=None)#对norm的操作不作混合精度 
        self.epsilon = epsilon
        self.momentum = momentum
        if (dtype is not None)and(dtype.name=="mixed_float16"):
            self.call = _norm_wrapper_mixed(self.call)
    def build(self,input_shape):
        super(BatchNormalization,self).build(input_shape)
        if len(input_shape)>=2:
            min_axis_index = 0
            max_axis_index = len(input_shape)-1-1
            self.consider_axis = [x for x in range(min_axis_index,max_axis_index+1)] #e.g. axis = [0,1,2]
            weight_shape = [1 for x in range(min_axis_index,max_axis_index+1)] 
            if input_shape[0] is not None:
                batch_size = int(input_shape[0])
            else:
                batch_size = 1
            channels = int(input_shape[-1])
            weight_shape = weight_shape+[channels] #e.g. shape = [1,1,1,C]
        else:
            raise ValueError("Tensor must be more than 2 dimensions for BatchNormalization.")
        
        self.offset = self.add_weight('offset',shape=weight_shape,initializer=tf.initializers.Zeros(),trainable=True)
        self.scale = self.add_weight('scale',shape=weight_shape,initializer=tf.initializers.Ones(),trainable=True)

        self.mean = self.add_weight('mean',shape=weight_shape,initializer=tf.initializers.Zeros(),trainable=False)
        self.variance = self.add_weight('variance',shape=weight_shape,initializer=tf.initializers.Ones(),trainable=False)#参与测试不参与训练
        output_shape = input_shape[:]
        return output_shape 

    def call(self,x,training=True):
        if training:
            mean,variance = tf.nn.moments(x,axes=self.consider_axis,keepdims=True)
            tmp1 = self.mean*self.momentum+(1-self.momentum)*mean
            tmp2 = self.variance*self.momentum+(1-self.momentum)*variance
            self.mean.assign(tmp1)
            self.variance.assign(tmp2)
            y = tf.nn.batch_normalization(x,mean,variance,self.offset,self.scale,self.epsilon)
        else:
            y = tf.nn.batch_normalization(x,self.mean,self.variance,self.offset,self.scale,self.epsilon)
        return y
#------------------------------------------------------------------------------------------------------------------------------#
class LayerNormalization(tf.keras.layers.Layer):
    """LayerNormalization 
    B-X-C
    B-H-W-C
    B-D-H-W-C
    对-(...)-C 求norm 最小要求维度为2
    但是应当要求B维度为1
    moments后维度[B,...,1]
    而gamma beta 与求norm维度并不一致 维度在通道维度C独立 即[B=1,1,C]or[B=1,1,1,C]or[B=1,1,1,1,C]
    """
    def __init__(self,epsilon=1e-3,name=None,dtype=None):
        super(LayerNormalization,self).__init__(name=name,dtype=None)#对norm的操作不作混合精度 
        self.epsilon = epsilon
        if (dtype is not None)and(dtype.name=="mixed_float16"):
            self.call = _norm_wrapper_mixed(self.call)
    
    def build(self,input_shape):
        super(LayerNormalization,self).build(input_shape)
        if len(input_shape)>=2:
            min_axis_index = 0+1
            max_axis_index = len(input_shape)-1
            self.consider_axis = [x for x in range(min_axis_index,max_axis_index+1)]  #e.g. [B,X,C] axis = [1,2]
            weight_shape = [1 for x in range(min_axis_index,max_axis_index+1)] 
            if input_shape[0] is not None:
                batch_size = int(input_shape[0])
            else:
                batch_size = 1
            weight_shape = [1]+weight_shape  
            channels = int(input_shape[-1])
            weight_shape[-1] = channels #e.g. [B,X,C] weight_shape=[1,1,C]
        else:
            raise ValueError("Tensor must be more than 2 dimensions for LayerNormalization.")
        # if batch_size!=1:
        #     raise ValueError("LayerNormalization needs batch size fixed to 1.")
        self.offset = self.add_weight('offset',shape=weight_shape,initializer=tf.initializers.Zeros(),trainable=True)
        self.scale = self.add_weight('scale',shape=weight_shape,initializer=tf.initializers.Ones(),trainable=True)
        output_shape = input_shape[:]
        return output_shape

    def call(self,x,training):
        mean,variance = tf.nn.moments(x,axes=self.consider_axis,keepdims=True)
        y = tf.nn.batch_normalization(x,mean,variance,self.offset,self.scale,self.epsilon)
        return y
#------------------------------------------------------------------------------------------------------------------------------#   
class InstanceNormalization(tf.keras.layers.Layer):
    """InstanceNormalization
    B-X-C
    B-HW-C
    B-DHW-C
    对-()- 求norm 仅保留B，C维度 最小要求维度为3
    但是应当要求B维度为1
    moments后维度[B,...,C]
    而gamma beta 与求norm维度并不一致 在通道维度C独立 即[B=1,1,C]or[B=1,1,1,C]or[B=1,1,1,1,C]
    """
    def __init__(self,epsilon=1e-3,name=None,dtype=None):
        super(InstanceNormalization,self).__init__(name=name,dtype=None)
        self.epsilon = epsilon
        if (dtype is not None)and(dtype.name=="mixed_float16"):
            self.call = _norm_wrapper_mixed(self.call)
    
    def build(self,input_shape):
        super(InstanceNormalization,self).build(input_shape)
        if len(input_shape)>=3:
            min_axis_index = 0+1
            max_axis_index = len(input_shape)-1-1
            self.consider_axis = [x for x in range(min_axis_index,max_axis_index+1)] #e.g. [B,X,C] axis = [1]
            weight_shape = [1 for x in range(min_axis_index,max_axis_index+1)] 
            if input_shape[0] is not None:
                batch_size = int(input_shape[0])
            else:
                batch_size = 1
            channels = int(input_shape[-1])
            weight_shape = [1]+weight_shape+[channels]   #e.g. [B,X,C] weight_shape=[1,1,C]
        else:
            raise ValueError("Tensor must be more than 3 dimensions for InstanceNormalization.")
        if batch_size!=1:
            raise ValueError("Instance Normalization needs batch size fixed to 1.")
        self.offset = self.add_weight('offset',shape=weight_shape,initializer=tf.initializers.Zeros(),trainable=True)
        self.scale = self.add_weight('scale',shape=weight_shape,initializer=tf.initializers.Ones(),trainable=True)
        output_shape = input_shape[:]
        return output_shape
    def call(self,x,training):
        mean,variance = tf.nn.moments(x,axes=self.consider_axis,keepdims=True)
        y = tf.nn.batch_normalization(x,mean,variance,self.offset,self.scale,self.epsilon)
        return y
#------------------------------------------------------------------------------------------------------------------------------#
class SpectralNormalization(tf.keras.layers.Layer):
    """SpectralNormalization
    为谱范数正则化定义若干超参数用以研究
    分别是迭代次数 
    iter_k
    是否clip
    """
    def __init__(self,iter_k=1,clip_flag=True,clip_range=100.0,name=None,dtype=None):
        super(SpectralNormalization,self).__init__(name=name,dtype=None)#对norm的操作不作混合精度 
        self.iter_k = iter_k
        self.clip_flag = clip_flag
        self.clip_range = clip_range
        
        if self.clip_flag:
            self.call = self.with_clip
        else:
            self.call = self.no_clip
        print(self.iter_k)
        print(self.clip_flag)
        print(self.clip_range)
    def build(self,input_shape,w):
        super(SpectralNormalization,self).build(input_shape)
        self.w = w
        self.w_shape = input_shape
        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name="sn_u")
    def call(self,x,training):
        print("call 必须被重载！！！！")
    def with_clip(self,x,training):
        if training:
            w = tf.reshape(x,[-1,self.w_shape[-1]])
            u = self.u
            with tf.name_scope("spectral_normalize"):
                for _ in range(self.iter_k):
                    v = tf.math.l2_normalize(tf.matmul(u,w,transpose_b=True)) # u@W.T 得到v
                    u = tf.math.l2_normalize(tf.matmul(v,w)) # v@W 得到u
                sigma = tf.matmul(tf.matmul(v,w),u,transpose_b=True)
                self.w.assign(tf.clip_by_value(tf.math.divide_no_nan(self.w,sigma),clip_value_min=-self.clip_range,clip_value_max=self.clip_range))
                self.u.assign(u)
        else:
            pass
    def no_clip(self,x,training):
        if training:
            w = tf.reshape(x,[-1,self.w_shape[-1]])
            u = self.u
            with tf.name_scope("spectral_normalize"):
                for _ in range(self.iter_k):
                    v = tf.math.l2_normalize(tf.matmul(u,w,transpose_b=True)) # u@W.T 得到v
                    u = tf.math.l2_normalize(tf.matmul(v,w)) # v@W 得到u
                sigma = tf.matmul(tf.matmul(v,w),u,transpose_b=True)
                self.w.assign(tf.math.divide_no_nan(self.w,sigma))
                self.u.assign(u)
        else:
            pass
    
# class SpectralNormalization(tf.keras.layers.Layer):
#     """SpectralNormalization
#     定义W 不变
#     W.T@x x 为u 
#     W@x  x 为v
#     同笔记 以u_0 开始迭代
#             v_i = norm(W.T@u_i-1)
#             u_i = norm(W@v_i)
#         v_i+1 = norm(W.T@u_i)
#         sigma = v_i+1.T@W.T@u_i 
#         假设v_i+1笔记v_i 
#     算法变为:
#             v_i = norm(W.T@u_i-1)
#             u_i = norm(W@v_i)
#         sigma = v_i.T@W.T@u_i 
#     以 v_0 开始迭代 算法变为
#             u_i = norm(W@v_i-1)
#             v_i = norm(W.T@u_i)
#         sigma = u_i.T@W@v_i 
#     --------------------------------
#     tfa中 
#     W.T@y y 为v
#     W@y y 为u 
#     且以新u_0开始迭代(即遵从原先v_0迭代的算法)
#     算法变为:
#             v_i = norm(W@u_i-1)
#             u_i = norm(W.T@v_i)
#         sigma = v_i.T@W@u_i 
#     转换为行向量的形式且保持W不变:
#             v_i = norm(u_i-1@W.T)
#             u_i = norm(v_i@W)
#         sigma = v_i@W@u_i.T = (v_i@W)@u_i.T 算法证毕
#     """
#     def __init__(self,iter_k=1,name=None,dtype=None,**kwargs):
#         super(SpectralNormalization,self).__init__(name=name,dtype=None)#对norm的操作不作混合精度 
#         self.iter_k = iter_k
#     def build(self,input_shape,w):
#         super(SpectralNormalization,self).build(input_shape)
#         self.w = w
#         self.w_shape = input_shape
#         self.u = self.add_weight(shape=(1, self.w_shape[-1]),
#                                  initializer=tf.initializers.TruncatedNormal(stddev=0.02),
#                                  trainable=False,
#                                  name="sn_u")
#     def call(self,x,training):
#         if training:
#             w = tf.reshape(x,[-1,self.w_shape[-1]])
#             u = self.u
#             with tf.name_scope("spectral_normalize"):
#                 for _ in range(self.iter_k):
#                     v = tf.math.l2_normalize(tf.matmul(u,w,transpose_b=True)) # u@W.T 得到v
#                     u = tf.math.l2_normalize(tf.matmul(v,w)) # v@W 得到u
#                 sigma = tf.matmul(tf.matmul(v,w),u,transpose_b=True)
#                 # self.w.assign(tf.math.divide_no_nan(self.w,sigma))
#                 self.w.assign(tf.clip_by_value(tf.math.divide_no_nan(self.w,sigma),clip_value_min=-100.0,clip_value_max=100.0))
#                 self.u.assign(u)
#         else:
#             pass
    # def call(self,x,training):
    #     if training:
    #         w = tf.reshape(x, [-1, self.w_shape[-1]])
    #         u = self.u
    #         with tf.name_scope("spectral_normalize"):
    #             for _ in range(self.iter_k):
    #                 v = tf.math.l2_normalize(tf.matmul(u,w,transpose_b=True))  #u@W.T 得到v
    #                 u = tf.math.l2_normalize(tf.matmul(v,w)) #v@W 得到u
    #             sigma = tf.matmul(tf.matmul(v,w),u,transpose_b=True)
    #             self.w.assign(self.w/sigma)
    #             self.u.assign(u)
    #     else:
    #         pass
#------------------------------------------------------------------------------------------------------------------------------#
if __name__=="__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # BN = tf.keras.layers.BatchNormalization(fused=False)
    # BN.build(input_shape=[None,34,36,73])
    # BN2 = BatchNormalization()
    # BN2.build(input_shape=[None,34,36,73])
    # print(BN.gamma.shape)
    # print(BN.beta.shape)
    # print(BN2.scale.shape)
    # print(BN2.offset.shape)
    # x = tf.random.normal(shape=[128,34,36,73])
    # for _ in range(100):
    #     y = tf.reduce_mean(BN(x,training=True)-BN2(x,training=True))
    #     print(y)

    # LN = tf.keras.layers.LayerNormalization(axis=[1,2,3])
    # LN.build(input_shape=[128, 20, 30, 40])
    # LN2 = LayerNormalization()
    # LN2.build(input_shape=[128, 20, 30, 40])
    # print(LN.gamma.shape)
    # print(LN.beta.shape)
    # print(LN2.scale.shape)
    # print(LN2.offset.shape)
    # x = tf.random.normal(shape=[128, 20, 30, 40])
    # for _ in range(100):
    #     y = tf.reduce_mean(LN(x,training=True)-LN2(x,training=True))
    #     print(y)
    a = tf.constant(1.0,tf.float32)
    for _ in range(70):
        a = a/(1e-5+1e-12)
        print(a.numpy())

    # import tensorflow_addons as tfa
    # IN2 = tfa.layers.InstanceNormalization(axis=-1)
    # x = tf.random.normal(shape=[5, 20, 30, 40])
    # IN2.build(input_shape=[5, 20, 30, 40])
    # print(IN2.trainable_variables)
    # print(y:=IN2(x))


    # input_shape = tf.keras.backend.int_shape(x)
    # tensor_input_shape = tf.shape(x)
    # print(input_shape,tensor_input_shape)

    # group_shape = tf.keras.backend.int_shape(x)
    # group_reduction_axes = list(range(1, len(group_shape)))
    # # print(group_reduction_axes)
    # is_instance_norm = True
    # axis = -2
    # if not is_instance_norm:
    #     axis = -2 if axis == -1 else axis - 1
    # else:
    #     axis = -1 if axis == -1 else axis - 1
    # group_reduction_axes.pop(axis)
    # print(group_reduction_axes)
    # print(IN2.gamma.shape)
    # print(IN2.beta.shape)

    # LN2 = LayerNormalization()
    # LN2.build(input_shape=[5, 20, 30, 128])
    # print(LN2.scale.shape)
    # print(LN2.offset.shape)

    # GN2 = tfa.layers.GroupNormalization(groups=1,axis=2)
    # GN2.build(input_shape=[5, 20, 30, 128])
    # print(GN2.gamma.shape)
    # print(GN2.beta.shape)

    # IN2 = tfa.layers.GroupNormalization(groups=128,axis=2)
    # IN2.build(input_shape=[5, 20, 128, 32])
    # print(IN2.gamma.shape)
    # print(IN2.beta.shape)
    # y=normalization_axis_cal(input_shape=[None,128,128,128,3],axis=[-1])
    # print(y)
    # print(IN2.axis)

