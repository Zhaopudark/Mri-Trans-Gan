import tensorflow as tf
import numpy as np
from functools import wraps
import logging
__all__ = [
    "BatchNormalization",
    "LayerNormalization",
    "InstanceNormalization",
    "SpectralNormalization",
]
def _norm_wrapper_mixed(cls):
    """
    为了数值稳定性 忽略用户的混合精度计算要求 对norm的计算不做混合精度 并配合前后级的精度模式(float16->float32(norm计算)->float16)
    默认为正常计算
    因此一旦指定了dtype 就需要进行patch
    NOTE 不考虑 float64
    NOTE 通过添加类变量 cls._mixed_precision_fused对__init__()和call()进行装饰
    对__init__()的装饰在实例化之前 以捕添加获用户指定的计算精度的功能 因此是对类装饰 修改类的初始化方法 _init_wrapper(cls.__init__) 必须返回class
    对call()的装饰在实例化之后 依据捕获的计算精度考虑是否进行装饰 因此是对实例方法的装饰  只能放在_init_wrapper中而不可以放在wrappered_cls中
    """
    def _call_wrapper(call):
        @wraps(call)
        def call_mixed(_input,*args,**kwargs):
            _input = tf.cast(_input,tf.float32)
            _output = call(_input,*args,**kwargs)
            return tf.cast(_output,tf.float16)
        return call_mixed
    def _init_wrapper(init):
        @wraps(init)
        def wrappered_init(self,*args,**kwargs):
            if "dtype" in kwargs.keys(): 
                if hasattr(kwargs["dtype"],"name"): # isinstance(kwargs["dtype"],tf.keras.mixed_precision.Policy)
                    if kwargs["dtype"].name=="mixed_float16":
                        self._mixed_precision_fused = True  
                        kwargs["dtype"] = None
                    else:
                        pass
                elif kwargs["dtype"] is None: 
                    pass 
                else: # NOTE 附带一个免费的类型检查
                    raise TypeError("dtype must be 'None' or a 'Policy'")
            else:
                pass 
            init(self,*args,**kwargs)
            if self._mixed_precision_fused:
                self.call = _call_wrapper(self.call)
        return wrappered_init
    class WrapperedCls(cls): 
        _mixed_precision_fused = False
        @_init_wrapper
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
    return WrapperedCls


# @_norm_wrapper_mixed
class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    mimic tf.keras.layers.BatchNormalization
    patch the call func to make sure "norm" is not calculated in float16 
    since in tf.keras.layers.BatchNormalization:
        ...
        mean, variance = self._moments(tf.cast(inputs, self._param_dtype),reduction_axes,keep_dims=keep_dims)
        ...
    where self._param_dtype is float32 in deed 
    so dtype is not work if set to float16(mixed_float16)

    input:...see the reference
    output:...see the reference
    reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
    """
    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)
#------------------------------------------------------------------------------------------------------------------------------#
# @_norm_wrapper_mixed 
class LayerNormalization(tf.keras.layers.LayerNormalization):
    """
    mimic tf.keras.layers.LayerNormalization 
    patch the call func to make sure "norm" is not calculated in float16 
    since in tf.keras.layers.LayerNormalization :
        ...
        input_dtype = inputs.dtype
        if input_dtype in ('float16', 'bfloat16') and self.dtype == 'float32':
            # If mixed precision is used, cast inputs to float32 so that this is at
            # least as numerically stable as the fused version.
            inputs = tf.cast(inputs, 'float32')

        # Calculate the moments on the last axis (layer activations).
        mean, variance = tf.nn.moments(inputs, self.axis, keepdims=True)
        ...
    where mean and variance are calculated in float32 indeed 
    so dtype is not work if set to float16(mixed_float16)

    input:...see the reference
    output:...see the reference
    reference: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
#########################################################
@_norm_wrapper_mixed
class GroupNormalization(tf.keras.layers.Layer):
    """
    Copy from the tfa.layers.GroupNormalization 
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization
    remove input type check
    patch the call func to make sure "norm" is not calculated in float16
    """
    def __init__(
        self,
        groups: int = 32,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()
    def build(self, input_shape):
        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)
        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)
    def call(self, inputs):
        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)
        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )
        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs
        return outputs
    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}
    def compute_output_shape(self, input_shape):
        return input_shape
    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            return inputs, group_shape
    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)
        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )
        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs
    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta
    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )
    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]
        if self.groups == -1:
            self.groups = dim
    def _check_size_of_dimensions(self, input_shape):
        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )
        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )
    def _check_axis(self):
        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead"
            )
    def _create_input_spec(self, input_shape):
        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )
    def _add_gamma_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None
    def _add_beta_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None
    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape
#------------------------------------------------------------------------------------------------------------------------------#   
class InstanceNormalization(GroupNormalization):
    """
    Copy from the tfa.layers.InstanceNormalization
    https://tensorflow.google.cn/addons/api_docs/python/tfa/layers/InstanceNormalization
    patch the call func to make sure "norm" is not calculated in float16
    if GroupNormalization's call func has been patched by @_norm_wrapper_mixed, there is
    no need to patch again in InstanceNormalization
    """
    def __init__(self,*args,**kwargs):
        if "groups" in kwargs:
            logging.warning("The given value for groups will be overwritten.")
        kwargs["groups"] = -1
        super().__init__(*args,**kwargs)   
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
class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )
    def call(self, inputs):
        print("????")
        print(inputs)
        tf.print("????")
        tf.print(inputs)
        return tf.matmul(inputs, self.w) + self.b

if __name__=="__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # c = Linear(10)
    # c.build(input_shape=[None,3])
    # s = c.compute_output_shape(input_shape=[None,3])
    # print(s)
    # s = c.compute_output_signature(input_shape=[127,3])
    # print(s)
    # T = TypeVar('T', int, float, complex)
    # Vector = Iterable[Tuple[T, T]]
    # print(Vector)

    # from typeguard import typechecked
    # from typing import TypeVar, Generic, Sized, Iterable, Container, Tuple,Dict
    # T = TypeVar('T', int, float,complex)
    # S = TypeVar('S', str,bytes)
    # Vector_0 = Dict[S,Dict[S,Tuple[T,T]]]
    # print(Vector_0)
    # @typechecked
    # def cat_a(a:int,b:Vector_0):
    #     return a
    # d = {}
    # d["1"] = {"2":(1.0,2)}
    # print(d)
    # print(cat_a(1,d))

    policy1 = tf.keras.mixed_precision.Policy('mixed_float16')
    policy2 = None
    policy_list = [policy1,policy2]
    for policy in policy_list:
        BN = tf.keras.layers.BatchNormalization(dtype=policy,fused=False)
        BN.build(input_shape=[None,34,36])
        BN2 = BatchNormalization(dtype=policy)
        BN2.build(input_shape=[None,34,36])
        print(BN.compute_dtype)
        print(BN._param_dtype)
        print(BN.gamma.shape,BN.gamma.dtype)
        print(BN.beta.shape,BN.beta.dtype)
        print(BN2.compute_dtype)
        print(BN2.gamma.shape)
        print(BN2.beta.shape)
        for _ in range(10):
            if isinstance(policy,tf.keras.mixed_precision.Policy):
                x = tf.random.normal(shape=[128,34,36],dtype=tf.float16)
            else:
                x = tf.random.normal(shape=[128,34,36],dtype=tf.float32)
            y = tf.reduce_mean(BN(x,training=True)-BN2(x,training=True))
            print(y)

    for policy in policy_list:    
        LN = tf.keras.layers.LayerNormalization(axis=[1,2,3],dtype=policy)
        LN.build(input_shape=[12, 20, 30, 40])
        LN2 = LayerNormalization(axis=[1,2,3],dtype=policy)
        LN2.build(input_shape=[12, 20, 30, 40])
        print(LN.gamma.shape)
        print(LN.beta.shape)
        print(LN2.gamma.shape)
        print(LN2.beta.shape)
        for _ in range(10):
            if isinstance(policy,tf.keras.mixed_precision.Policy):
                x = tf.random.normal(shape=[12, 20, 30, 40],dtype=tf.float16)
            else:
                x = tf.random.normal(shape=[12, 20, 30, 40],dtype=tf.float32)
            y = tf.reduce_mean(LN(x,training=True)-LN2(x,training=True))
            print(y)
    import tensorflow_addons as tfa
    for policy in policy_list:
        GN1 = tfa.layers.GroupNormalization(axis=-1,dtype=policy)
        GN1.build(input_shape=[5, 20, 30, 128])
        GN2 = GroupNormalization(axis=-1,dtype=policy)
        GN2.build(input_shape=[5, 20, 30, 128])
        for _ in range(10):
            if isinstance(policy,tf.keras.mixed_precision.Policy):
                x = tf.random.normal(shape=[5, 20, 30, 128],dtype=tf.float16)
            else:
                x = tf.random.normal(shape=[5, 20, 30, 128],dtype=tf.float32)
            y = tf.reduce_mean(GN1(x,training=True)-GN2(x,training=True))
            print(y)
    


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

