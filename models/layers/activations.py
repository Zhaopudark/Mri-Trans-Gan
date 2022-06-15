import logging
import tensorflow as tf 

__all__ = [
    'activation_slect',
    # 'Dropout',
    'Activation',
]
def activation_slect(activation_name):# 统一用tf.keras.activations内实现
    if activation_name == 'relu':
        return tf.keras.activations.relu
    elif activation_name == 'leaky_relu':
        logging.getLogger(__name__).warning("You have used leaky_relu! Remember give specific 'alpha' parameter to it!Default threshold is 0.")
        return tf.keras.activations.relu
    elif activation_name == 'gelu':
        return tf.keras.activations.gelu
    elif activation_name == 'sigmoid':
        return tf.keras.activations.sigmoid
    elif activation_name == 'tanh':
        return tf.keras.activations.tanh
    elif activation_name == 'linear':
        return tf.keras.activations.linear
    elif activation_name == 'softmax':
        return tf.keras.activations.softmax    
    elif activation_name is None:
        return tf.keras.activations.linear
    else:
        raise ValueError(f"Unsupported activation: {activation_name}")
class Dropout(tf.keras.layers.Layer):
    """
    Dropout 本质  
    均衡看待输入x的每个点 每个点能量视为1
    
    以一个random seed 
    生成noise_shape的 0 1 矩阵 mask 
    该mask的元素均值(期望)为(1-rate) 
    然后
    将0 1 矩阵 mask broadcast到输入的tensor shape
    与输入tensor进行哈达玛积操作
    输入tensor的每个点 遇到0 视为丢弃 遇到1 视为保留 即舍弃rate*100%的点
    然后 对于保留的点 乘以1.0/(1 - rate) 使得最终的点总和(总能量)不变

    E(x)===E(x*(1-rate)*1/(1-rate))

    """
    def __init__(self,
                 rate,
                 noise_shape=None,
                 seed=None,
                 name=None,
                 dtype=None):
        super(Dropout,self).__init__(name=name,dtype=dtype)
        self.tmp_kwargs ={}
        self.tmp_kwargs['rate'] = rate
        self.tmp_kwargs['noise_shape'] = noise_shape 
        self.tmp_kwargs['seed'] = seed
    def build(self,input_shape):
        super(Dropout,self).build(input_shape)
        return input_shape[:]
    def call(self,x,training):
        if not training:
            self.tmp_kwargs['rate'] = 0
        return tf.nn.dropout(x,**self.tmp_kwargs)

class Activation(tf.keras.layers.Layer):
    def __init__(self,activation_name,name=None,dtype=None,**kwargs):
        super(Activation,self).__init__(name=name,dtype=dtype)
        if activation_name != None:
            self.activation_name = activation_name.lower()
        else:
            self.activation_name = None
        def _activation_wrapper(activation_name,**kwargs):
            _special_flag = False
            if  activation_name in ['relu','leaky_relu']:
                alpha = 0.0
                max_value = None
                threshold = 0
                if 'alpha' in kwargs.keys():
                        alpha = kwargs['alpha']
                if 'max_value' in kwargs.keys():
                        max_value = kwargs['max_value']
                if 'threshold' in kwargs.keys():
                        threshold = kwargs['threshold']
                tmp_kwargs ={}
                tmp_kwargs['alpha'] = alpha
                tmp_kwargs['max_value'] = max_value
                tmp_kwargs['threshold'] = threshold
            elif activation_name in ['gelu']:
                approximate = False
                if 'approximate' in kwargs.keys():
                        approximate = kwargs['approximate']
                tmp_kwargs ={}
                tmp_kwargs['approximate'] = approximate
            elif activation_name in ['softmax']:
                axis = -1
                if 'axis' in kwargs.keys():
                        axis = kwargs['axis']
                tmp_kwargs ={}
                tmp_kwargs['axis'] = axis
            elif activation_name in ['domain_shift_sigmoid']:
                activation_name = 'sigmoid'
                domain = kwargs['output_domain']
                scale = max(domain)-min(domain) #与sigmoid的0-1比较
                offset = min(domain) #与sigmoid的0-1比较
                tmp_kwargs ={}
                call = activation_slect(activation_name)#摒弃原call函数 因为一定需要被重载
                _special_flag=True
                def activation_call(x,training):
                    return scale*call(x,**tmp_kwargs)+offset
            else:
                tmp_kwargs ={}
            if _special_flag:
                pass
            else:
                call = activation_slect(activation_name)#摒弃原call函数 因为一定需要被重载
                def activation_call(x,training):
                    return call(x,**tmp_kwargs)
            return activation_call
        self.call = _activation_wrapper(self.activation_name,**kwargs)
    def build(self,input_shape):
        super(Activation,self).build(input_shape)
        return input_shape[:]
    def call(self,x,training):
        raise ValueError("Activation call func need to be override again.")


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    act = Activation('linear')
    x = tf.random.normal(shape=[1,2,3])
    print(y:=act(x))
    act = Activation('relu',alpha=0.1,dtype=policy)
    x = tf.random.normal(shape=[1,2,3])
    print(y:=act(x))
    print(act.name)
    act = Activation('relu',alpha=0.1,dtype=policy)
    x = tf.random.normal(shape=[1,2,3])
    print(y:=act(x))
    print(act.name)

    dp = Dropout(rate=0.6,noise_shape=[10,1],dtype=policy)
    x = tf.ones(shape=[10,10])
    print(y:=dp(x,training=True))
    print(y:=dp(x,training=False))
    x = tf.keras.backend.random_bernoulli([1000], p=0.3)

    dp=tf.keras.layers.Dropout(0.5,noise_shape=[10,1],dtype=policy)
    x = tf.ones(shape=[10,10])
    print(y:=dp(x,training=True))
    print(y:=dp(x,training=False))
    print(y:=dp(x,training=None))
    print(y:=dp(x))

    act = Activation('domain_shift_sigmoid',output_domain=[-10.0,30.0],dtype=policy)
    x = tf.random.normal(shape=[10,10])
    print(y:=act(x))
    print(act.name)


   
