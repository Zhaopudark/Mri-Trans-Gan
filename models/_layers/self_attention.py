"""
Attention 模块相关
"""
import sys
import os
import tensorflow as tf
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)
from convolutions.conv2d import Conv2D
from convolutions.conv3d import Conv3D
class ScaledPointAttention():
    def __init__(self,
                 scaling=8,
                 name=None,
                 dtype=None):
        super(ScaledPointAttention,self).__init__(name=name,dtype=dtype)
        self.inner_dtype = dtype
        self.scaling = scaling
    def build(self,input_shape):
        pass
    def call(self,x,training=True):
        pass
class SelfAtten2D(tf.keras.Model):
    def __init__(self,
                 scaling=8,
                 name=None,
                 dtype=None):
        super(SelfAtten2D,self).__init__(name=name,dtype=dtype)
        self.inner_dtype = dtype
        self.scaling = scaling
    def build(self,input_shape):
        if len(input_shape)==4:
            channels = int(input_shape[-1])
            filters = int(channels//self.scaling)
            if filters*self.scaling!=channels:
                raise ValueError("Channels can not be divisible by scaling {}.".format(self.scaling))
            if input_shape[0] is None:
                batch_size = int(input_shape[0])
            else:
                batch_size = 1
            self.dims = [int(input_shape[1]),int(input_shape[2])]
            self.N = self.dims[0]*self.dims[1]
            self.C_ = filters
            self.C = channels
        else:
            raise ValueError("SelfAtten2D needs 4 dims for input tensor.")
        
        self.Wg = Conv2D(filters=filters,kernel_size=[1,1],strides=[1,1],padding="valid",dtype=self.inner_dtype) 
        self.Wf = Conv2D(filters=filters,kernel_size=[1,1],strides=[1,1],padding="valid",dtype=self.inner_dtype) 
        self.Wh = Conv2D(filters=channels,kernel_size=[1,1],strides=[1,1],padding="valid",dtype=self.inner_dtype)
        self.gamma = self.add_weight('gamma',shape=(channels,),initializer=tf.initializers.Zeros(),trainable=True)#不同通道分开

        _ = self.Wg.build(input_shape=input_shape)
        _ = self.Wf.build(input_shape=input_shape)
        flow_shape = self.Wh.build(input_shape=input_shape)
        self.built = True
        output_shape = flow_shape[:]
        return output_shape
    def call(self,x,training=True):
        x_g = tf.reshape(self.Wg(x),shape=[-1,self.N,self.C_])
        x_f = tf.reshape(self.Wf(x),shape=[-1,self.N,self.C_])
        x_h = tf.reshape(self.Wh(x),shape=[-1,self.N,self.C])
        s = x_g@tf.transpose(x_f,perm=[0,2,1])#B N N
        s = tf.nn.softmax(s,axis=-1)  #沿着最后一维计算softmax
        o = s@x_h                     #[B,N,N]@[B,N,C]
        o = tf.reshape(o,shape=[-1,self.dims[0],self.dims[1],self.C])
        y = self.gamma*o+x
        return y

class SelfAtten3D(tf.keras.Model):
    def __init__(self,
                 scaling=8,
                 name=None,
                 dtype=None):
        super(SelfAtten3D,self).__init__(name=name,dtype=dtype)
        self.inner_dtype = dtype
        self.scaling = scaling
    def build(self,input_shape):
        if len(input_shape)==5:
            channels = int(input_shape[-1])
            filters = int(channels//self.scaling)
            if filters*self.scaling!=channels:
                raise ValueError("Channels can not be divisible by scaling {}.".format(self.scaling))
            if input_shape[0] is None:
                batch_size = int(input_shape[0])
            else:
                batch_size = 1
            self.dims = [int(input_shape[1]),int(input_shape[2]),int(input_shape[3])]
            self.N = self.dims[0]*self.dims[1]*self.dims[2]
            self.C_ = filters
            self.C = channels
        else:
            raise ValueError("SelfAtten3D needs 5 dims for input tensor.")
        
        self.Wg = Conv3D(filters=filters,kernel_size=[1,1,1],strides=[1,1,1],padding="valid",dtype=self.inner_dtype) 
        self.Wf = Conv3D(filters=filters,kernel_size=[1,1,1],strides=[1,1,1],padding="valid",dtype=self.inner_dtype) 
        self.Wh = Conv3D(filters=channels,kernel_size=[1,1,1],strides=[1,1,1],padding="valid",dtype=self.inner_dtype)
        self.gamma = self.add_weight('gamma',shape=(channels,),initializer=tf.initializers.Zeros(),trainable=True) #不同通道分开

        _ = self.Wg.build(input_shape=input_shape)
        _ = self.Wf.build(input_shape=input_shape)
        flow_shape = self.Wh.build(input_shape=input_shape)
        self.built = True
        output_shape = flow_shape[:]
        return output_shape
    def call(self,x,training=True):
        x_g = tf.reshape(self.Wg(x),shape=[-1,self.N,self.C_])
        x_f = tf.reshape(self.Wf(x),shape=[-1,self.N,self.C_])
        x_h = tf.reshape(self.Wh(x),shape=[-1,self.N,self.C ])
        s = x_g@tf.transpose(x_f,perm=[0,2,1])#B N N
        s = tf.nn.softmax(s,axis=-1)  #沿着最后一维计算softmax
        o = s@x_h                     #[B,N,N]@[B,N,C]
        o = tf.reshape(o,shape=[-1,self.dims[0],self.dims[1],self.dims[2],self.C])
        y = self.gamma*o+x
        return y

    
if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    x = tf.random.normal(shape=[1,7,9,24])
    Wg = Conv2D(filters=3,kernel_size=[1,1],strides=[1,1],padding="valid") 
    Wf = Conv2D(filters=3,kernel_size=[1,1],strides=[1,1],padding="valid") 
    Wh = Conv2D(filters=24,kernel_size=[1,1],strides=[1,1],padding="valid")
    y1 = Wg(x)
    print(y1.shape,y1.dtype)
    y2 = Wf(x)
    print(y2.shape,y2.dtype)
    y3 = Wh(x)
    print(y3.shape,y3.dtype)
    
    a = tf.random.normal(shape=[1,63,3])
    b = tf.random.normal(shape=[1,3,63])
    s = tf.matmul(a,b)
    print(s.shape,s.dtype)

    a = tf.constant([[[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]],[[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]]])
    b = tf.nn.softmax(a)
    print(a)
    print(b)
    a = tf.constant([[[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]],[[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]]])
    b = tf.nn.softmax(a,axis=0)
    print(b)
    a = tf.constant([[[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]],[[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]]])
    b = tf.nn.softmax(a,axis=1)
    print(b)
    a = tf.constant([[[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]],[[1.,2.,3.],[1.,2.,3.],[1.,2.,3.]]])
    b = tf.nn.softmax(a,axis=2)
    print(b)
    x = tf.random.normal(shape=[12,32,32,32])
    model = SelfAtten2D()
    model.build(input_shape=[12,32,32,32])
    y = model(x)
    print(y.shape,y.dtype)

    x = tf.random.normal(shape=[1,24,24,24,16])
    model = SelfAtten3D()
    y = model(x)
    print(y.shape,y.dtype)