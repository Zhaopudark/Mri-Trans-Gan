import sys
import os
import tensorflow as tf
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)
from activations import activation_slect
from initializers import initializer_slect
__all__ = [
    "SubpixelConv1D",
    "SubpixelConv2D",
    "SubpixelConv3D",
]
class SubpixelConv1D(tf.keras.layers.Layer):
    def __init__(self,
                 scale,
                 activation=None,
                 name=None,
                 dtype=None):
        super(SubpixelConv1D,self).__init__(name=name,dtype=dtype)
        if not isinstance(scale,list):
            raise ValueError("SubpixelConv1D needs 1-D list for scale.") 
        if len(scale)!=1:
            raise ValueError("SubpixelConv1D needs 1-D list for scale.") 
        self.scale = tf.constant(scale,dtype=tf.int32)
        self.activation = activation_slect(activation)
    def build(self,input_shape):
        super(SubpixelConv1D,self).build(input_shape=input_shape)
        if len(input_shape)!=3:
            raise ValueError("SubpixelConv1D needs 3-D input.")
        in_channel = input_shape[-1]
        out_channel = int(in_channel/tf.math.reduce_prod(self.scale))
        traget_in_channel = out_channel*tf.math.reduce_prod(self.scale)
        if int(traget_in_channel) != int(in_channel):
            raise ValueError("Unsupported channel or scale shape.") 
        output_shape = [input_shape[0]]
        for index in range(1,len(input_shape)-1):
            output_shape.append(int(input_shape[index]*self.scale[index-1]))
        output_shape.append(out_channel)
        return output_shape
    def call(self,x,training):
        x = self.pixel_shuffle(x=x,r=self.scale)
        y = self.activation(x)
        return y
    def pixel_shuffle(self,x,r):
        x = tf.transpose(x,[2, 1, 0])  # [B-(DHW)-C] -> [C-(WHD)-B]
        r = tf.reverse(r,axis=[-1])
        x = tf.batch_to_space(input=x, block_shape=r, crops=[[0, 0]])  # (C/r,W*H*D,B)
        x = tf.transpose(x,[2, 1, 0])  # [C-(WHD)-B] -> [B-(DHW)-C]
        return x
class SubpixelConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 scale,
                 activation=None,
                 name=None,
                 dtype=None):
        super(SubpixelConv2D,self).__init__(name=name,dtype=dtype)
        if not isinstance(scale,list):
            raise ValueError("SubpixelConv2D needs 2-D list for scale.") 
        if len(scale)!=2:
            raise ValueError("SubpixelConv2D needs 2-D list for scale.") 
        self.scale = tf.constant(scale,dtype=tf.int32)
        self.activation = activation_slect(activation)
    def build(self,input_shape):
        super(SubpixelConv2D,self).build(input_shape=input_shape)
        if len(input_shape)!=4:
            raise ValueError("SubpixelConv2D needs 4-D input.")
        in_channel = input_shape[-1]
        out_channel = int(in_channel/tf.math.reduce_prod(self.scale))
        traget_in_channel = out_channel*tf.math.reduce_prod(self.scale)
        if int(traget_in_channel) != int(in_channel):
            raise ValueError("Unsupported channel or scale shape.") 
        output_shape = [input_shape[0]]
        for index in range(1,len(input_shape)-1):
            output_shape.append(int(input_shape[index]*self.scale[index-1]))
        output_shape.append(out_channel)
        return output_shape
    def call(self,x,training):
        x = self.pixel_shuffle(x=x,r=self.scale)
        y = self.activation(x)
        return y
    def pixel_shuffle(self,x,r):
        # x = tf.nn.depth_to_space(input=x,block_size=r) # NOTE depth_to_space无法支持维度间不同缩放尺度的操作
        x = tf.transpose(x,[3, 2, 1, 0])  # [B-H-W-C] -> [C-W-H-B]
        r = tf.reverse(r,axis=[-1])
        x = tf.batch_to_space(input=x, block_shape=r, crops=[[0, 0],[0, 0]])  # (C/r**3,W,H,B)
        x = tf.transpose(x,[3, 2, 1, 0])  # [C-W-H-B] -> [B-H-W-C]
        return x

class SubpixelConv3D(tf.keras.layers.Layer):
    def __init__(self,
                 scale,
                 activation=None,
                 name=None,
                 dtype=None):
        super(SubpixelConv3D,self).__init__(name=name,dtype=dtype)
        if not isinstance(scale,list):
            raise ValueError("SubpixelConv3D needs 3-D list for scale.") 
        if len(scale)!=3:
            raise ValueError("SubpixelConv3D needs 3-D list for scale.") 
        self.scale = tf.constant(scale,dtype=tf.int32)
        self.activation = activation_slect(activation)
    def build(self,input_shape):
        super(SubpixelConv3D,self).build(input_shape=input_shape)
        if len(input_shape)!=5:
            raise ValueError("SubpixelConv3D needs 5-D input.")
        in_channel = input_shape[-1]
        out_channel = int(in_channel/tf.math.reduce_prod(self.scale))
        traget_in_channel = out_channel*tf.math.reduce_prod(self.scale)
        if int(traget_in_channel) != int(in_channel):
            raise ValueError("Unsupported channel or scale shape.") 
        output_shape = [input_shape[0]]
        for index in range(1,len(input_shape)-1):
            output_shape.append(int(input_shape[index]*self.scale[index-1]))
        output_shape.append(out_channel)
        return output_shape
    def call(self,x,training):
        x = self.pixel_shuffle(x=x,r=self.scale)
        y = self.activation(x)
        return y
    def pixel_shuffle(self,x,r):
        x = tf.transpose(x,[4, 3, 2, 1, 0])  # [B-D-H-W-C] -> [C-W-H-D-B]
        r = tf.reverse(r,axis=[-1])
        x = tf.batch_to_space(input=x, block_shape=r, crops=[[0, 0],[0, 0],[0, 0]])  # (C/r**3,W,H,D,B)
        x = tf.transpose(x,[4, 3, 2, 1, 0])  # [C-W-H-D-B] -> [B-D-H-W-C]
        return x

if __name__=="__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    x = tf.random.normal([4, 16, 16, 512])
    model = SubpixelConv2D(scale=[2,2])
    y =  model(x)
    print(y.shape,y.dtype)
    # policy = tf.keras.mixed_precision.Policy('mixed_float16')
    # model = SubpixelConv2D(scale=4,dtype=policy)
    # y =  model(x)
    # print(y.shape,y.dtype)
    # policy = tf.keras.mixed_precision.Policy('float32')
    # model = SubpixelConv2D(scale=4,dtype=policy)
    # y =  model(x)
    # print(y.shape,y.dtype)
    
    # x0 = tf.transpose(x,[3, 2, 1,0]) 
    # x0 = tf.batch_to_space(x0, [2,2], [[0, 0],[0, 0]])
    # y0 = tf.transpose(x0,[3, 2, 1,0])
    # print(y0.shape,y0.dtype)

    # tmp = y-y0
    # print(tf.reduce_mean(tmp))
    
    x1 = tf.random.normal([4,16,512])
    x2 = tf.random.normal([4,16,16,512])
    x3 = tf.random.normal([4,16,16,16,512])
    model1 = SubpixelConv1D(scale=[2])
    model2 = SubpixelConv2D(scale=[2,2])
    model3 = SubpixelConv3D(scale=[2,2,2])
    print(model1.build(input_shape=[None,16,512]))
    print(model2.build(input_shape=[None,16,16,512]))
    print(model3.build(input_shape=[None,16,16,16,512]))
    y1 = model1(x1)
    y2 = model2(x2)
    y3 = model3(x3)
    print(y1.shape,y1.dtype)
    print(y2.shape,y2.dtype)
    print(y3.shape,y3.dtype)

    x1 = tf.random.normal([4,16,512])
    x2 = tf.random.normal([4,16,16,512])
    x3 = tf.random.normal([4,16,16,16,512])
    model1 = SubpixelConv1D(scale=[1])
    model2 = SubpixelConv2D(scale=[1,2])
    model3 = SubpixelConv3D(scale=[1,2,2])
    print(model1.build(input_shape=[None,16,512]))
    print(model2.build(input_shape=[None,16,16,512]))
    print(model3.build(input_shape=[None,16,16,16,512]))
    y1 = model1(x1)
    y2 = model2(x2)
    y3 = model3(x3)
    print(y1.shape,y1.dtype)
    print(y2.shape,y2.dtype)
    print(y3.shape,y3.dtype)






