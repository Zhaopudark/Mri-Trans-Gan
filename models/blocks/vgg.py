import sys
import os
__all__ = [
    "Vgg16LayerBuf_V2",
    "Vgg16LayerBuf_V4",
]
import tensorflow as tf
import numpy as np 
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base,'../'))
from craft.convolutions.conv2d import Conv2DVgg
from craft.convolutions.conv3d import Vgg2Conv3D
from craft.denses import DenseVgg
class Vgg16(tf.keras.Model):
    def __init__(self,path="D:\\Datasets\\VGG\\vgg16.npy",
                 name=None,
                 dtype=None):
        self.path = path 
        self.data_dict = np.load(self.path,encoding='latin1',allow_pickle=True).item()
        super(Vgg16,self).__init__(name=name,dtype=dtype)
        self.conv1_1 = Conv2DVgg(filters=64,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv1_2 = Conv2DVgg(filters=64,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.l1_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.conv2_1 = Conv2DVgg(filters=128,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv2_2 = Conv2DVgg(filters=128,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.l2_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.conv3_1 = Conv2DVgg(filters=256,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv3_2 = Conv2DVgg(filters=256,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv3_3 = Conv2DVgg(filters=256,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.l3_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.conv4_1 = Conv2DVgg(filters=512,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv4_2 = Conv2DVgg(filters=512,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv4_3 = Conv2DVgg(filters=512,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.l4_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.conv5_1 = Conv2DVgg(filters=512,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv5_2 = Conv2DVgg(filters=512,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv5_3 = Conv2DVgg(filters=512,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.l5_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.flatten= tf.keras.layers.Flatten()
        self.dense6 = DenseVgg(kernel_size=4096,use_bias=True,activation=None,dtype=dtype)
        self.dense7 = DenseVgg(kernel_size=4096,use_bias=True,activation=None,dtype=dtype)
        self.dense8 = DenseVgg(kernel_size=1000,use_bias=True,activation=None,dtype=dtype)
    def build(self,input_shape):
        flow_shape=self.conv1_1.build(input_shape=input_shape)
        flow_shape=self.conv1_2.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv2_1.build(input_shape=flow_shape) 
        flow_shape=self.conv2_2.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv3_1.build(input_shape=flow_shape) 
        flow_shape=self.conv3_2.build(input_shape=flow_shape) 
        flow_shape=self.conv3_3.build(input_shape=flow_shape)
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv4_1.build(input_shape=flow_shape) 
        flow_shape=self.conv4_2.build(input_shape=flow_shape) 
        flow_shape=self.conv4_3.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv5_1.build(input_shape=flow_shape) 
        flow_shape=self.conv5_2.build(input_shape=flow_shape) 
        flow_shape=self.conv5_3.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        tmp = 1
        for index in range(len(flow_shape)-1):
            tmp = tmp*flow_shape[index+1]
        flow_shape = [flow_shape[0],tmp]
        flow_shape=self.dense6.build(input_shape=flow_shape)
        flow_shape=self.dense7.build(input_shape=flow_shape)
        flow_shape=self.dense8.build(input_shape=flow_shape)
        self.built = True
        output_shape = flow_shape[:]

        self.conv1_1.w.assign(self.data_dict["conv1_1"][0])
        self.conv1_1.b.assign(self.data_dict["conv1_1"][1])
        self.conv1_2.w.assign(self.data_dict["conv1_2"][0])
        self.conv1_2.b.assign(self.data_dict["conv1_2"][1])
        self.conv2_1.w.assign(self.data_dict["conv2_1"][0])
        self.conv2_1.b.assign(self.data_dict["conv2_1"][1])
        self.conv2_2.w.assign(self.data_dict["conv2_2"][0])
        self.conv2_2.b.assign(self.data_dict["conv2_2"][1])
        self.conv3_1.w.assign(self.data_dict["conv3_1"][0])
        self.conv3_1.b.assign(self.data_dict["conv3_1"][1])
        self.conv3_2.w.assign(self.data_dict["conv3_2"][0])
        self.conv3_2.b.assign(self.data_dict["conv3_2"][1])
        self.conv3_3.w.assign(self.data_dict["conv3_3"][0])
        self.conv3_3.b.assign(self.data_dict["conv3_3"][1])
        self.conv4_1.w.assign(self.data_dict["conv4_1"][0])
        self.conv4_1.b.assign(self.data_dict["conv4_1"][1])
        self.conv4_2.w.assign(self.data_dict["conv4_2"][0])
        self.conv4_2.b.assign(self.data_dict["conv4_2"][1])
        self.conv4_3.w.assign(self.data_dict["conv4_3"][0])
        self.conv4_3.b.assign(self.data_dict["conv4_3"][1])
        self.conv5_1.w.assign(self.data_dict["conv5_1"][0])
        self.conv5_1.b.assign(self.data_dict["conv5_1"][1])
        self.conv5_2.w.assign(self.data_dict["conv5_2"][0])
        self.conv5_2.b.assign(self.data_dict["conv5_2"][1])
        self.conv5_3.w.assign(self.data_dict["conv5_3"][0])
        self.conv5_3.b.assign(self.data_dict["conv5_3"][1])
        self.dense6.w.assign(self.data_dict["fc6"][0])
        self.dense6.b.assign(self.data_dict["fc6"][1])
        self.dense7.w.assign(self.data_dict["fc7"][0])
        self.dense7.b.assign(self.data_dict["fc7"][1])
        self.dense8.w.assign(self.data_dict["fc8"][0])
        self.dense8.b.assign(self.data_dict["fc8"][1])

        # ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
        return output_shape
    def call(self,x,training=True,scale=8):
        x = tf.broadcast_to(x,shape=x.shape[0:-1]+[3])
        if scale < 1:
            return x 
        x=self.conv1_1(x,training=training)
        x=self.conv1_2(x,training=training)
        if scale < 2:
            return x 
        x=self.l1_max_pool(x,training=training)
        x=self.conv2_1(x,training=training)
        x=self.conv2_2(x,training=training)
        if scale < 3:
            return x
        x=self.l2_max_pool(x,training=training)
        x=self.conv3_1(x,training=training)
        x=self.conv3_2(x,training=training)
        x=self.conv3_3(x,training=training)
        if scale < 4:
            return x
        x=self.l3_max_pool(x,training=training)
        x=self.conv4_1(x,training=training)
        x=self.conv4_2(x,training=training)
        x=self.conv4_3(x,training=training)
        if scale < 5:
            return x
        x=self.l4_max_pool(x,training=training)
        x=self.conv5_1(x,training=training)
        x=self.conv5_2(x,training=training)
        x=self.conv5_3(x,training=training)
        if scale < 6:
            return x
        x=self.l5_max_pool(x,training=training)
        x = self.flatten(x)
        x=self.dense6(x,training=training)
        if scale < 7:
            return x
        x=self.dense7(x,training=training)
        if scale < 8:
            return x
        x=self.dense8(x,training=training)
        return x
#######################################################################################################
class Vgg16LayerBuf(tf.keras.Model):
    # 输出VGG16的前指定若干层
    def __init__(self,path="D:\\Datasets\\VGG\\vgg16.npy",
                 name=None,
                 dtype=None):
        self.path = path 
        self.data_dict = np.load(self.path,encoding='latin1',allow_pickle=True).item()
        super(Vgg16LayerBuf,self).__init__(name=name,dtype=dtype)
        self.conv1_1 = Conv2DVgg(filters=64,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv1_2 = Conv2DVgg(filters=64,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.l1_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.conv2_1 = Conv2DVgg(filters=128,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv2_2 = Conv2DVgg(filters=128,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.l2_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.conv3_1 = Conv2DVgg(filters=256,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv3_2 = Conv2DVgg(filters=256,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv3_3 = Conv2DVgg(filters=256,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.l3_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.conv4_1 = Conv2DVgg(filters=512,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv4_2 = Conv2DVgg(filters=512,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv4_3 = Conv2DVgg(filters=512,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.l4_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.conv5_1 = Conv2DVgg(filters=512,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv5_2 = Conv2DVgg(filters=512,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv5_3 = Conv2DVgg(filters=512,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.l5_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.flatten= tf.keras.layers.Flatten()
        self.dense6 = DenseVgg(kernel_size=4096,use_bias=True,activation=None,dtype=dtype)
        self.dense7 = DenseVgg(kernel_size=4096,use_bias=True,activation=None,dtype=dtype)
        self.dense8 = DenseVgg(kernel_size=1000,use_bias=True,activation=None,dtype=dtype)
    def build(self,input_shape):
        input_shape = [None,224,224,3]
        flow_shape=self.conv1_1.build(input_shape=input_shape)
        flow_shape=self.conv1_2.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv2_1.build(input_shape=flow_shape) 
        flow_shape=self.conv2_2.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv3_1.build(input_shape=flow_shape) 
        flow_shape=self.conv3_2.build(input_shape=flow_shape) 
        flow_shape=self.conv3_3.build(input_shape=flow_shape)
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv4_1.build(input_shape=flow_shape) 
        flow_shape=self.conv4_2.build(input_shape=flow_shape) 
        flow_shape=self.conv4_3.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv5_1.build(input_shape=flow_shape) 
        flow_shape=self.conv5_2.build(input_shape=flow_shape) 
        flow_shape=self.conv5_3.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        tmp = 1
        for index in range(len(flow_shape)-1):
            tmp = tmp*flow_shape[index+1]
        flow_shape = [flow_shape[0],tmp]
        flow_shape=self.dense6.build(input_shape=flow_shape)
        flow_shape=self.dense7.build(input_shape=flow_shape)
        flow_shape=self.dense8.build(input_shape=flow_shape)
        self.built = True
        output_shape = flow_shape[:]

        self.conv1_1.w.assign(self.data_dict["conv1_1"][0])
        self.conv1_1.b.assign(self.data_dict["conv1_1"][1])
        self.conv1_2.w.assign(self.data_dict["conv1_2"][0])
        self.conv1_2.b.assign(self.data_dict["conv1_2"][1])
        self.conv2_1.w.assign(self.data_dict["conv2_1"][0])
        self.conv2_1.b.assign(self.data_dict["conv2_1"][1])
        self.conv2_2.w.assign(self.data_dict["conv2_2"][0])
        self.conv2_2.b.assign(self.data_dict["conv2_2"][1])
        self.conv3_1.w.assign(self.data_dict["conv3_1"][0])
        self.conv3_1.b.assign(self.data_dict["conv3_1"][1])
        self.conv3_2.w.assign(self.data_dict["conv3_2"][0])
        self.conv3_2.b.assign(self.data_dict["conv3_2"][1])
        self.conv3_3.w.assign(self.data_dict["conv3_3"][0])
        self.conv3_3.b.assign(self.data_dict["conv3_3"][1])
        self.conv4_1.w.assign(self.data_dict["conv4_1"][0])
        self.conv4_1.b.assign(self.data_dict["conv4_1"][1])
        self.conv4_2.w.assign(self.data_dict["conv4_2"][0])
        self.conv4_2.b.assign(self.data_dict["conv4_2"][1])
        self.conv4_3.w.assign(self.data_dict["conv4_3"][0])
        self.conv4_3.b.assign(self.data_dict["conv4_3"][1])
        self.conv5_1.w.assign(self.data_dict["conv5_1"][0])
        self.conv5_1.b.assign(self.data_dict["conv5_1"][1])
        self.conv5_2.w.assign(self.data_dict["conv5_2"][0])
        self.conv5_2.b.assign(self.data_dict["conv5_2"][1])
        self.conv5_3.w.assign(self.data_dict["conv5_3"][0])
        self.conv5_3.b.assign(self.data_dict["conv5_3"][1])
        self.dense6.w.assign(self.data_dict["fc6"][0])
        self.dense6.b.assign(self.data_dict["fc6"][1])
        self.dense7.w.assign(self.data_dict["fc7"][0])
        self.dense7.b.assign(self.data_dict["fc7"][1])
        self.dense8.w.assign(self.data_dict["fc8"][0])
        self.dense8.b.assign(self.data_dict["fc8"][1])

        # ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
        return output_shape
    def call(self,x,training=True,scale=4):
        layer_buf = []
        if len(x.shape)==5:
            assert x.shape[-1]==0
            assert x.shape[-2]==3
            x = tf.reshape(x,shape=[1,128,128,3]) # 3D视为3通道的形式
        elif len(x.shape)==4:
            if x.shape[-1]==1:
                x = tf.broadcast_to(x,shape=x.shape[0:-1]+[3])
                x = x/3
            elif x.shape[-1]==3:
                pass
            else:
                raise ValueError("Vgg input shape mast be [1,x,x,1] or [1,x,x,3] or [1,x,x,3,1]")
        else:
            raise ValueError("Vgg input shape mast be [1,x,x,1] or [1,x,x,3] or [1,x,x,3,1]")
        # x = tf.broadcast_to(x,shape=x.shape[0:-1]+[3])
        # x = tf.reshape(x,shape=[1,128,128,3])

        layer_buf.append(x)
        if scale < 1:
            return layer_buf

        x=self.conv1_1(x,training=training)
        x=self.conv1_2(x,training=training)

        layer_buf.append(x)
        if scale < 2:
            return layer_buf 

        x=self.l1_max_pool(x,training=training)
        x=self.conv2_1(x,training=training)
        x=self.conv2_2(x,training=training)

        layer_buf.append(x)
        if scale < 3:
            return layer_buf

        x=self.l2_max_pool(x,training=training)
        x=self.conv3_1(x,training=training)
        x=self.conv3_2(x,training=training)
        x=self.conv3_3(x,training=training)

        layer_buf.append(x)
        if scale < 4:
            return layer_buf

        x=self.l3_max_pool(x,training=training)
        x=self.conv4_1(x,training=training)
        x=self.conv4_2(x,training=training)
        x=self.conv4_3(x,training=training)

        layer_buf.append(x)
        if scale < 5:
            return layer_buf

        x=self.l4_max_pool(x,training=training)
        x=self.conv5_1(x,training=training)
        x=self.conv5_2(x,training=training)
        x=self.conv5_3(x,training=training)
        layer_buf.append(x)
        if scale < 6:
            return layer_buf
        x=self.l5_max_pool(x,training=training)
        x = self.flatten(x)
        x=self.dense6(x,training=training)

        layer_buf.append(x)
        if scale < 7:
            return layer_buf

        x=self.dense7(x,training=training)

        layer_buf.append(x)
        if scale < 8:
            return layer_buf

        x=self.dense8(x,training=training)
        layer_buf.append(x)
        return layer_buf
######################################################
#######################################################################################################
class Vgg16LayerBuf_V2(tf.keras.Model):
    # 输出VGG16的前指定若干层
    def __init__(self,path="D:\\Datasets\\VGG\\vgg16.npy",
                 name=None,
                 dtype=None):
        self.path = path 
        self.data_dict = np.load(self.path,encoding='latin1',allow_pickle=True).item()
        super(Vgg16LayerBuf_V2,self).__init__(name=name,dtype=dtype)
        self.conv1_1 = Conv2DVgg(filters=64,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv1_2 = Conv2DVgg(filters=64,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)

        self.conv2_1 = Conv2DVgg(filters=128,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv2_2 = Conv2DVgg(filters=128,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
       
        self.conv3_1 = Conv2DVgg(filters=256,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv3_2 = Conv2DVgg(filters=256,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv3_3 = Conv2DVgg(filters=256,kernel_size=[3,3],strides=[1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
      
    def build(self,input_shape):
        input_shape = [None,224,224,3]
        flow_shape=self.conv1_1.build(input_shape=input_shape)
        flow_shape=self.conv1_2.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv2_1.build(input_shape=flow_shape) 
        flow_shape=self.conv2_2.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv3_1.build(input_shape=flow_shape) 
        flow_shape=self.conv3_2.build(input_shape=flow_shape) 
        flow_shape=self.conv3_3.build(input_shape=flow_shape)
        output_shape = flow_shape[:]
        self.conv1_1.w.assign(self.data_dict["conv1_1"][0])
        self.conv1_1.b.assign(self.data_dict["conv1_1"][1])
        self.conv1_2.w.assign(self.data_dict["conv1_2"][0])
        self.conv1_2.b.assign(self.data_dict["conv1_2"][1])
        self.conv2_1.w.assign(self.data_dict["conv2_1"][0])
        self.conv2_1.b.assign(self.data_dict["conv2_1"][1])
        self.conv2_2.w.assign(self.data_dict["conv2_2"][0])
        self.conv2_2.b.assign(self.data_dict["conv2_2"][1])
        self.conv3_1.w.assign(self.data_dict["conv3_1"][0])
        self.conv3_1.b.assign(self.data_dict["conv3_1"][1])
        self.conv3_2.w.assign(self.data_dict["conv3_2"][0])
        self.conv3_2.b.assign(self.data_dict["conv3_2"][1])
        self.conv3_3.w.assign(self.data_dict["conv3_3"][0])
        self.conv3_3.b.assign(self.data_dict["conv3_3"][1])
 

        # ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
        return output_shape
    def call(self,x,training=True,scale=4):
        layer_buf = []
        x = tf.broadcast_to(x,shape=x.shape[0:-1]+[3])
        x = x/3
        layer_buf.append(x)
        if scale < 1:
            return layer_buf
        x=self.conv1_1(x,training=training)
        layer_buf.append(x)
        if scale < 2:
            return layer_buf
        x=self.conv1_2(x,training=training)
        layer_buf.append(x)
        if scale < 3:
            return layer_buf 
        x=self.conv2_1(x,training=training)
        layer_buf.append(x)
        if scale < 4:
            return layer_buf
        x=self.conv2_2(x,training=training)
        layer_buf.append(x)
        if scale < 5:
            return layer_buf
        x=self.conv3_1(x,training=training)
        layer_buf.append(x)
        if scale < 6:
            return layer_buf
        x=self.conv3_2(x,training=training)
        layer_buf.append(x)
        if scale < 7:
            return layer_buf
        x=self.conv3_3(x,training=training)
        layer_buf.append(x)
        if scale < 8:
            return layer_buf
        else:
            raise ValueError("Layers Must Under 7")
#######################################################
class Vgg16LayerBuf_V3(tf.keras.Model):
    # 输出VGG16的前指定若干层
    def __init__(self,path="D:\\Datasets\\VGG\\vgg16.npy",
                 name=None,
                 dtype=None):
        self.path = path 
        self.data_dict = np.load(self.path,encoding='latin1',allow_pickle=True).item()
        super(Vgg16LayerBuf_V3,self).__init__(name=name,dtype=dtype)
        self.conv1_1 = Vgg2Conv3D(filters=64,kernel_size=[3,3,1],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv1_2 = Vgg2Conv3D(filters=64,kernel_size=[3,3,1],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        # self.l1_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.conv2_1 = Vgg2Conv3D(filters=128,kernel_size=[3,3,1],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv2_2 = Vgg2Conv3D(filters=128,kernel_size=[3,3,1],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)

        self.conv3_1 = Vgg2Conv3D(filters=256,kernel_size=[3,3,1],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv3_2 = Vgg2Conv3D(filters=256,kernel_size=[3,3,1],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv3_3 = Vgg2Conv3D(filters=256,kernel_size=[3,3,1],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
    def build(self,input_shape):
        input_shape = [None,224,224,1,3]
        flow_shape=self.conv1_1.build(input_shape=input_shape)
        flow_shape=self.conv1_2.build(input_shape=flow_shape) 
        flow_shape[-4]//=2
        flow_shape[-3]//=2
        flow_shape=self.conv2_1.build(input_shape=flow_shape) 
        flow_shape=self.conv2_2.build(input_shape=flow_shape) 
        flow_shape[-4]//=2
        flow_shape[-3]//=2
        flow_shape=self.conv3_1.build(input_shape=flow_shape) 
        flow_shape=self.conv3_2.build(input_shape=flow_shape) 
        flow_shape=self.conv3_3.build(input_shape=flow_shape)
        output_shape = flow_shape[:]
     
        self.conv1_1.w.assign(tf.reshape(self.data_dict["conv1_1"][0],[3,3,1,3,64]))
        
        self.conv1_1.b.assign(self.data_dict["conv1_1"][1])
        self.conv1_2.w.assign(tf.reshape(self.data_dict["conv1_2"][0],[3,3,1,64,64]))
        self.conv1_2.b.assign(self.data_dict["conv1_2"][1])
        self.conv2_1.w.assign(tf.reshape(self.data_dict["conv2_1"][0],[3,3,1,64,128]))
        self.conv2_1.b.assign(self.data_dict["conv2_1"][1])
        self.conv2_2.w.assign(tf.reshape(self.data_dict["conv2_2"][0],[3,3,1,128,128]))
        self.conv2_2.b.assign(self.data_dict["conv2_2"][1])
        self.conv3_1.w.assign(tf.reshape(self.data_dict["conv3_1"][0],[3,3,1,128,256]))
        self.conv3_1.b.assign(self.data_dict["conv3_1"][1])
        self.conv3_2.w.assign(tf.reshape(self.data_dict["conv3_2"][0],[3,3,1,256,256]))
        self.conv3_2.b.assign(self.data_dict["conv3_2"][1])
        self.conv3_3.w.assign(tf.reshape(self.data_dict["conv3_3"][0],[3,3,1,256,256]))
        self.conv3_3.b.assign(self.data_dict["conv3_3"][1])

        # print(self.conv1_1.b.shape)
        # print(self.conv1_2.w.shape)
        # print(self.conv1_2.b.shape)
        # print(self.conv2_1.w.shape)
        # print(self.conv2_1.b.shape)
        # print(self.conv2_2.w.shape)
        # print(self.conv2_2.b.shape)
        # print(self.conv3_1.w.shape)
        # print(self.conv3_1.b.shape)
        # print(self.conv3_2.w.shape)
        # print(self.conv3_2.b.shape)
        # print(self.conv3_3.w.shape)
        # print(self.conv3_3.b.shape)

        # ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
        return output_shape
    def call(self,x,training=True,scale=4):
        layer_buf = []
        x = tf.broadcast_to(x,x.shape[0:-1]+[3])
        x = x/3
        layer_buf.append(x)
        if scale < 1:
            return layer_buf
        x=self.conv1_1(x,training=training)
        layer_buf.append(x)
        if scale < 2:
            return layer_buf
        x=self.conv1_2(x,training=training)
        layer_buf.append(x)
        if scale < 3:
            return layer_buf 
        x=self.conv2_1(x,training=training)
        layer_buf.append(x)
        if scale < 4:
            return layer_buf
        x=self.conv2_2(x,training=training)
        layer_buf.append(x)
        if scale < 5:
            return layer_buf
        x=self.conv3_1(x,training=training)
        layer_buf.append(x)
        if scale < 6:
            return layer_buf
        x=self.conv3_2(x,training=training)
        layer_buf.append(x)
        if scale < 7:
            return layer_buf
        x=self.conv3_3(x,training=training)
        layer_buf.append(x)
        if scale < 8:
            return layer_buf
        else:
            raise ValueError("Layers Must Under 7")


#######################################################
class Vgg16LayerBuf_V4(tf.keras.Model):
    # 输出VGG16的前指定若干层
    def __init__(self,path="D:\\Datasets\\VGG\\vgg16.npy",
                 name=None,
                 dtype=None):
        self.path = path 
        self.data_dict = np.load(self.path,encoding='latin1',allow_pickle=True).item()
        super(Vgg16LayerBuf_V4,self).__init__(name=name,dtype=dtype)
        self.conv1_1 = Vgg2Conv3D(filters=64,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv1_2 = Vgg2Conv3D(filters=64,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        # self.l1_max_pool = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding='valid')
        self.conv2_1 = Vgg2Conv3D(filters=128,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv2_2 = Vgg2Conv3D(filters=128,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)

        self.conv3_1 = Vgg2Conv3D(filters=256,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv3_2 = Vgg2Conv3D(filters=256,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
        self.conv3_3 = Vgg2Conv3D(filters=256,kernel_size=[1,3,3],strides=[1,1,1],padding="SAME",use_bias=True,activation=None,dtype=dtype)
    def build(self,input_shape):
        input_shape = [None,1,224,224,3]
        flow_shape=self.conv1_1.build(input_shape=input_shape)
        flow_shape=self.conv1_2.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv2_1.build(input_shape=flow_shape) 
        flow_shape=self.conv2_2.build(input_shape=flow_shape) 
        flow_shape[-3]//=2
        flow_shape[-2]//=2
        flow_shape=self.conv3_1.build(input_shape=flow_shape) 
        flow_shape=self.conv3_2.build(input_shape=flow_shape) 
        flow_shape=self.conv3_3.build(input_shape=flow_shape)
        output_shape = flow_shape[:]
     
        self.conv1_1.w.assign(tf.reshape(self.data_dict["conv1_1"][0],[1,3,3,3,64]))
        self.conv1_1.b.assign(self.data_dict["conv1_1"][1])
        self.conv1_2.w.assign(tf.reshape(self.data_dict["conv1_2"][0],[1,3,3,64,64]))
        self.conv1_2.b.assign(self.data_dict["conv1_2"][1])
        self.conv2_1.w.assign(tf.reshape(self.data_dict["conv2_1"][0],[1,3,3,64,128]))
        self.conv2_1.b.assign(self.data_dict["conv2_1"][1])
        self.conv2_2.w.assign(tf.reshape(self.data_dict["conv2_2"][0],[1,3,3,128,128]))
        self.conv2_2.b.assign(self.data_dict["conv2_2"][1])
        self.conv3_1.w.assign(tf.reshape(self.data_dict["conv3_1"][0],[1,3,3,128,256]))
        self.conv3_1.b.assign(self.data_dict["conv3_1"][1])
        self.conv3_2.w.assign(tf.reshape(self.data_dict["conv3_2"][0],[1,3,3,256,256]))
        self.conv3_2.b.assign(self.data_dict["conv3_2"][1])
        self.conv3_3.w.assign(tf.reshape(self.data_dict["conv3_3"][0],[1,3,3,256,256]))
        self.conv3_3.b.assign(self.data_dict["conv3_3"][1])

        return output_shape
    def call(self,x,training=True,scale=4):
        layer_buf = []
        x = tf.broadcast_to(x,x.shape[0:-1]+[3])
        x = x/3
        layer_buf.append(x)
        if scale < 1:
            return layer_buf
        x=self.conv1_1(x,training=training)
        layer_buf.append(x)
        if scale < 2:
            return layer_buf
        x=self.conv1_2(x,training=training)
        layer_buf.append(x)
        if scale < 3:
            return layer_buf 
        x=self.conv2_1(x,training=training)
        layer_buf.append(x)
        if scale < 4:
            return layer_buf
        x=self.conv2_2(x,training=training)
        layer_buf.append(x)
        if scale < 5:
            return layer_buf
        x=self.conv3_1(x,training=training)
        layer_buf.append(x)
        if scale < 6:
            return layer_buf
        x=self.conv3_2(x,training=training)
        layer_buf.append(x)
        if scale < 7:
            return layer_buf
        x=self.conv3_3(x,training=training)
        layer_buf.append(x)
        if scale < 8:
            return layer_buf
        else:
            raise ValueError("Layers Must Under 7")

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # if self.mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    # policy = None
    # V = Vgg16LayerBuf_V3()
    # x = tf.random.normal(shape=[1,128,128,16,3])
    # V.build(input_shape=None)
    # print(V(x))
    # # print(time.time()-start)
    # V = Vgg16LayerBuf_V2()
    # x = tf.random.normal(shape=[1,128,128,1])
    # V.build(input_shape=None)
    # print(V(x))
    #-----------------------------V2-V3比较------------------------#
    import time
    V = Vgg16LayerBuf_V3(dtype=policy)
    V.build(input_shape=None)
    x = tf.random.uniform(shape=[1,128,128,16]) 
    x_ = tf.reshape(x,[1,128,128,16,1])
    x_ = tf.broadcast_to(x_,[1,128,128,16,3])
    tmp = x_[:,:,:,:,:]
    m = [tf.keras.metrics.Mean() for _ in range(5)]
    # start = time.time()
    # for _ in range(100):
    #     for index in range(5):
    #         m[index].reset_states()
    #     for index,y in enumerate(V(x_)):
    #         m[index](y)
    # print(time.time()-start)
    # for index in range(5):
    #     print(m[index].result().numpy())
    # start = time.time()
    # for _ in range(100):
    #     for index in range(5):
    #         m[index].reset_states()
    #     for index,y in enumerate(V(x_)):
    #         m[index](y)
    # print(time.time()-start)
    # for index in range(5):
    #     print(m[index].result().numpy())
    # print("******************V3********************")
    # V = Vgg16LayerBuf_V2(dtype=policy)
    # V.build(input_shape=None)
    # start = time.time()
    # for index in range(5):
    #     m[index].reset_states()
    # for _ in range(100):
    #     for index in range(5):
    #         m[index].reset_states()
    #     for i in range(16):
    #         x_ = x[:,:,:,i:i+1]
    #         for index,y in enumerate(V(x_)):
    #             m[index](y)
    # print(time.time()-start)
    # for index in range(5):
    #     print(m[index].result().numpy())
    # print("******************V2********************")
    #-----------------------------V3-V4比较------------------------#
    # import time
    # V = Vgg16LayerBuf_V3(dtype=policy)
    # V.build(input_shape=None)
    # # x = tf.random.uniform(shape=[1,128,128,16]) 
    # # x_ = tf.reshape(x,[1,128,128,16,1])
    # # x_ = tf.broadcast_to(x_,[1,128,128,16,3])
    # x_ = tmp[:,:,:,:,:]
    # m = [tf.keras.metrics.Mean() for _ in range(5)]
    # start = time.time()
    # for _ in range(100):
    #     for index in range(5):
    #         m[index].reset_states()
    #     for index,y in enumerate(V(x_)):
    #         m[index](y)
    # print(time.time()-start)
    # for index in range(5):
    #     print(m[index].result().numpy())
    # start = time.time()
    # for _ in range(100):
    #     for index in range(5):
    #         m[index].reset_states()
    #     for index,y in enumerate(V(x_)):
    #         m[index](y)
    # print(time.time()-start)
    # for index in range(5):
    #     print(m[index].result().numpy())
    # print("*******************V3*******************")
    V = Vgg16LayerBuf_V4(dtype=policy)
    V.build(input_shape=None)
    x_ = tf.transpose(x_,perm=[0,3,1,2,4])
    m = [tf.keras.metrics.Mean() for _ in range(5)]
    start = time.time()
    for _ in range(100):
        for index in range(5):
            m[index].reset_states()
        for index,y in enumerate(V(x_)):
            m[index](y)
    print(time.time()-start)
    for index in range(5):
        print(m[index].result().numpy())
    start = time.time()
    for _ in range(100):
        for index in range(5):
            m[index].reset_states()
        for index,y in enumerate(V(x_)):
            m[index](y)
    print("time",time.time()-start)
    for index in range(5):
        print(m[index].result().numpy())
    print("********************V4******************")
    # V = Vgg16LayerBuf_V2(dtype=policy)
    # V.build(input_shape=None)
    # start = time.time()
    # for index in range(5):
    #     m[index].reset_states()
    # x = tf.transpose(x,perm=[0,3,1,2])
    # for _ in range(100):
    #     for index in range(5):
    #         m[index].reset_states()
    #     for i in range(16):
    #         x_ = x[:,i:i+1,:,:]
    #         x_ = tf.transpose(x_,perm=[0,2,3,1])
    #         for index,y in enumerate(V(x_)):
    #             m[index](y)
    # print("time",time.time()-start)
    # for index in range(5):
    #     print(m[index].result().numpy())
    # print("*******************V2 transpose*******************")
    #----------------------------------------------------#
    # V = Vgg16LayerBuf_V2(dtype=policy)
    # V.build(input_shape=None)
    # start = time.time()
    # for index in range(5):
    #     m[index].reset_states()
    # for _ in range(100):
    #     for index in range(5):
    #         m[index].reset_states()
    #     for i in range(16):
    #         x_ = x[:,:,:,i:i+1]
    #         for index,y in enumerate(V(x_)):
    #             m[index](y)
    # print(time.time()-start)
    # for index in range(5):
    #     print(m[index].result().numpy())


    # V = Vgg16LayerBuf_V2(dtype=policy)
    # V.build(input_shape=None)
    # start = time.time()
    # for index in range(5):
    #     m[index].reset_states()
    # for _ in range(100):
    #     for index in range(5):
    #         m[index].reset_states()
    #     for i in range(3):
    #         x_ = x[:,:,:,i:i+1]
    #         for index,y in enumerate(V(x_)):
    #             m[index](y)
    # print(time.time()-start)
    # for index in range(5):
    #     print(m[index].result().numpy())

    # V = Vgg16LayerBuf_V2(dtype=policy)
    # V.build(input_shape=None)
    # start = time.time()
    # for index in range(5):
    #     m[index].reset_states()
    # for _ in range(100):
    #     for index in range(5):
    #         m[index].reset_states()
    #     for i in range(3):
    #         x_ = x[:,:,:,i:i+1]
    #         for index,y in enumerate(V(x_)):
    #             m[index](y)
    # print(time.time()-start)
    # for index in range(5):
    #     print(m[index].result().numpy())