"""
Transformer 相关的基本层
"""
import tensorflow as tf

from .activations import Activation
from .activations import Dropout
from .initializers import initializer_slect
from .convolutions import Conv2D
from .convolutions import Conv3D
from .denses import Dense
__all__ = [ 
    "MultiHeadSelfAttention",
    "PatchEmbedding2D",
    "Mlp",
    "StochasticDepth",
]
class MultiHeadSelfAttention(tf.keras.Model):
    def __init__(self,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 name=None,
                 dtype=None,):
        super(MultiHeadSelfAttention,self).__init__(name=name,dtype=dtype)
        
        self.inner_dtype = dtype

        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    def build(self,input_shape):
        if len(input_shape)==3:
            if input_shape[0] is None:
                B = 1 #batch_size
            else:
                B = int(input_shape[0])
            N = int(input_shape[1])
            C = int(input_shape[2])
            if (C%self.num_heads)!=0:
                raise ValueError("Channels can not be divisible by num_heads!")
            dim = C
            head_dim = dim//self.num_heads
            self.scale = head_dim ** -0.5
            self.re_shape_1 = [-1,N,3,self.num_heads,C//self.num_heads]
            self.re_shape_2 = [-1,N,C]
        else:
            raise ValueError("MultiHeadSelfAttention needs 3 dims for input tensor.")
        
        self.l1_qkv = Dense(kernel_size=C*3,use_bias=self.qkv_bias)
        self.l2_attn_drop = Dropout(rate=self.attn_drop)
        self.l3_proj = Dense(kernel_size=C)
        self.l4_proj_drop = Dropout(rate=self.proj_drop)

        flow_shape = self.l1_qkv.build(input_shape=input_shape)#[B,N,3C]
        flow_shape = [B,self.num_heads,N,N]
        flow_shape = self.l2_attn_drop.build(input_shape=flow_shape)# [B,self.num_heads,N,N]
        flow_shape = [B,N,C]
        flow_shape = self.l3_proj.build(input_shape=flow_shape)
        flow_shape = self.l4_proj_drop.build(input_shape=flow_shape)
        self.built = True
        output_shape = input_shape[:]
        return output_shape

    def call(self,x,training):
        x = self.l1_qkv(x,training=training) # [B,N,3C]
        qkv = tf.reshape(x,shape=self.re_shape_1) # [B,N,3,self.num_heads,C//self.num_heads]
        qkv = tf.transpose(qkv,perm=[2,0,3,1,4]) # [3,B,self.num_heads,N,C//self.num_heads]
        q,k,v = qkv[0],qkv[1],qkv[2] # [B,self.num_heads,N,C//self.num_heads]
        
        attn = q@tf.transpose(k,perm=[0,1,3,2]) # [B,self.num_heads,N,C//self.num_heads]*[B,self.num_heads,C//self.num_heads,N] = [B,self.num_heads,N,N]
        scaled_atten = attn*self.scale
        attn_map = tf.nn.softmax(scaled_atten,axis=-1) # [B,self.num_heads,N,N] 最后一维做softmax  以 最后两两维度来看 [N,N] 每一行代表了改点对于所有点的权值
        attn_map = self.l2_attn_drop(attn_map,training=training)

        x = attn_map@v # [B,self.num_heads,N,N]@[B,self.num_heads,N,C//self.num_heads]=[B,self.num_heads,N,C//self.num_heads]
        x = tf.transpose(x,perm=[0,2,1,3]) # [B,N,self.num_heads,C//self.num_heads]
        x = tf.reshape(x,shape=self.re_shape_2) # [B,N,C]
        x = self.l3_proj(x,training=training) # [B,N,C]
        x = self.l4_proj_drop(x,training=training)# [B,N,C]
        return x

class PatchEmbedding2D(tf.keras.Model):
    """ Image to Patch Embedding 2D
    X   [B,H,W,C]
    将图像在H,W尺度上切分为N个PxP的部分
    X_  [B,N,P,P,C] 其中 N*P*P=H*W
    reshape 对每个patch flatten
    X__ [B,N,P*P*C]
    再将每个flatten后的patch由P*P*C维度映射到 C_ 维度
    Y   [B,N,C_] 
    即对每个patch 分配一个 DenseLayer 共N个 DenseLayer 输入P*P*C 输出C_

    因为在Tensor层面,从[B,H,W,C]维度中拆分出patch并不方便
    实现时,采用卷积的等效方式进行
    
    对上述操作,相当于对[B,H,W,C]Tensor做了kernel_size=[P,P] strides=P 的卷积
    卷积矩阵为 [P,P,C,C_]
    卷积时 确实是对每个[P,P,C] 乘以 [P,P,C] 对输入的每个通道C分配[P,P]卷积核 
    重复 C_ 获得 C_ 层输出
    """
    def __init__(self,
                 patch_size=None,
                 embed_dim=None,
                 use_bias=False,
                 name=None,
                 dtype=None):
        super(PatchEmbedding2D,self).__init__(name=name,dtype=dtype)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.filters = self.embed_dim
        self.kernel_size = [self.patch_size,self.patch_size]
        self.strides = [self.patch_size,self.patch_size]
        self.padding = "VALID" 
        self.use_bias = use_bias

        self.patch_embedConv2D = Conv2D(filters=self.filters,kernel_size=self.kernel_size,strides=self.strides,padding=self.padding,use_bias=self.use_bias,dtype=dtype)
    def build(self,input_shape):
        if len(input_shape)==4:
            H = int(input_shape[1])
            W = int(input_shape[2])
            P = int(self.patch_size)
            if (H % P)!=0:
                raise ValueError("Input shape H can not be divisible by patch size!")    
            if (W % P)!=0:
                raise ValueError("Input shape W can not be divisible by patch size!")   
            self.num_patches = (H//P)*(W//P)
        else:
            raise ValueError("PatchEmbedding2D needs 4 dims for input tensor.")
        flow_shape = self.patch_embedConv2D.build(input_shape=input_shape)
        self.re_shape = [flow_shape[0],flow_shape[1]*flow_shape[2],flow_shape[3]]
        self.built = True
        output_shape = self.re_shape[:]
        return output_shape       
    def call(self,x,training):
        x = self.patch_embedConv2D(x)
        y = tf.reshape(x,shape=self.re_shape)
        return y

class Mlp(tf.keras.Model):
    def __init__(self,
                  hidden_features=None,
                  out_features=None,
                  activation="gelu",
                  drop=0.0,
                  name=None,
                  dtype=None):
        super(Mlp,self).__init__(name=name,dtype=dtype)
        self.l1_dense = Dense(hidden_features)
        self.l2_activation = Activation(activation)
        self.l3_drop_out = Dropout(rate=drop)
        self.l4_dense =  Dense(out_features)
        self.l5_drop_out = Dropout(rate=drop)
    def build(self,input_shape):
        flow_shap = self.l1_dense.build(input_shape=input_shape)
        flow_shap = self.l2_activation.build(input_shape=flow_shap)
        flow_shap = self.l3_drop_out.build(input_shape=flow_shap)
        flow_shap = self.l4_dense.build(input_shape=flow_shap)
        flow_shap = self.l5_drop_out.build(input_shape=flow_shap)
        self.built = True
        output_shape = flow_shap[:]
        return output_shape      
    def call(self,x,training):
        x = self.l1_dense(x)
        x = self.l2_activation(x)
        x = self.l3_drop_out(x)
        x = self.l4_dense(x)
        y = self.l5_drop_out(x)
        return y

class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self,survival_probability:float=0.5,name=None,dtype=None):
        super(StochasticDepth,self).__init__(name=name,dtype=dtype)
        self.survival_probability = survival_probability
    def build(self,input_shape):
        super(StochasticDepth,self).build(input_shape)
        output_shape = input_shape[:]
        return output_shape
    def call(self,x,training):
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError("input must be a list of length 2.")
        shortcut, residual = x
        # Random bernoulli variable indicating whether the branch should be kept or not or not
        b_l = tf.keras.backend.random_bernoulli([], p=self.survival_probability)

        def _call_train():
            return shortcut + b_l * residual

        def _call_test():
            return shortcut + self.survival_probability * residual

        return tf.keras.backend.in_train_phase(
            _call_train, _call_test, training=training
        )
if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = PatchEmbedding2D(patch_size=16,embed_dim=128)
    x  = tf.random.normal(shape=[128,128,128,3])
    y = model(x)
    print(y.shape,y.dtype)
    print(model.num_patches)

    model = MultiHeadSelfAttention(num_heads=8)
    x  = tf.random.normal(shape=[128,128,32])
    y = model(x)
    print(y.shape,y.dtype)

    model = Mlp(hidden_features=8,out_features=7)
    x  = tf.random.normal(shape=[128,128,32])
    y = model(x)
    print(y.shape,y.dtype)


    print(x)
    print(tf.reduce_mean(x))
    x = tf.ones(shape=[100])
    drop_path = StochasticDepth(survival_probability=0.5)
    y = drop_path([x,x],training=False)
    print(y)
    for _ in range(100):
        y = drop_path([x,x],training=True)
        print(y)




