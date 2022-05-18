import sys
import os
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)
sys.path.append(os.path.join(base,'../'))
import tensorflow as tf
from models.networks.mri_trans_gan_v4 import Generator as Generator1
from models.networks.mri_trans_gan_v3_bak import Generator as Generator2
from models.networks.mri_trans_gan_v3 import Generator as Generator3
from models.networks.mri_trans_gan_v4 import Discriminator as Discriminator1
from models.networks.mri_trans_gan_v3_bak import Discriminator as Discriminator2
from models.networks.mri_trans_gan_v3 import Discriminator as Discriminator3
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.utils.set_random_seed(1000)
tf.config.experimental.enable_op_determinism()
tf.keras.utils.set_random_seed(1000)

x = tf.random.normal(shape=[1,16,128,128,1])
x_m = tf.random.normal(shape=[1,16,240,240,1])
m = tf.random.normal(shape=[1,16,128,128,1])
class tmp_args():
    def __init__(self) -> None:
        pass
args = tmp_args()
args.capacity_vector = 32
args.up_sampling_method = 'up_conv'
args.res_blocks_num = 9
args.self_attention_G = None
args.dimensions_type = '3D'
args.self_attention_D = None
args.spectral_normalization = False
args.sn_iter_k = 1
args.sn_clip_flag = True
args.sn_clip_range = 128.0
args.domain = [0.0,1.0]
args.gan_loss_name = "WGAN-GP"
policy = tf.keras.mixed_precision.Policy('mixed_float16')
g1 = Generator1(args,dtype=policy) 
d1 = Discriminator1(args,dtype=policy) 
tf.keras.utils.set_random_seed(1000)
g1.build(input_shape=[1,16,128,128,1])  
tf.keras.utils.set_random_seed(1000) 
d1.build(input_shape=[1,16,128,128,1]) 
tf.keras.utils.set_random_seed(1000)
tf.config.experimental.enable_op_determinism()
g2 = Generator2(args,dtype=policy) 
d2 = Discriminator2(args,dtype=policy) 
tf.keras.utils.set_random_seed(1000)
g2.build(input_shape=[1,16,128,128,1])  
tf.keras.utils.set_random_seed(1000) 
d2.build(input_shape=[1,16,128,128,1]) 

tf.keras.utils.set_random_seed(1000)
tf.config.experimental.enable_op_determinism()
g3 = Generator3(args,dtype=policy) 
d3 = Discriminator3(args,dtype=policy) 
tf.keras.utils.set_random_seed(1000)
g3.build(input_shape=[1,16,128,128,1])  
tf.keras.utils.set_random_seed(1000) 
d3.build(input_shape=[1,16,128,128,1]) 


assert len(g1.trainable_variables)==len(g2.trainable_variables)
assert len(g1.variables)==len(g2.variables)
for item1,item2 in zip(g1.trainable_variables,g2.trainable_variables):
    print(item1.name,tf.reduce_mean(item1-item2))
assert len(d1.trainable_variables)==len(d2.trainable_variables)
assert len(d1.variables)==len(d2.variables)
for item1,item2 in zip(d1.trainable_variables,d2.trainable_variables):
    print(item1.name,tf.reduce_mean(item1-item2))

input_ = [x,x_m,m]
y = g1(input_)
print(tf.reduce_mean(y),y.shape,y.dtype)
y = g2(input_)
print(tf.reduce_mean(y),y.shape,y.dtype)
y = g3(input_)
print(tf.reduce_mean(y),y.shape,y.dtype)
y = d1(input_)
print(tf.reduce_mean(y),y.shape,y.dtype)
y = d2(input_)
print(tf.reduce_mean(y),y.shape,y.dtype)
y = d3(input_)
print(tf.reduce_mean(y),y.shape,y.dtype)


