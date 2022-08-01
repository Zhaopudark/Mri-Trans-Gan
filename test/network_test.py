from random import seed
import tensorflow as tf 
from models.networks.mri_trans_gan_v5 import Generator, Discriminator

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(1000)
    x = tf.random.normal(shape=[1,64,64,64,1])
    x_m = tf.random.normal(shape=[1,64,64,64,1])
    m = tf.random.normal(shape=[1,64,64,64,1])
    args = {}
    args['capacity_vector'] = 32
    args['up_sampling_method'] = 'up_conv'
    args['res_blocks_num'] = 9
    args['self_attention_G'] = None
    args['dimensions_type'] = '3D'
    args['self_attention_D'] = None
    args['spectral_normalization'] = False
    args['sn_iter_k'] = 1
    args['sn_clip_flag'] = True
    args['sn_clip_range'] = 128.0
    args['domain'] = [0.0,1.0]
    args['gan_loss_name'] = "WGAN-GP"
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    g = Generator(args,dtype=policy)
    d = Discriminator(args,dtype=policy)
    g.build(input_shape=[1,64,64,64,1])
    d.build(input_shape=[1,64,64,64,1])
    print(len(g.trainable_variables))
    for item in g.trainable_variables:
        print(tf.reduce_mean(item))
    input_ = [x,x_m,m]
    y = g(input_)
    print(y.shape,y.dtype)
    print(tf.reduce_mean(y))
    y = d(input_)
    print(y.shape,y.dtype)
    print(tf.reduce_mean(y))
    # Standalone usage:
    tf.keras.utils.set_random_seed(1000)
    initializer = tf.keras.initializers.LecunNormal(seed=10)
    values = initializer(shape=(2, 2))
    print(values)
    values = initializer(shape=(2, 2))
    print(values)
    values = initializer(shape=(2, 2))
    print(values)
    # tf.keras.utils.set_random_seed(1000)
    initializer = tf.keras.initializers.LecunNormal(seed=10)
    values = initializer(shape=(2, 2))
    print(values)
