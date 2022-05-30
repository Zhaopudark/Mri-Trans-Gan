import tensorflow as tf 
import tensorflow.experimental.numpy as np
from sklearn.neighbors import KernelDensity
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def my_numpy_func(x):
    y = np.pad(x,[[2,3],[2,3]],'constant',constant_values=0)
    return y

def tf_function0(input):
    y = np.pad(input,[[2,3],[2,3]],'constant',constant_values=0)
    return y
def tf_function1(input):
    y = tf.numpy_function(my_numpy_func, [input], tf.int32)
    return y
@tf.function
def tf_function2(input):
    y = tf.numpy_function(my_numpy_func, [input], tf.int32)
    return y
@tf.function
def tf_function3(input):
    np.random.uniform(-30, 30)
    y = tf.pad(input,[[2,3],[2,3]],'constant',constant_values=0)
    return tf.reduce_mean(y)
x = tf.ones(shape=[1024,1024],dtype=tf.int32)
import time
for _ in range(5):
    start = time.perf_counter() 
    for __ in range(500):
        tf_function0(x)
    print(time.perf_counter()-start)
for _ in range(5):
    start = time.perf_counter() 
    for __ in range(500):
        tf_function1(x)
    print(time.perf_counter()-start)
for _ in range(5):
    start = time.perf_counter() 
    for __ in range(500):
        tf_function2(x)
    print(time.perf_counter()-start)
for _ in range(5):
    start = time.perf_counter() 
    for __ in range(500):
        tf_function3(x)
    print(time.perf_counter()-start)

# import scipy.ndimage as ndimage
# @tf.function
# def random_rotate_image(image):
#     image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
#     return image

# image = np.random.uniform(0.,1.,size=[1,128,128,1])
# y = random_rotate_image(image)
# print(y.shape)