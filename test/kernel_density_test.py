import tensorflow as tf 
import numpy as np
from sklearn.neighbors import KernelDensity
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf2pi = tf.constant(2*np.pi,dtype=tf.float64)
def log_gauss_norm(h,d):
    return -0.5*d*tf.math.log(tf2pi)-d*tf.math.log(h)
def gauss(x,d,h):
    y = log_gauss_norm(h,d)-0.5*tf.reduce_sum(x**2,axis=-1)
    return tf.math.exp(y)
@tf.function
def my_kde(x,data_array,bandwidth=2.):
    n_features = tf.cast(float(data_array.shape[-1]),tf.float64)
    bandwidth = tf.cast(bandwidth,tf.float64)
    assert len(x.shape)==2
    x = x[:,tf.newaxis,:]
    y = gauss((x-data_array)/bandwidth,d=n_features,h=bandwidth)
    y = tf.reduce_mean(y,axis=-1)
    return tf.math.log(y)
# succeed
np.random.seed(0)
basic = np.array(np.random.normal(0,1.0,size=[10000,40]),dtype=np.float64)
kde = KernelDensity(kernel='gaussian',bandwidth=2).fit(basic)
y1 = my_kde(basic[0:2],basic)
tf.print(y1) 
y2 = kde.score_samples(basic[0:2])
print(y2)  
assert all(np.isclose(y1-y2,0.0)) 

# overflow
np.random.seed(0)
basic = np.array(np.random.uniform(-1.0,1.0,size=[10,800]),dtype=np.float64)
tested = np.array(np.random.uniform(-1.0,1.0,size=[10,800]),dtype=np.float64)
kde = KernelDensity(kernel='gaussian',bandwidth=2.1).fit(basic)
y1 = my_kde(basic[0:3],basic)
tf.print(y1) 
y2 = kde.score_samples(basic[0:3])
print(y2) 
y3 = my_kde(tested[0:3],basic)
print(y3) 
y4 = kde.score_samples(tested[0:3])
print(y4) 

# import time
# for _ in range(5):
#     start = time.perf_counter() 
#     for __ in range(10):
#         y = my_kde(basic[0:30],basic)
#     print(tf.reduce_mean(y))
#     print(time.perf_counter()-start)
# for _ in range(5):
#     start = time.perf_counter() 
#     for __ in range(10):
#         y = kde.score_samples(basic[0:30])
#     print(tf.reduce_mean(y))
#     print(time.perf_counter()-start)
# for _ in range(5):
#     start = time.perf_counter() 
#     for __ in range(10):
#         y = kde.score_samples(basic[0:30])
#     print(tf.reduce_mean(y))
#     print(time.perf_counter()-start)
# for _ in range(5):
#     start = time.perf_counter() 
#     for __ in range(10):
#         y = my_kde(basic[0:30],basic)
#     print(tf.reduce_mean(y))
#     print(time.perf_counter()-start)


 