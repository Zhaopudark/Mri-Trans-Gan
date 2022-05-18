import tensorflow as tf 
import numpy as np
import collections
import functools
from typeguard import typechecked
from typing import Iterable
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

optimizer =  tf.keras.optimizers.SGD(1.0)

w = [tf.Variable([0.0,0.1]),tf.Variable([0.0,0.2])]
x = 2.0
with tf.GradientTape(persistent=True) as tape:
    y = x*(w[0]+w[1])
tf.print(y)
grads = tape.gradient(y,w)
tf.print(grads)
optimizer.minimize(y,w,grad_loss=[tf.constant([1.,2.]),tf.constant([2.,2.])],tape=tape)
tf.print(w)
# optimizer.apply_gradients(zip(grads, vae.trainable_weights))
w = tf.Variable([1.0,0.0])
x = 2.0
with tf.GradientTape() as tape:
    y = x*w
optimizer.minimize(y,w,grad_loss=[tf.constant([2.0,5.0]),tf.constant([3.0,4.0])],tape=tape)
tf.print(w)

w = [tf.Variable([1.0,0.0]),tf.Variable([1.0,2.0,3.0])]
x = 2.0
with tf.GradientTape(persistent=True) as tape:
    y = x*(tf.reduce_mean(w[0])+tf.reduce_mean(w[1]))
gd = tape.gradient(y,w)
for item in gd:
    tf.print("gradient shape",item.shape)
optimizer.minimize(y,w,grad_loss=[[-2.0,-4.0],[-5.0,-6.0,-9.0]],tape=tape)
tf.print(w)

class Ones(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        return tf.ones(shape, dtype)*3.0
regularizer = tf.keras.regularizers.L1(l1=0.1)
constraint = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=0.001, rate=1.0, axis=0)
l1 = tf.keras.layers.Dense(2,kernel_initializer=Ones(),kernel_regularizer=None,activation=tf.nn.relu,kernel_constraint=constraint,activity_regularizer=regularizer)
l2 = tf.keras.layers.Dense(2,kernel_initializer=Ones(),kernel_regularizer=None,activation=tf.nn.relu)
x = tf.constant([[2.0,3.0]])
with tf.GradientTape(persistent=True) as tape:
    y1 = l1(x)
    tf.print("y1",y1)
    y2 = l2(y1)
    tf.print("y2",y2)
    y3 = 0.0*y2
gd = tape.gradient(y2,[l1.trainable_variables,l2.trainable_variables])
# optimizer.minimize(y,w,grad_loss=[[-2.0,-4.0],[-5.0,-6.0,-9.0]],tape=tape)
print("activity_regularizer",l1.activity_regularizer)
print("1-losses",l1.losses)
print("2-losses",l2.losses)
for item in gd:
    tf.print("gradient shape",item)

optimizer.minimize(y3,l2.trainable_variables,tape=tape)
print(l1.trainable_variables,l2.trainable_variables)
optimizer.minimize(y3,l1.trainable_variables,tape=tape)
print(l1.trainable_variables,l2.trainable_variables)

import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

if __name__ == "__main__":
    # np.random.seed(1)

    # # x = np.concatenate((np.random.normal(0, 1, int(0.3*100)), np.random.normal(5, 1, int(0.7*100))))[:, np.newaxis]
    # x = np.random.normal(0, 1, int(784))[:, np.newaxis]
    # plot_x = np.linspace(-5, 10, 1000)[:, np.newaxis]
    # # true_dens = 0.3*norm(0, 1).pdf(plot_x) + 0.7*norm(5, 1).pdf(plot_x)
    # true_dens = norm(0, 1).pdf(plot_x)
    # (train_images,train_labels), _ = tf.keras.datasets.mnist.load_data()
    # x = train_images = (train_images.reshape(train_images.shape[0], 784).astype('float32')/255)[0:1000,0:1]
    # log_dens = KernelDensity(bandwidth=0.5,).fit(x,plot_x)

    # plt.figure(),
    # plt.fill(plot_x, true_dens, fc='#AAAAFF', label='true_density')
    # plt.plot(plot_x, np.exp(log_dens), 'r', label='estimated_density')
    # for _ in range(x.shape[0]):
    #     plt.plot(x[:, 0], np.zeros(x.shape[0])-0.01, 'g*') 
    # plt.legend()

    # plt.show()

    from sklearn.neighbors import KernelDensity
    import numpy as np
    rng = np.random.RandomState(42)
    X = rng.random_sample((100, 3))
    X = np.random.normal(0,1,size=(100,3))
    (train_images,train_labels), _ = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 784).astype('float32')/255
    X1 = train_images[0:100,...]
    X2 = train_images[100:200,...]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    log_density = kde.score_samples(X[:3])
    print(log_density)
    log_density = kde.score(X)
    print(log_density)
        
    # plt.figure(),
    # plt.fill(plot_x, true_dens, fc='#AAAAFF', label='true_density')
    # plt.plot(plot_x, np.exp(log_dens), 'r', label='estimated_density')
    # for _ in range(x.shape[0]):
    #     plt.plot(x[:, 0], np.zeros(x.shape[0])-0.01, 'g*') 
    # plt.legend()

    # plt.show()
    