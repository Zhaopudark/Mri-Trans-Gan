import tensorflow as tf 
import tensorflow_addons as tfa
import numpy as np 
from matplotlib import pyplot as plt
from tensorflow.keras import layers

physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()

class Generator(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""
    def __init__(self,input_dims:int,output_shape:list[int],**kwargs):
        super().__init__(**kwargs)
        self.dense_1 = layers.Dense(input_dims,activation=tf.nn.relu)
        self._output_shape = output_shape
        self.dense_2 = layers.Dense(tf.math.reduce_prod(output_shape),activation=tf.nn.sigmoid)
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return tf.reshape(x,shape=[-1]+self._output_shape)

class Discriminator(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""
    def __init__(self,input_dims:int,**kwargs):
        super().__init__(**kwargs)
        self.dense_1 = layers.Dense(input_dims,activation=tf.nn.relu)
        self.dense_2 = layers.Dense(3)
        self.dense_3 = tfa.layers.Maxout(num_units=1)
    def call(self, inputs):
        x = tf.reshape(inputs,[inputs.shape[0],-1])
        x = self.dense_1(x)
        x = self.dense_2(x)
        return tf.nn.sigmoid(self.dense_3(x))

g = Generator(128,output_shape=[28,28,1])
d = Discriminator(128)
lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_steps=1000,decay_rate=0.999,staircase=False,name=None)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

from sklearn.neighbors import KernelDensity
(train_images,train_labels), _ = tf.keras.datasets.mnist.load_data()
basic = train_images.reshape(train_images.shape[0], 784).astype('float32')/255
train_images = basic.reshape(basic.shape[0], 28, 28,1)
dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(128)
kde = KernelDensity(kernel='gaussian',bandwidth=0.5).fit(basic)

# log_density = kde.score_samples(basic[:128])
# print(tf.reduce_mean(log_density))
# plt.imshow(train_images[0, :, :], cmap='gray')
# plt.show()

@tf.function
def training_step(x):
    with tf.GradientTape(persistent=True) as tape:
        z = tf.random.uniform(shape=[x.shape[0],100],minval=-1.,maxval=1.,seed=0)
        real_x = x
        fake_x = g(z)
        real_y = d(real_x)
        fake_y = d(fake_x)
        d_loss = -tf.reduce_mean(tf.math.log(real_y+tf.keras.backend.epsilon())+tf.math.log(1.0-fake_y+tf.keras.backend.epsilon()))
        g_loss = -tf.reduce_mean(tf.math.log(fake_y+tf.keras.backend.epsilon()))
    d_optimizer.minimize(d_loss,d.trainable_weights,tape=tape)
    g_optimizer.minimize(g_loss,g.trainable_weights,tape=tape)
    return d_loss,g_loss
seed = tf.random.uniform([100, 100],-1.0,1.0,seed=0)
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input,training=False)
    gend = tf.reshape(predictions,[predictions.shape[0],784])
    print(gend.shape)
    log_density = kde.score_samples(gend)
    print(tf.reduce_mean(log_density))
    plt.figure(figsize=(10,10))
    for i in range(predictions.shape[0]):
        plt.subplot(10,10,i+1)
        plt.imshow(tf.reshape(predictions[i,:],shape=(28,28))*255.0, cmap='gray')
        plt.axis('off')
    plt.savefig('./SGAN/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
for epo in range(2000):
    for step, x in enumerate(dataset):
        d_loss,g_loss = training_step(x)
    if epo%100==0:
        print("epo:", epo, "d_loss:", d_loss," g_loss:",g_loss)
        generate_and_save_images(g,epo,seed)
        # Stop after 1000 steps.
        # Training the model to convergence is left
        # as an exercise to the reader.
        # if step >= 1000*10:
        #     break



