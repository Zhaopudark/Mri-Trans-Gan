import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
tf.keras.utils.set_random_seed(1000)
tf.config.experimental.enable_op_determinism()

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

x = tf.Variable(tf.random.normal(shape=[128,28,28,1]))
model = MyModel()
optimizer = tf.keras.optimizers.Adam()
with tf.GradientTape() as tape:
    y = model(x) 

gradients = tape.gradient(y, model.trainable_variables)
print(len(gradients))
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
print(len(model.trainable_variables))