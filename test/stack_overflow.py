import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from datetime import datetime

class Conv2D(tf.keras.layers.Conv2D):
    def add_weight(self,*args,**kwargs):
        tf.print("hai hai hai")
        return super().add_weight(*args,**kwargs)

# Define a layer with an eager side effect
class EagerLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EagerLayer, self).__init__(**kwargs)
        # Do some kind of initialization here
        # self.dense = tf.keras.layers.Dense(32)
        self.conv = Conv2D(32,kernel_size=[3,3],strides=[1,1],use_bias=False)

    def build(self,input_shape):
        # self.conv.build(input_shape)
        super().build(input_shape)
 
    def call(self, inputs):
        print("\nCurrently running eagerly", str(datetime.now()))
        return self.conv(inputs)

input_data = tf.random.uniform([60, 28, 28,3])
layer = EagerLayer()
layer.build(input_shape=[60, 28, 28,3])
# _ = layer(input_data)
tf_func_layer = tf.function(layer)
print("=============")
print(tf.executing_eagerly(),layer.built,layer._autocast)
# layer._maybe_build(inputs=input_data)
layer._activity_regularizer = False
layer._supports_masking = False
layer._saved_model_inputs_spec = 1
_ = tf_func_layer(input_data)
# print(layer.variables)
# inputs, args, kwargs = layer._split_out_first_arg(args=[input_data], kwargs={})
# print(inputs.shape, args, kwargs)
# input_list = tf.nest.flatten(inputs)
# print(any(tf.keras.backend.is_keras_tensor(tensor) for tensor in tf.nest.flatten([inputs, args, kwargs])))
# def _convert_numpy_or_python_types(x):
#   if isinstance(x, (tf.Tensor, np.ndarray, float, int)):
#     return tf.convert_to_tensor(x)
#   return x
# if any(isinstance(x, (tf.Tensor, np.ndarray, float, int)) for x in input_list):
#     inputs = tf.nest.map_structure(_convert_numpy_or_python_types, inputs)
#     input_list = tf.nest.flatten(inputs)
print("=============")
conv = Conv2D(32,kernel_size=[3,3],strides=[1,1],use_bias=False)
@tf.function
def test_func(inputs):
    print("\nCurrently running eagerly1", str(datetime.now()))
    tf.print("\nCurrently running eagerly", str(datetime.now()))
    # conv.add_weight(shape=(),name="www")
    y =  conv(inputs)
    print("\nCurrently running eagerly2", str(datetime.now()))
    print("\nCurrently running eagerly3", str(datetime.now()))
    print("\nCurrently running eagerly4", str(datetime.now()))
    return y
    # return inputs
_ = test_func(input_data)