import tensorflow as tf 
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

base_model = tf.keras.applications.vgg19.VGG19(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)
inputs = tf.keras.Input(shape=(224, 224, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs, training=False)
print(x.shape,x.dtype)
base_model.summary()
base_model = tf.keras.applications.vgg16.VGG16(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)
inputs = tf.keras.Input(shape=(224, 224, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs, training=False)
print(x.shape,x.dtype)
base_model.summary()

import itertools
from typeguard import typechecked
from typing import Union,List
from collections.abc import Iterable

for feature_maps_indicators in [((1,2,4,5),),((1,2,4,5),()),((),(1,2,4,5)),((5,),(2,5,9,13)),((),(2,5,9,13))]:
    for feature_maps_indicator in feature_maps_indicators:
        buf = []
        for index in feature_maps_indicator:
            print(str(index))

