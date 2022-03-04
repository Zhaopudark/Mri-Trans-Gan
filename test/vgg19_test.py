import tensorflow as tf 
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

base_model = tf.keras.applications.vgg19.VGG19(
    include_top=True, weights='imagenet', input_tensor=None,
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
#------------------------------------------------------#
base_model = tf.keras.applications.vgg16.VGG16(
    include_top=True, weights='imagenet', input_tensor=None,
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
inputs = tf.random.normal(shape=[1,128, 128, 3])
inputs_ = tf.keras.applications.vgg16.preprocess_input(inputs)

print(tf.reduce_mean(inputs))
print(tf.reduce_mean(inputs_))
x = base_model.get_layer(name=None, index=1)(inputs,training=False)
print(tf.reduce_mean(x),x.shape,x.dtype)
x = base_model.get_layer(name=None, index=1)(inputs_,training=False)
print(tf.reduce_mean(x),x.shape,x.dtype)