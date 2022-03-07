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
tensor = tf.random.normal(shape=[1,16,128,128,64])
layer_out_index = 1
_feature_maps_indicators = ((1, 2, 4, 5, 7, 8, 9), ())
feature_maps_vectors = [[],[]]
def _dist_feature_maps(tensor,layer_out_index,feature_maps_vectors):
    """
    put a tensor to feature_maps_vectors
    layer_out_index indicates the tensor is output by which layer, corresponding layer index
    feature_maps_vectors is a 2D list to contain feature_maps
    since
        self._feature_maps_indicators is a 2D tuple of integer
        If layer_out_index in some row of self._feature_maps_indicators,
            it means the tensor is one of target feature map, should be recorded in feature_maps_vectors correspondingly.
    """
    for row_index in range(len(feature_maps_vectors)):
        if layer_out_index in _feature_maps_indicators[row_index]: # 
            tmp_row = feature_maps_vectors[row_index]
            tmp_row.append(tf.nn.relu(tensor))
_dist_feature_maps(tensor,layer_out_index,feature_maps_vectors)
print(feature_maps_vectors)