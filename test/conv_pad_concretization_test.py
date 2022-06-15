
import pytest
import numpy as np
import tensorflow as tf 
from models.layers.convolutions import ConvPadConcretization
from models.layers.convolutions import UpSampalingConv1D
from models.layers.convolutions import UpSampalingConv2D
from models.layers.convolutions import UpSampalingConv3D
def _assert_allclose_according_to_type(
    a,
    b,
    rtol=1e-6,
    atol=1e-6,
    float_rtol=1e-6,
    float_atol=1e-6,
    half_rtol=1e-3,
    half_atol=1e-3,
    bfloat16_rtol=1e-2,
    bfloat16_atol=1e-2,
):
    """
    Similar to tf.test.TestCase.assertAllCloseAccordingToType()
    but this doesn't need a subclassing to run.
    """
    a = np.array(a)
    b = np.array(b)
    # types with lower tol are put later to overwrite previous ones.
    if (
        a.dtype == np.float32
        or b.dtype == np.float32
        or a.dtype == np.complex64
        or b.dtype == np.complex64
    ):
        rtol = max(rtol, float_rtol)
        atol = max(atol, float_atol)
    if a.dtype == np.float16 or b.dtype == np.float16:
        rtol = max(rtol, half_rtol)
        atol = max(atol, half_atol)
    if a.dtype == tf.bfloat16.as_numpy_dtype or b.dtype == tf.bfloat16.as_numpy_dtype:
        rtol = max(rtol, bfloat16_rtol)
        atol = max(atol, bfloat16_atol)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

@pytest.mark.parametrize('input_shape', [31,32,33])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('kernel_size',[1,2,3])
@pytest.mark.parametrize('stride',[1,2,3])
@pytest.mark.parametrize('padding', ['same', 'valid','causal'])
@pytest.mark.parametrize('dilation_rate',[1,2,3])
def test_convpadconcretization_behavior_1d(input_shape,dtype,kernel_size,stride,padding,dilation_rate):
    if dilation_rate>1 and stride!=1:
        pass
    else:
        dim = 1
        input_shape = [8]+[input_shape,]*dim+[3]
        tf.keras.utils.set_random_seed(1000)
        x = tf.random.normal(shape=input_shape,seed=1000,dtype=dtype)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        conv1d = tf.keras.layers.Conv1D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
        y = conv1d(x)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        conv1d_ = tf.keras.layers.Conv1D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
        conv1d_2 = ConvPadConcretization(conv1d_,padding_mode='constant',padding_constant=0)
        y_ = conv1d_2(x)

        assert y.shape==y_.shape
        computed = tf.reduce_mean(y-y_)
        _assert_allclose_according_to_type(computed,0.0)

@pytest.mark.parametrize('input_shape', [31,32,33])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('kernel_size',[1,2,3])
@pytest.mark.parametrize('stride',[1,2,3])
@pytest.mark.parametrize('padding', ['same', 'valid'])
@pytest.mark.parametrize('dilation_rate', [1,2,3])
def test_convpadconcretization_behavior_2d(input_shape,dtype,kernel_size,stride,padding,dilation_rate):
    if dilation_rate>1 and stride!=1:
        pass
    else:
        dim = 2
        input_shape = [8]+[input_shape,]*dim+[3]
        tf.keras.utils.set_random_seed(1000)
        x = tf.random.normal(shape=input_shape,seed=1000,dtype=dtype)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        conv2d = tf.keras.layers.Conv2D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
        y = conv2d(x)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        conv2d_ = tf.keras.layers.Conv2D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
        conv2d_2 = ConvPadConcretization(conv2d_,padding_mode='constant',padding_constant=0)
        y_ = conv2d_2(x)

        assert y.shape==y_.shape
        computed = tf.reduce_mean(y-y_)
        _assert_allclose_according_to_type(computed,0.0)

@pytest.mark.parametrize('input_shape', [31,32,33])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('kernel_size',[1,2,3])
@pytest.mark.parametrize('stride',[1,2,3])
@pytest.mark.parametrize('padding', ['same', 'valid'])
@pytest.mark.parametrize('dilation_rate', [1,2,3])
def test_convpadconcretization_behavior_3d(input_shape,dtype,kernel_size,stride,padding,dilation_rate):
    if dilation_rate>1 and stride!=1:
        pass
    else:
        dim = 3
        input_shape = [8]+[input_shape,]*dim+[3]
        tf.keras.utils.set_random_seed(1000)
        x = tf.random.normal(shape=input_shape,seed=1000,dtype=dtype)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        conv3d = tf.keras.layers.Conv3D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
        y = conv3d(x)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        conv3d_ = tf.keras.layers.Conv3D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
        conv3d_2 = ConvPadConcretization(conv3d_,padding_mode='constant',padding_constant=0)
        y_ = conv3d_2(x)

        assert y.shape==y_.shape
        computed = tf.reduce_mean(y-y_)
        _assert_allclose_according_to_type(computed,0.0)



@pytest.mark.parametrize('input_shape', [31,32,33])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('kernel_size',[1,2,3])
@pytest.mark.parametrize('stride',[1,2,3])
@pytest.mark.parametrize('padding', ['same', 'valid','causal'])
@pytest.mark.parametrize('dilation_rate',[1,2,3])
@pytest.mark.parametrize('padding_mode', ['constant', 'reflect', 'symmetric'])
@pytest.mark.parametrize('padding_constant', [0,1,2,3])
def test_convpadconcretization_behavior_1d_with_padding_mode(input_shape,dtype,kernel_size,stride,padding,dilation_rate,padding_mode,padding_constant):
    if dilation_rate>1 and stride!=1:
        pass
    elif padding == 'causal' and padding_mode!='constant':
        pass
    else:
        dim = 1
        input_shape = [8]+[input_shape,]*dim+[3]
        tf.keras.utils.set_random_seed(1000)
        x = tf.random.normal(shape=input_shape,seed=1000,dtype=dtype)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        conv1d = tf.keras.layers.Conv1D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
        y = conv1d(x)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        conv1d_ = tf.keras.layers.Conv1D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
        conv1d_2 = ConvPadConcretization(conv1d_,padding_mode=padding_mode,padding_constant=padding_constant)
        y_ = conv1d_2(x)
        assert y.shape==y_.shape
        config = conv1d_2.get_config()
        tf.keras.utils.set_random_seed(1000)
        conv1d_3 = ConvPadConcretization.from_config(config)
        y__ = conv1d_3(x)
        assert y.shape==y__.shape
        computed = tf.reduce_mean(y__-y_)
        _assert_allclose_according_to_type(computed,0.0)



@pytest.mark.parametrize('input_shape', [31,32,33])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('kernel_size',[1,2,3])
@pytest.mark.parametrize('stride',[1,2,3])
@pytest.mark.parametrize('padding', ['same', 'valid'])
@pytest.mark.parametrize('dilation_rate',[1,2,3])
@pytest.mark.parametrize('padding_mode', ['constant', 'reflect', 'symmetric'])
@pytest.mark.parametrize('padding_constant', [0,1,2,3])
def test_convpadconcretization_behavior_2d_with_padding_mode(input_shape,dtype,kernel_size,stride,padding,dilation_rate,padding_mode,padding_constant):
    if dilation_rate>1 and stride!=1:
        pass
    else:
        dim = 2
        input_shape = [8]+[input_shape,]*dim+[3]
        tf.keras.utils.set_random_seed(1000)
        x = tf.random.normal(shape=input_shape,seed=1000,dtype=dtype)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        conv2d = tf.keras.layers.Conv2D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
        y = conv2d(x)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        conv2d_ = tf.keras.layers.Conv2D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
        conv2d_2 = ConvPadConcretization(conv2d_,padding_mode=padding_mode,padding_constant=padding_constant)
        y_ = conv2d_2(x)
        assert y.shape==y_.shape
        config = conv2d_2.get_config()
        tf.keras.utils.set_random_seed(1000)
        conv2d_3 = ConvPadConcretization.from_config(config)
        y__ = conv2d_3(x)
        assert y.shape==y__.shape
        computed = tf.reduce_mean(y__-y_)
        _assert_allclose_according_to_type(computed,0.0)


@pytest.mark.parametrize('input_shape', [31,32,33])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('kernel_size',[1,2,3])
@pytest.mark.parametrize('stride',[1,2,3])
@pytest.mark.parametrize('padding', ['same', 'valid'])
@pytest.mark.parametrize('dilation_rate',[1,2,3])
@pytest.mark.parametrize('padding_mode', ['constant', 'reflect', 'symmetric'])
@pytest.mark.parametrize('padding_constant', [0,1,2,3])
def test_convpadconcretization_behavior_3d_with_padding_mode(input_shape,dtype,kernel_size,stride,padding,dilation_rate,padding_mode,padding_constant):
    if dilation_rate>1 and stride!=1:
        pass
    else:
        dim = 3
        input_shape = [8]+[input_shape,]*dim+[3]
        tf.keras.utils.set_random_seed(1000)
        x = tf.random.normal(shape=input_shape,seed=1000,dtype=dtype)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        conv3d = tf.keras.layers.Conv3D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
        y = conv3d(x)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        conv3d_ = tf.keras.layers.Conv3D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim, padding=padding,dtype=dtype,dilation_rate=dilation_rate,use_bias=False,kernel_initializer=kernel_initializer)
        conv3d_2 = ConvPadConcretization(conv3d_,padding_mode=padding_mode,padding_constant=padding_constant)
        y_ = conv3d_2(x)
        assert y.shape==y_.shape
        config = conv3d_2.get_config()
        tf.keras.utils.set_random_seed(1000)
        conv3d_3 = ConvPadConcretization.from_config(config)
        y__ = conv3d_3(x)
        assert y.shape==y__.shape
        computed = tf.reduce_mean(y__-y_)
        _assert_allclose_according_to_type(computed,0.0)

@pytest.mark.parametrize('input_shape', [31,32,33])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('kernel_size',[1,2,3])
@pytest.mark.parametrize('size',[1,2,3])
@pytest.mark.parametrize('interpolation',['nearest','bilinear'])
@pytest.mark.parametrize('stride',[1])
@pytest.mark.parametrize('dilation_rate',[1,2,3])
def test_upsampling_conv_1d(input_shape,dtype,kernel_size,size,interpolation,stride,dilation_rate):
    
    dim = 1
    if dim!=2 and interpolation!='nearest':
        pass
    else:
        input_shape = [8]+[input_shape,]*dim+[3]
        tf.keras.utils.set_random_seed(1000)
        x = tf.random.normal(shape=input_shape,seed=1000,dtype=dtype)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        upc = UpSampalingConv1D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim,size=(size,)*dim,interpolation=interpolation, padding='same',dtype=dtype,dilation_rate=dilation_rate,kernel_initializer=kernel_initializer)
        y = upc(x)
        assert int(y.shape[1])==int(x.shape[1])*size
        config = upc.get_config()
        upc_ = UpSampalingConv1D.from_config(config)
        tf.keras.utils.set_random_seed(1000)
        y_ = upc_(x)
        assert y.shape==y_.shape
        computed = tf.reduce_mean(y-y_)
        _assert_allclose_according_to_type(computed,0.0)

@pytest.mark.parametrize('input_shape', [31,32,33])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('kernel_size',[1,2,3])
@pytest.mark.parametrize('size',[1,2,3])
@pytest.mark.parametrize('interpolation',['nearest','bilinear'])
@pytest.mark.parametrize('stride',[1])
@pytest.mark.parametrize('dilation_rate',[1,2])
def test_upsampling_conv_2d(input_shape,dtype,kernel_size,size,interpolation,stride,dilation_rate):
    
    dim = 2
    if dim!=2 and interpolation!='nearest':
        pass
    else:
        input_shape = [8]+[input_shape,]*dim+[3]
        tf.keras.utils.set_random_seed(1000)
        x = tf.random.normal(shape=input_shape,seed=1000,dtype=dtype)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        upc = UpSampalingConv2D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim,size=(size,)*dim,interpolation=interpolation, padding='same',dtype=dtype,dilation_rate=dilation_rate,kernel_initializer=kernel_initializer)
        y = upc(x)
        assert int(y.shape[1])==int(x.shape[1])*size
        config = upc.get_config()
        upc_ = UpSampalingConv2D.from_config(config)
        tf.keras.utils.set_random_seed(1000)
        y_ = upc_(x)
        assert y.shape==y_.shape
        computed = tf.reduce_mean(y-y_)
        _assert_allclose_according_to_type(computed,0.0)
    
@pytest.mark.parametrize('input_shape', [31,32,33])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('kernel_size',[1,2,3])
@pytest.mark.parametrize('size',[1,2,3])
@pytest.mark.parametrize('interpolation',['nearest','bilinear'])
@pytest.mark.parametrize('stride',[1])
@pytest.mark.parametrize('dilation_rate',[1,2])
def test_upsampling_conv_3d(input_shape,dtype,kernel_size,size,interpolation,stride,dilation_rate):
    
    dim = 3
    if dim!=2 and interpolation!='nearest':
        pass
    else:
        input_shape = [8]+[input_shape,]*dim+[3]
        tf.keras.utils.set_random_seed(1000)
        x = tf.random.normal(shape=input_shape,seed=1000,dtype=dtype)

        tf.keras.utils.set_random_seed(1000)
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=1000)
        upc = UpSampalingConv3D(filters=8, kernel_size=[kernel_size,]*dim, strides=(stride,)*dim,size=(size,)*dim,interpolation=interpolation, padding='same',dtype=dtype,dilation_rate=dilation_rate,kernel_initializer=kernel_initializer)
        y = upc(x)
        assert int(y.shape[1])==int(x.shape[1])*size
        config = upc.get_config()
        upc_ = UpSampalingConv3D.from_config(config)
        tf.keras.utils.set_random_seed(1000)
        y_ = upc_(x)
        assert y.shape==y_.shape
        computed = tf.reduce_mean(y-y_)
        _assert_allclose_according_to_type(computed,0.0)