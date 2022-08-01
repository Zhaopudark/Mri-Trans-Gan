
import functools
import pytest
import numpy as np
import tensorflow as tf 
from training.metrics.psnr import PeakSignal2NoiseRatio2D,PeakSignal2NoiseRatio3D
from training.metrics.ssim import StructuralSimilarity2D

physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

    
@pytest.mark.parametrize('shape',
[[1,256],[7,256],
[1,256,3],[7,256,3],
[1,128,128,3],[7,128,128,3],
[1,64,64,64,3],[7,64,64,64,3]])
@pytest.mark.parametrize('dtype', ['float32','float64'])
@pytest.mark.parametrize('maxval', [1.0,3.0,10.1,255.0])
def test_mse(shape,dtype,maxval):
    tf.keras.utils.set_random_seed(0)
    # tf.config.experimental.enable_op_determinism()
    x = tf.random.uniform(shape=shape,dtype=dtype,maxval=maxval)
    y = tf.random.uniform(shape=shape,dtype=dtype,maxval=maxval)
    # mse1 =  tf.reduce_mean(tf.math.square(x-y),axis=list(range(1,len(shape))))
    # M = functools.reduce(lambda x,y:x*y,shape[1:],1)
    mse1 =  tf.reduce_mean(tf.math.square(tf.reshape(x-y,[shape[0],-1])),axis=-1)
    # mse2 =  tf.math.square(
    #             tf.norm(tf.reshape(x-y,[shape[0],-1]),
    #             ord=2,
    #             axis=-1)
    #             )/M
    mse3 = tf.keras.metrics.mean_squared_error(tf.reshape(x,[shape[0],-1]), tf.reshape(y,[shape[0],-1]))
    # print(mse1)
    # print(mse2)
    # _assert_allclose_according_to_type(tf.reduce_mean(mse1-mse2),0.0,atol=1e-3)
    _assert_allclose_according_to_type(tf.reduce_mean(mse1-mse3),0.0)

@pytest.mark.parametrize('shape', [[1],[2,3],[4,5,6],[7,8,9,10]])
def test_broadcast(shape):
    x1 = np.random.normal(size=shape)
    x2 = np.random.normal(size=shape[1:])
    # x2 = np.random.normal(size=shape[:-1]) # cannot broadcast
    assert (x1+x2).shape==x1.shape
    assert (x1-x2).shape==x1.shape
    assert (x1*x2).shape==x1.shape
    assert (x1/x2).shape==x1.shape


    x1 = tf.random.normal(shape=shape)
    x2 = tf.random.normal(shape=shape[1:])
    # x2 = tf.random.normal(shape=shape[:-1]) # cannot broadcast
    assert (x1+x2).shape==x1.shape
    assert (x1-x2).shape==x1.shape
    assert (x1*x2).shape==x1.shape
    assert (x1/x2).shape==x1.shape

    # tf.broadcast_to(tf.random.normal(shape=[3,5]),[3,5,7])
    tf.broadcast_to(tf.random.normal(shape=[3,5,1]),[3,5,7])
    tf.broadcast_to(tf.random.normal(shape=[5,7]),[3,5,7])
    sample_weight  = tf.random.normal(shape=shape,dtype=tf.float16)
    sample_weight = tf.convert_to_tensor(sample_weight)
    sample_weight = tf.cast(sample_weight,tf.float32)
    tf.debugging.assert_type(sample_weight,tf.float32)

    # 右端原则

@pytest.mark.parametrize('shape', [[3,128,128,7],[7,128,128,11],[7,128,128,3],[5,128,128,1]])
@pytest.mark.parametrize('maxval', [1.0,3.0,10.1,255.0])
def test_psnr2d(shape,maxval):
    y_ture = tf.random.uniform(shape=shape,maxval=maxval)
    y_pred = tf.random.uniform(shape=shape,maxval=maxval)
    out1 = tf.reduce_mean(tf.image.psnr(y_ture,y_pred,max_val=maxval))
    sample_weight = tf.ones(shape=shape[0])
    psnr = PeakSignal2NoiseRatio2D()
    psnr(y_ture,y_pred,maxval,sample_weight=sample_weight)
    out2 = psnr.result()
    psnr = PeakSignal2NoiseRatio3D() 
    
    psnr(tf.expand_dims(y_ture,-1),tf.expand_dims(y_pred,-1),maxval,sample_weight=sample_weight)
    out3 = psnr.result()

    _assert_allclose_according_to_type(out1-out2,0.0)
    _assert_allclose_according_to_type(out1-out3,0.0)


@pytest.mark.parametrize('shape', [[3,128,128,7],[7,128,128,11],[7,128,128,3],[5,128,128,1]])
@pytest.mark.parametrize('maxval', [1.0,3.0,10.1,255.0])
def test_ssim2d(shape,maxval):
    y_ture = tf.random.uniform(shape=shape,maxval=maxval)
    y_pred = tf.random.uniform(shape=shape,maxval=maxval)
    out1 = tf.reduce_mean(tf.image.ssim(y_ture,y_pred,max_val=maxval))
    sample_weight = tf.ones(shape=shape[0])
    ssim = StructuralSimilarity2D()
    ssim(y_ture,y_pred,maxval,sample_weight=sample_weight)
    out2 = ssim.result()
    psnr = PeakSignal2NoiseRatio3D() 
    psnr(tf.expand_dims(y_ture,-1),tf.expand_dims(y_pred,-1),maxval,sample_weight=sample_weight)
    out3 = psnr.result()

    _assert_allclose_according_to_type(out1-out2,0.0)
    # _assert_allclose_according_to_type(out1-out3,0.0)