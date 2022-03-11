import pytest
import numpy as np
import tensorflow as tf 
from training.losses._image_losses import MeanVolumeGradientError
from training.losses._image_losses import MeanFeatureReconstructionError
from training.losses._image_losses import MeanStyleReconstructionError
from training.losses._image_losses import LossAcrossListWrapper
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


def mgd(x,y):
    if len(x.shape)==5:
        dz1,dy1,dx1 = pix_gradient_3D(x)
        dz2,dy2,dx2 = pix_gradient_3D(y)
        return tf.reduce_mean(tf.abs(dz1-dz2))/3 + tf.reduce_mean(tf.abs(dy1-dy2))/3 + tf.reduce_mean(tf.abs(dx1-dx2))/3
    elif len(x.shape)==4:
        dy1,dx1 = pix_gradient_2D(x)
        dy2,dx2 = pix_gradient_2D(y)
        return tf.reduce_mean(tf.abs(dy1-dy2))/2 + tf.reduce_mean(tf.abs(dx1-dx2))/2
    else:
        raise ValueError("mgd only support for 4 dims or 5 dims.")

def mgdl2(x,y):
    if len(x.shape)==5:
        dz1,dy1,dx1 = pix_gradient_3D(x)
        dz2,dy2,dx2 = pix_gradient_3D(y)
        return tf.reduce_mean(tf.square(dz1-dz2))/3 + tf.reduce_mean(tf.square(dy1-dy2))/3 + tf.reduce_mean(tf.square(dx1-dx2))/3
    elif len(x.shape)==4:
        dy1,dx1 = pix_gradient_2D(x)
        dy2,dx2 = pix_gradient_2D(y)
        return tf.reduce_mean(tf.square(dy1-dy2))/2 + tf.reduce_mean(tf.square(dx1-dx2))/2
    else:
        raise ValueError("mgd only support for 4 dims or 5 dims.")
def pix_gradient_2D(img): #shape=[b,h(y),w(x),c] 计算(x, y)点dx为[I(x+1,y)-I(x, y)] 末端pad 0 
    dx = img[:,:,1::,:]-img[:,:,0:-1,:]
    dy = img[:,1::,:,:]-img[:,0:-1,:,:]
    dx = tf.pad(dx,paddings=[[0,0],[0,0],[0,1],[0,0]]) # 末端pad 0
    dy = tf.pad(dy,paddings=[[0,0],[0,1],[0,0],[0,0]]) # 末端pad 0 
    return dy,dx
def pix_gradient_3D(img): #shape=[b,d(z),h(y),w(x),c] 计算(x,y,z)点dx为[I(x+1,y,z)-I(x,y,z)] 末端pad 0 
    dx = img[:,:,:,1::,:]-img[:,:,:,0:-1,:]
    dy = img[:,:,1::,:,:]-img[:,:,0:-1,:,:]
    dz = img[:,1::,:,:,:]-img[:,0:-1,:,:,:]
    dx = tf.pad(dx,paddings=[[0,0],[0,0],[0,0],[0,1],[0,0]]) # 末端pad 0
    dy = tf.pad(dy,paddings=[[0,0],[0,0],[0,1],[0,0],[0,0]]) # 末端pad 0 
    dz = tf.pad(dz,paddings=[[0,0],[0,1],[0,0],[0,0],[0,0]]) # 末端pad 0
    return dz,dy,dx   

def grma_2D(x):
    b,h,w,c = x.shape
    m = tf.reshape(x,[b,-1,c])
    m_T = tf.transpose(m,perm=[0,2,1])
    g = (1.0/(h*w*c))*tf.matmul(m_T,m)
    # tf.print(tf.reduce_mean(g))
    return g # [B,C,C]
def grma_3D(x):
    b,d,h,w,c = x.shape
    m = tf.reshape(x,[b,-1,c])
    m_T = tf.transpose(m,perm=[0,2,1])
    g = (1.0/(d*h*w*c))*tf.matmul(m_T,m)
    return g # [B,C,C]
def style_diff_2D(x,y):
    style_diff = tf.reduce_mean(tf.square(tf.norm(grma_2D(x)-grma_2D(y),ord="fro",axis=[1,2]))) # 在batch 维度取均值
    return style_diff
def style_diff_3D(x,y):
    style_diff = tf.reduce_mean(tf.square(tf.norm(grma_3D(x)-grma_3D(y),ord="fro",axis=[1,2]))) # 在batch 维度取均值
    return style_diff
    

@pytest.mark.parametrize("shape", [[2,3,4,5],[2,3,4,5,6],[2,3,4,5,6,7]])
@pytest.mark.parametrize("mode", ["L1","L2"])
@pytest.mark.parametrize("data_format", ["channels_last","channels_first"])
@pytest.mark.parametrize("reduction", [tf.keras.losses.Reduction.AUTO,tf.keras.losses.Reduction.NONE,tf.keras.losses.Reduction.SUM,tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE])
def test_MeanVolumeGradientError_accuracy(shape,mode,data_format,reduction):
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    loss = MeanVolumeGradientError(mode=mode,data_format=data_format,reduction=reduction)
    y_true = tf.random.normal(shape=shape)
    y_pred = tf.random.normal(shape=shape)
    # y = loss(y_true,y_pred)
    y = loss(y_true,y_pred,sample_weight=[1]*(len(shape)-2))
    if mode == "L2":
        _mgd = mgdl2
    else:
        _mgd = mgd 
    N_2 = len(shape)-2
    B = shape[0]
    if len(shape)<=5:
        if data_format == "channels_first":
            C = shape[1]
            if len(shape)==4:
                perm = [0,2,3,1]   
            else:
                perm = [0,2,3,4,1]
        else:
            C = shape[-1]
            perm = list(range(len(shape)))
        y_ = _mgd(tf.transpose(y_true,perm=perm),tf.transpose(y_pred,perm=perm))
        assert y.shape==[B,N_2,C]
        computed = tf.reduce_mean(y-y_)
        _assert_allclose_according_to_type(computed,0.0)
    else:
        if data_format == "channels_first":
            C = shape[1]
        else:
            C = shape[-1]
        assert y.shape==[B,N_2,C]


@pytest.mark.parametrize("reduction", [tf.keras.losses.Reduction.AUTO,tf.keras.losses.Reduction.NONE,tf.keras.losses.Reduction.SUM,tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE])
def test_LossAcrossListWrapper_accuracy(reduction):
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()

    feature_1 = tf.random.normal(shape=[7,4,5,6,3])
    feature_2 = tf.random.normal(shape=[7,4,5,6,4])
    feature_3 = tf.random.normal(shape=[7,4,5,6,5])
    feature_4 = tf.random.normal(shape=[7,4,5,6,3])
    feature_5 = tf.random.normal(shape=[7,4,5,6,4])
    feature_6 = tf.random.normal(shape=[7,4,5,6,5])
    y_true = [feature_1,feature_2,feature_3]
    y_pred = [feature_4,feature_5,feature_6]

    loss1 = MeanVolumeGradientError(reduction=reduction)
    _loss = MeanVolumeGradientError(reduction=reduction)
    loss2 = LossAcrossListWrapper(_loss)
    y1 = 1/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 1/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 1/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y2 = loss2(y_true,y_pred)
    computed = tf.reduce_mean(y1-y2)
    _assert_allclose_according_to_type(computed,0.0)
    y1 = 2/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 3/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 4/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y2 = loss2(y_true,y_pred,[2,3,4])
    computed = tf.reduce_mean(y1-y2)
    _assert_allclose_according_to_type(computed,0.0)

    loss1 = MeanFeatureReconstructionError(reduction=reduction)
    _loss = MeanFeatureReconstructionError(reduction=reduction)
    loss2 = LossAcrossListWrapper(_loss)
    y1 = 1/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 1/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 1/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y2 = loss2(y_true,y_pred)
    computed = tf.reduce_mean(y1-y2)
    _assert_allclose_according_to_type(computed,0.0)
    y1 = 2/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 3/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 4/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y2 = loss2(y_true,y_pred,[2,3,4])
    computed = tf.reduce_mean(y1-y2)
    _assert_allclose_according_to_type(computed,0.0)


    loss1 = MeanStyleReconstructionError(reduction=reduction)
    _loss = MeanStyleReconstructionError(reduction=reduction)
    loss2 = LossAcrossListWrapper(_loss)
    y1 = 1/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 1/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 1/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y2 = loss2(y_true,y_pred)
    computed = tf.reduce_mean(y1-y2)
    _assert_allclose_according_to_type(computed,0.0)
    y1 = 2/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 3/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 4/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y2 = loss2(y_true,y_pred,[2,3,4])
    computed = tf.reduce_mean(y1-y2)
    _assert_allclose_according_to_type(computed,0.0)

    loss1 = MeanVolumeGradientError(reduction=reduction)
    _loss = MeanVolumeGradientError(reduction=reduction)
    loss2 = LossAcrossListWrapper(_loss)
    loss3 = LossAcrossListWrapper.from_config(loss2.get_config(),custom_objects={"MeanVolumeGradientError":MeanVolumeGradientError})
    y1 = 1/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 1/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 1/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y3 = loss3(y_true,y_pred)
    computed = tf.reduce_mean(y1-y3)
    _assert_allclose_according_to_type(computed,0.0)
    y1 = 2/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 3/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 4/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y3 = loss3(y_true,y_pred,[2,3,4])
    computed = tf.reduce_mean(y1-y3)
    _assert_allclose_according_to_type(computed,0.0)

    loss1 = MeanFeatureReconstructionError(reduction=reduction)
    _loss = MeanFeatureReconstructionError(reduction=reduction)
    loss2 = LossAcrossListWrapper(_loss)
    loss3 = LossAcrossListWrapper.from_config(loss2.get_config(),custom_objects={"MeanFeatureReconstructionError":MeanFeatureReconstructionError})
    y1 = 1/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 1/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 1/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y3 = loss3(y_true,y_pred)
    computed = tf.reduce_mean(y1-y3)
    _assert_allclose_according_to_type(computed,0.0)
    y1 = 2/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 3/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 4/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y3 = loss3(y_true,y_pred,[2,3,4])
    computed = tf.reduce_mean(y1-y3)
    _assert_allclose_according_to_type(computed,0.0)

    loss1 = MeanStyleReconstructionError(reduction=reduction)
    _loss = MeanStyleReconstructionError(reduction=reduction)
    loss2 = LossAcrossListWrapper(_loss)
    loss3 = LossAcrossListWrapper.from_config(loss2.get_config(),custom_objects={"MeanStyleReconstructionError":MeanStyleReconstructionError})
    y1 = 1/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 1/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 1/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y3 = loss3(y_true,y_pred)
    computed = tf.reduce_mean(y1-y3)
    _assert_allclose_according_to_type(computed,0.0)
    y1 = 2/3*tf.reduce_mean(loss1(feature_1,feature_4))+ 3/3*tf.reduce_mean(loss1(feature_2,feature_5))+ 4/3*tf.reduce_mean(loss1(feature_3,feature_6))
    y3 = loss3(y_true,y_pred,[2,3,4])
    computed = tf.reduce_mean(y1-y3)
    _assert_allclose_according_to_type(computed,0.0)


 


@pytest.mark.parametrize("shape", [[2,3,4,5],[2,3,4,5,6],[2,3,4,5,6,7]])
@pytest.mark.parametrize("mode", ["L1","L2"])
@pytest.mark.parametrize("data_format", ["channels_last","channels_first"])
@pytest.mark.parametrize("reduction", [tf.keras.losses.Reduction.AUTO,tf.keras.losses.Reduction.NONE,tf.keras.losses.Reduction.SUM,tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE])
def test_MeanVolumeGradientError_sample_weight(shape,mode,data_format,reduction):
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    loss = MeanVolumeGradientError(mode=mode,data_format=data_format,reduction=reduction)
    y_true = tf.random.normal(shape=shape)
    y_pred = tf.random.normal(shape=shape)
    
    N_2 = len(shape)-2
    B = shape[0]
    if data_format == "channels_first":
        C = shape[1]
    else:
        C = shape[-1]
    shape_list = [  [],
                    [C],[1],[N_2],
                    [N_2,C],[N_2,1],[1,C],[1,1],
                    [B,N_2,C],[B,N_2,1],[B,1,C],[1,N_2,C],[B,1,1],[1,N_2,1],[1,1,C],[1,1,1],
                ]
    for shape in shape_list:
        sample_weight = tf.ones(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.zeros(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.random.normal(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.random.uniform(shape=shape)
        loss(y_true,y_pred,sample_weight)

@pytest.mark.parametrize("shape", [[2,3,4,5],[2,3,4,5,6],[2,3,4,5,6,7]])
@pytest.mark.parametrize("mode", ["L1","L2"])
@pytest.mark.parametrize("data_format", ["channels_last","channels_first"])
@pytest.mark.parametrize("reduction", [tf.keras.losses.Reduction.AUTO,tf.keras.losses.Reduction.NONE,tf.keras.losses.Reduction.SUM,tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE])
def test_MeanVolumeGradientError_from_config(shape,mode,data_format,reduction):
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    loss = MeanVolumeGradientError(mode=mode,data_format=data_format,reduction=reduction)
    y_true = tf.random.normal(shape=shape)
    y_pred = tf.random.normal(shape=shape)
    loss_ = MeanVolumeGradientError.from_config(loss.get_config())
    y = loss(y_true,y_pred)
    y_ = loss_(y_true,y_pred)
    assert y.shape==y_.shape
    computed = tf.reduce_mean(y-y_)
    _assert_allclose_according_to_type(computed,0.0)
  
         
    
#--------------------------------------------------------------------#
@pytest.mark.parametrize("mode", ["L1","L2"])
@pytest.mark.parametrize("reduction", [tf.keras.losses.Reduction.AUTO,tf.keras.losses.Reduction.NONE,tf.keras.losses.Reduction.SUM,tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE])
def test_MeanFeatureReconstructionError_accuracy(mode,reduction):
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    loss = MeanFeatureReconstructionError(mode=mode,reduction=reduction)
    y_true = tf.random.normal(shape=[2,8,8,3])
    y_pred = tf.random.normal(shape=[2,8,8,3])
    B = 2 
    y = loss(y_true,y_pred,sample_weight=[1,1])
    if mode == "L1":
        y_ = tf.reduce_mean(tf.abs(y_true-y_pred))
    else:
        y_ = tf.reduce_mean(tf.square(y_true-y_pred))
    assert y.shape==[B]
    computed = tf.reduce_mean(y-y_)
    _assert_allclose_according_to_type(computed,0.0)
  

@pytest.mark.parametrize("mode", ["L1","L2"])
@pytest.mark.parametrize("reduction", [tf.keras.losses.Reduction.AUTO,tf.keras.losses.Reduction.NONE,tf.keras.losses.Reduction.SUM,tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE])
def test_MeanFeatureReconstructionError_sample_weight(mode,reduction):
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    loss = MeanFeatureReconstructionError(mode=mode,reduction=reduction)
    y_true = tf.random.normal(shape=[2,8,8,3])
    y_pred = tf.random.normal(shape=[2,8,8,3])
    B = 2 
    shape_list = [[],
                  [B],[1],
                ]
    for shape in shape_list:
        sample_weight = tf.ones(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.zeros(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.random.normal(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.random.uniform(shape=shape)
        loss(y_true,y_pred,sample_weight)
    y_true = tf.random.normal(shape=[7,8,8,3])
    y_pred = tf.random.normal(shape=[7,8,8,3])
    B = 7
    shape_list = [[],
                  [B],[1],
                ]
    for shape in shape_list:
        sample_weight = tf.ones(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.zeros(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.random.normal(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.random.uniform(shape=shape)
        loss(y_true,y_pred,sample_weight)

@pytest.mark.parametrize("mode", ["L1","L2"])
@pytest.mark.parametrize("reduction", [tf.keras.losses.Reduction.AUTO,tf.keras.losses.Reduction.NONE,tf.keras.losses.Reduction.SUM,tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE])
def test_MeanFeatureReconstructionError_from_config(mode,reduction):
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    loss = MeanFeatureReconstructionError(mode=mode,reduction=reduction)
    loss_ = MeanFeatureReconstructionError.from_config(loss.get_config())

    y_true = tf.random.normal(shape=[2,8,8,3])
    y_pred = tf.random.normal(shape=[2,8,8,3])
    y = loss(y_true,y_pred)
    y_ = loss_(y_true,y_pred)
    assert y.shape==y_.shape
    computed = tf.reduce_mean(y-y_)
    _assert_allclose_according_to_type(computed,0.0)
    y_true = tf.random.normal(shape=[7,8,8,3])
    y_pred = tf.random.normal(shape=[7,8,8,3])
    y = loss(y_true,y_pred)
    y_ = loss_(y_true,y_pred)
    assert y.shape==y_.shape
    computed = tf.reduce_mean(y-y_)
    _assert_allclose_according_to_type(computed,0.0)

#--------------------------------------------------------------------#
@pytest.mark.parametrize("data_format", ["channels_last","channels_first"])
@pytest.mark.parametrize("reduction", [tf.keras.losses.Reduction.AUTO,tf.keras.losses.Reduction.NONE,tf.keras.losses.Reduction.SUM,tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE])
def test_MeanStyleReconstructionError_accuracy(data_format,reduction):
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    loss = MeanStyleReconstructionError(data_format=data_format,reduction=reduction)
    y_true = tf.random.normal(shape=[2,40,40,4])
    y_pred = tf.random.normal(shape=[2,40,40,4])
    B = 2 
    y = loss(y_true,y_pred,sample_weight=[1,1])
    if data_format == "channels_first":
        perm = [0,2,3,1]
        y_ = tf.reduce_mean(style_diff_2D(tf.transpose(y_true,perm=perm),tf.transpose(y_pred,perm=perm)))
    else:
        y_ = tf.reduce_mean(style_diff_2D(y_true,y_pred))
    
    assert y.shape==[B]
    computed = tf.reduce_mean(y-y_)
    _assert_allclose_according_to_type(computed,0.0)

    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    y_true = tf.random.normal(shape=[2,16,16,16,3])
    y_pred = tf.random.normal(shape=[2,16,16,16,3])
    B = 2 
    y = loss(y_true,y_pred,sample_weight=[1,1])
    if data_format == "channels_first":
        perm = [0,2,3,4,1]
        y_ = tf.reduce_mean(style_diff_3D(tf.transpose(y_true,perm=perm),tf.transpose(y_pred,perm=perm)))
    else:
        y_ = tf.reduce_mean(style_diff_3D(y_true,y_pred))
   
    assert y.shape==[B]
    computed = tf.reduce_mean(y-y_)
    _assert_allclose_according_to_type(computed,0.0)
  

@pytest.mark.parametrize("data_format", ["channels_last","channels_first"])
@pytest.mark.parametrize("reduction", [tf.keras.losses.Reduction.AUTO,tf.keras.losses.Reduction.NONE,tf.keras.losses.Reduction.SUM,tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE])
def test_MeanStyleReconstructionError_sample_weight(data_format,reduction):
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    loss = MeanStyleReconstructionError(data_format=data_format,reduction=reduction)
    y_true = tf.random.normal(shape=[2,8,8,3])
    y_pred = tf.random.normal(shape=[2,8,8,3])
    B = 2 
    shape_list = [[],
                  [B],[1],
                ]
    for shape in shape_list:
        sample_weight = tf.ones(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.zeros(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.random.normal(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.random.uniform(shape=shape)
        loss(y_true,y_pred,sample_weight)

    y_true = tf.random.normal(shape=[7,8,8,8,3])
    y_pred = tf.random.normal(shape=[7,8,8,8,3])
    B = 7
    shape_list = [[],
                  [B],[1],
                ]
    for shape in shape_list:
        sample_weight = tf.ones(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.zeros(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.random.normal(shape=shape)
        loss(y_true,y_pred,sample_weight)
        sample_weight = tf.random.uniform(shape=shape)
        loss(y_true,y_pred,sample_weight)

@pytest.mark.parametrize("data_format", ["channels_last","channels_first"])
@pytest.mark.parametrize("reduction", [tf.keras.losses.Reduction.AUTO,tf.keras.losses.Reduction.NONE,tf.keras.losses.Reduction.SUM,tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE])
def test_MeanStyleReconstructionError_from_config(data_format,reduction):
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    loss = MeanStyleReconstructionError(data_format=data_format,reduction=reduction)
    
    loss_ = MeanStyleReconstructionError.from_config(loss.get_config())
    y_true = tf.random.normal(shape=[2,8,8,3])
    y_pred = tf.random.normal(shape=[2,8,8,3])
    y = loss(y_true,y_pred)
    y_ = loss_(y_true,y_pred)
    assert y.shape==y_.shape
    computed = tf.reduce_mean(y-y_)
    _assert_allclose_according_to_type(computed,0.0)

    y_true = tf.random.normal(shape=[7,8,8,8,3])
    y_pred = tf.random.normal(shape=[7,8,8,8,3])
    y = loss(y_true,y_pred)
    y_ = loss_(y_true,y_pred)
    assert y.shape==y_.shape
    computed = tf.reduce_mean(y-y_)
    _assert_allclose_according_to_type(computed,0.0)