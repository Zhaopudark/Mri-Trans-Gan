import pytest
import numpy as np
import tensorflow as tf 
from models.blocks.vgg import FeatureMapsGetter
from models.blocks.vgg import PerceptualLossExtractor

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
        dz1,dy1,dx1 = pix_gradient_3d(x)
        dz2,dy2,dx2 = pix_gradient_3d(y)
        return tf.reduce_mean(tf.abs(dz1-dz2))/3 + tf.reduce_mean(tf.abs(dy1-dy2))/3 + tf.reduce_mean(tf.abs(dx1-dx2))/3
    elif len(x.shape)==4:
        dy1,dx1 = pix_gradient_2d(x)
        dy2,dx2 = pix_gradient_2d(y)
        return tf.reduce_mean(tf.abs(dy1-dy2))/2 + tf.reduce_mean(tf.abs(dx1-dx2))/2
    else:
        raise ValueError("mgd only support for 4 dims or 5 dims.")

def mgdl2(x,y):
    if len(x.shape)==5:
        dz1,dy1,dx1 = pix_gradient_3d(x)
        dz2,dy2,dx2 = pix_gradient_3d(y)
        return tf.reduce_mean(tf.square(dz1-dz2))/3 + tf.reduce_mean(tf.square(dy1-dy2))/3 + tf.reduce_mean(tf.square(dx1-dx2))/3
    elif len(x.shape)==4:
        dy1,dx1 = pix_gradient_2d(x)
        dy2,dx2 = pix_gradient_2d(y)
        return tf.reduce_mean(tf.square(dy1-dy2))/2 + tf.reduce_mean(tf.square(dx1-dx2))/2
    else:
        raise ValueError("mgd only support for 4 dims or 5 dims.")
def pix_gradient_2d(img): #shape=[b,h(y),w(x),c] 计算(x, y)点dx为[I(x+1,y)-I(x, y)] 末端pad 0 
    dx = img[:,:,1::,:]-img[:,:,0:-1,:]
    dy = img[:,1::,:,:]-img[:,0:-1,:,:]
    dx = tf.pad(dx,paddings=[[0,0],[0,0],[0,1],[0,0]]) # 末端pad 0
    dy = tf.pad(dy,paddings=[[0,0],[0,1],[0,0],[0,0]]) # 末端pad 0 
    return dy,dx
def pix_gradient_3d(img): #shape=[b,d(z),h(y),w(x),c] 计算(x,y,z)点dx为[I(x+1,y,z)-I(x,y,z)] 末端pad 0 
    dx = img[:,:,:,1::,:]-img[:,:,:,0:-1,:]
    dy = img[:,:,1::,:,:]-img[:,:,0:-1,:,:]
    dz = img[:,1::,:,:,:]-img[:,0:-1,:,:,:]
    dx = tf.pad(dx,paddings=[[0,0],[0,0],[0,0],[0,1],[0,0]]) # 末端pad 0
    dy = tf.pad(dy,paddings=[[0,0],[0,0],[0,1],[0,0],[0,0]]) # 末端pad 0 
    dz = tf.pad(dz,paddings=[[0,0],[0,1],[0,0],[0,0],[0,0]]) # 末端pad 0
    return dz,dy,dx   

def grma_2d(x):
    b,h,w,c = x.shape
    m = tf.reshape(x,[b,-1,c])
    m_T = tf.transpose(m,perm=[0,2,1])
    g = (1.0/(h*w*c))*tf.matmul(m_T,m)
    # tf.print(tf.reduce_mean(g))
    return g # [B,C,C]
def grma_3d(x):
    b,d,h,w,c = x.shape
    m = tf.reshape(x,[b,-1,c])
    m_T = tf.transpose(m,perm=[0,2,1])
    g = (1.0/(d*h*w*c))*tf.matmul(m_T,m)
    return g # [B,C,C]
def style_diff_2d(x,y):
    style_diff = tf.reduce_mean(tf.square(tf.norm(grma_2d(x)-grma_2d(y),ord='fro',axis=[1,2]))) # 在batch 维度取均值
    return style_diff
def style_diff_3d(x,y):
    style_diff = tf.reduce_mean(tf.square(tf.norm(grma_3d(x)-grma_3d(y),ord='fro',axis=[1,2]))) # 在batch 维度取均值
    return style_diff
    

@pytest.mark.parametrize('model_name', ['vgg16','vgg19'])
@pytest.mark.parametrize('data_format', ['channels_first','channels_last'])
@pytest.mark.parametrize('use_pooling', [True,False])
@pytest.mark.parametrize('feature_maps_indicators', [((1,2,4,5),),((1,2,4,5),()),((),(1,2,4,5)),((5,),(2,5,9,13)),((),(2,5,9,13))])
@pytest.mark.parametrize('dtype', ['float16','mixed_float16','float32','float64'])
def test_FeatureMapsGetter_accuracy(model_name,data_format,use_pooling,feature_maps_indicators,dtype):
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    x = tf.random.uniform(shape=[1,128,128,16]) 
    x = tf.reshape(x,[1,128,128,16,1])
    x = tf.broadcast_to(x,[1,128,128,16,3])
    x = tf.transpose(x,perm=[0,3,1,2,4]) # [1,16,128,128,3]

    if data_format == 'channels_first':
        raw_x = tf.transpose(x,perm=[0,4,1,2,3]) # [1,3,16,128,128]
    else:
        raw_x = x
       
    x = x # do not need to normalize_input_data_format
    x = tf.reshape(x,[16,128,128,3])
    x = x # do not need to broadcast_to
  
    if model_name == 'vgg16':
        model =  tf.keras.applications.vgg16.VGG16(
                include_top=False, weights='imagenet', input_tensor=None,
                input_shape=None, pooling=None, classes=1000,
                classifier_activation='softmax'
                )
        pooling_layer_indexes = [3,6,10,14,18]
        x = tf.keras.applications.vgg16.preprocess_input(x)
    else:
        model =  tf.keras.applications.vgg19.VGG19(
                include_top=False, weights='imagenet', input_tensor=None,
                input_shape=None, pooling=None, classes=1000,
                classifier_activation='softmax'
                )
        pooling_layer_indexes = [3,6,11,16,21]
        x = tf.keras.applications.vgg19.preprocess_input(x)
    
    valid_indexes = list(range(len(model.layers)))
    valid_indexes.remove(0)
    indexes_set = set()
    for indicator in feature_maps_indicators:
        for index in indicator:
            indexes_set.add(index)
    indexes_set = sorted(indexes_set)
    max_index = max(indexes_set)
    if max_index >6:
        use_pooling = True

    if use_pooling:
        pass
    else:
        for index in pooling_layer_indexes:
            if index in valid_indexes:
                valid_indexes.remove(index)

    model_outbuf = {}
    y = x
    for index in valid_indexes:
        x = model.get_layer(index=index)(y)
        y = x
        model_outbuf[str(index)] = y 
    result_buf = []
    for feature_maps_indicator in feature_maps_indicators:
        buf = []
        for index in feature_maps_indicator:
            buf.append(model_outbuf[str(index)])
        result_buf.append(buf)

    feature_maps_getter = FeatureMapsGetter(
        model_name=model_name,
        data_format=data_format,
        use_pooling=use_pooling,
        feature_maps_indicators=feature_maps_indicators,
        dtype=dtype)
    result_buf_2 = feature_maps_getter(raw_x)
    for result_list,result_list2 in zip(result_buf,result_buf_2):
        for result,result2 in zip(result_list,result_list2):
            assert  result.shape==result2.shape
            computed = tf.reduce_mean(result-result2)
            _assert_allclose_according_to_type(computed,0.0)

@pytest.mark.parametrize('model_name', ['vgg16','vgg19'])
@pytest.mark.parametrize('data_format', ['channels_first','channels_last'])
@pytest.mark.parametrize('transform_high_dimension',[True,False])
@pytest.mark.parametrize('use_pooling', [True,False])
@pytest.mark.parametrize('use_feature_reco_loss', [True,False])
@pytest.mark.parametrize('use_style_reco_loss', [True,False])
@pytest.mark.parametrize('feature_reco_something', [[[5,],[1]],[[1,2,4,5],[1,0.4,0.5,0.3]]])
@pytest.mark.parametrize('style_reco_something', [[[2,5,9,13],[0.3,0.4,0.5,0.6]],[[2,5,15],[0.3,0.4,0.5]]])
@pytest.mark.parametrize('dtype', ['float16','mixed_float16','float32','float64'])
def test_FeatureMapsGetter_accuracy(model_name,
    data_format,transform_high_dimension,
    use_pooling,use_feature_reco_loss,use_style_reco_loss,feature_reco_something,
    style_reco_something,dtype):
    
    feature_reco_index,_feature_reco_sample_weight = feature_reco_something
    style_reco_index,_style_reco_sample_weight = style_reco_something

    if any([use_feature_reco_loss,use_style_reco_loss]):
        feature_maps_indicators=(tuple(feature_reco_index) if use_feature_reco_loss else tuple([]),tuple(style_reco_index) if use_style_reco_loss else tuple([]))
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()
        x_true = _x_true = tf.random.uniform(shape=[1,16,128,128,1])
        x_pred = _x_pred = tf.random.uniform(shape=[1,16,128,128,1]) 
    
        if data_format == 'channels_first':
            x_true = tf.transpose(x_true,perm=[0,4,1,2,3]) # make channels_first input
            x_pred = tf.transpose(x_pred,perm=[0,4,1,2,3]) # make channels_first input
            _x_true = tf.transpose(_x_true,perm=[0,4,1,2,3]) # make channels_first input
            _x_pred = tf.transpose(_x_pred,perm=[0,4,1,2,3]) # make channels_first input
            if transform_high_dimension:
                x_true1 = tf.transpose(x_true,perm=[0,1,4,2,3])
                x_true2 = tf.transpose(x_true,perm=[0,1,3,2,4])
                x_true3 = tf.transpose(x_true,perm=[0,1,2,3,4])
                
                x_pred1 = tf.transpose(x_pred,perm=[0,1,4,2,3])
                x_pred2 = tf.transpose(x_pred,perm=[0,1,3,2,4])
                x_pred3 = tf.transpose(x_pred,perm=[0,1,2,3,4])
                x_true_list = [x_true1,x_true2,x_true3]
                x_pred_list = [x_pred1,x_pred2,x_pred3]
            else:
                x_true_list = [x_true]
                x_pred_list = [x_pred]
        else:
            if transform_high_dimension:
                x_true1 = tf.transpose(x_true,perm=[0,3,1,2,4])
                x_true2 = tf.transpose(x_true,perm=[0,2,1,3,4])
                x_true3 = tf.transpose(x_true,perm=[0,1,2,3,4])
                x_pred1 = tf.transpose(x_pred,perm=[0,3,1,2,4])
                x_pred2 = tf.transpose(x_pred,perm=[0,2,1,3,4])
                x_pred3 = tf.transpose(x_pred,perm=[0,1,2,3,4])
                x_true_list = [x_true1,x_true2,x_true3]
                x_pred_list = [x_pred1,x_pred2,x_pred3]
            else:
                x_true_list = [x_true]
                x_pred_list = [x_pred]

        feature_maps_getter = FeatureMapsGetter(
            model_name=model_name,
            data_format=data_format,
            use_pooling=use_pooling,
            feature_maps_indicators=feature_maps_indicators,
            dtype=dtype)
        true_feature_reco_results = []
        true_feature_reco_sample_weight = []
        true_style_reco_results = []
        true_style_reco_sample_weight = []
        for x in x_true_list:
            true_feature_reco_result,true_style_reco_result = feature_maps_getter(x)
            true_feature_reco_results += true_feature_reco_result
            true_feature_reco_sample_weight += _feature_reco_sample_weight
            true_style_reco_results += true_style_reco_result 
            true_style_reco_sample_weight += _style_reco_sample_weight

        pred_feature_reco_results = []
        pred_feature_reco_sample_weight = []
        pred_style_reco_results = []
        pred_style_reco_sample_weight = []
        for x in x_pred_list:
            pred_feature_reco_result,pred_style_reco_result = feature_maps_getter(x)
            pred_feature_reco_results += pred_feature_reco_result
            pred_feature_reco_sample_weight += _feature_reco_sample_weight
            pred_style_reco_results += pred_style_reco_result 
            pred_style_reco_sample_weight += _style_reco_sample_weight

        assert true_feature_reco_sample_weight==pred_feature_reco_sample_weight
        assert true_style_reco_sample_weight==pred_style_reco_sample_weight
        loss = 0.0
        if use_feature_reco_loss:
            loss_buf = []
            for y_true,y_pred in zip(true_feature_reco_results,pred_feature_reco_results):
                loss_buf.append(tf.reduce_mean(tf.square(y_true-y_pred)))
            loss_buf_with_sampled_weight = []
            for single_loss,single_sample_weight in zip(loss_buf,true_feature_reco_sample_weight):
                loss_buf_with_sampled_weight.append(single_loss*single_sample_weight)
            loss += tf.reduce_mean(loss_buf_with_sampled_weight)
        if use_style_reco_loss:
            loss_buf = []
            for y_true,y_pred in zip(true_style_reco_results,pred_style_reco_results):
                loss_buf.append(style_diff_2d(y_true,y_pred))
            loss_buf_with_sampled_weight = []
            for single_loss,single_sample_weight in zip(loss_buf,true_style_reco_sample_weight):
                loss_buf_with_sampled_weight.append(single_loss*single_sample_weight)
            loss += tf.reduce_mean(loss_buf_with_sampled_weight)
        vf = PerceptualLossExtractor(model_name=model_name,
                 data_format=data_format,
                 transform_high_dimension=transform_high_dimension,
                 use_pooling=use_pooling,
                 use_feature_reco_loss=use_feature_reco_loss,
                 use_style_reco_loss=use_style_reco_loss,
                 feature_reco_index=feature_reco_index,
                 feature_reco_sample_weight=_feature_reco_sample_weight,
                 style_reco_index=style_reco_index,
                 style_reco_sample_weight=_style_reco_sample_weight)
        loss2 = vf([_x_true,_x_pred])
        computed = tf.reduce_mean(loss-loss2)
        _assert_allclose_according_to_type(
            computed,0.0,
            rtol=1e-3,
            atol=1e-3,
            float_rtol=1e-3,
            float_atol=1e-3,)