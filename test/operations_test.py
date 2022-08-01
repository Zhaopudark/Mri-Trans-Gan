import pytest
import numpy as np
from utils.operations import (np_individual_min,
                              np_individual_max,
                              np_individual_min_max,
                              np_div_no_nan,
                              np_zero_close,
                              np_nan_to_zero,
                              np_sequence_reduce_max,
                              np_sequence_reduce_min,
                              np_sequence_reduce_min_max,
                              norm_z_score,
                              norm_min_max,redomain)
#---------------------------------------#helper---------------------------------------#
def _assert_allclose_according_to_type(a,b):
    """
    Similar to tf.test.TestCase.assertAllCloseAccordingToType()
    but this doesn't need a subclassing to run.
    """
    types = [np.float16,np.float32,np.float64]
    tols = [1e-3,1e-7,1e-15]
    rtol = 1e-8
    # assert a.dtype in types
    # assert b.dtype in types
    a_t = tols[types.index(a.dtype)]
    b_t = tols[types.index(b.dtype)]
    atol = max(a_t,b_t)
    np.testing.assert_allclose(a, b,atol=atol,rtol=rtol,equal_nan=False)
def _gen_single_input(shape,dtype,unit=1):
    np.random.seed(0)
    return np.array(np.random.uniform(0,unit,size=shape),dtype=dtype)
def _gen_sequence_inputs(shape,dtype):
    np.random.seed(0)
    return [np.array(np.random.uniform(0,1,size=shape),dtype=dtype) for _ in range(10)]
def _gen_mask(shape,dtype):
    mask = np.array(np.random.uniform(0,10,size=shape),dtype=dtype)
    return np.where(mask>5,1,0)
def _cal_sequence_min_max(arrays_list:list[np.ndarray]):
    min_buf = None
    max_buf = None
    for x in arrays_list:
        if min_buf is None and max_buf is None:
            min_buf = x
            max_buf = x
        else:
            min_buf = np.minimum(min_buf,x)
            max_buf = np.maximum(max_buf,x)
    return min_buf,max_buf

def _cal_sequence_min(arrays_list:list[np.ndarray]):
    min_buf = None
    for x in arrays_list:
        if min_buf is None:
            min_buf = x
        else:
            min_buf = np.minimum(min_buf,x)
    return min_buf

def _cal_sequence_max(arrays_list:list[np.ndarray]):
    max_buf = None
    for x in arrays_list:
        if max_buf is None:
            max_buf = x
        else:
            max_buf = np.maximum(max_buf,x)
    return max_buf

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('mdtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
def test_np_individual_min(input_shape,dtype,mdtype):
    x = _gen_single_input(input_shape,dtype)
    mask = _gen_mask(input_shape,mdtype)
    y = np_individual_min(x,mask=mask)
    assert y.dtype==x.dtype==dtype
    _assert_allclose_according_to_type(y,(np.amin((x-1)*mask+1)))
    y = np_individual_min(x,mask=None)
    assert y.dtype==x.dtype==dtype
    _assert_allclose_according_to_type(y,np.amin(x))

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('mdtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
def test_np_individual_max(input_shape,dtype,mdtype):
    x = _gen_single_input(input_shape,dtype)
    mask = _gen_mask(input_shape,mdtype)
    y = np_individual_max(x,mask=mask)
    assert y.dtype==x.dtype==dtype
    _assert_allclose_according_to_type(y,(np.amax((x+1)*mask-1)))
    y = np_individual_max(x,mask=None)
    assert y.dtype==x.dtype==dtype
    _assert_allclose_according_to_type(y,np.amax(x))


@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('mdtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
def test_np_individual_min_max(input_shape,dtype,mdtype):
    x = _gen_single_input(input_shape,dtype)
    mask = _gen_mask(input_shape,mdtype)
    y_min,y_max = np_individual_min_max(x,mask=mask)
    assert y_min.dtype==y_max.dtype==x.dtype==dtype
    mask.astype(dtype)
    _assert_allclose_according_to_type(y_min,(np.amin((x-1)*mask+1)))
    _assert_allclose_according_to_type(y_max,(np.amax((x+1)*mask-1)))
    
    y_min,y_max = np_individual_min_max(x,mask=None)
    assert y_min.dtype==y_max.dtype==x.dtype==dtype
    np.testing.assert_equal(y_min,np.amin(x))
    np.testing.assert_equal(y_max,np.amax(x))



@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_np_div_no_nan(input_shape,dtype):
    a = np.array([0,np.inf,np.NINF,np.NINF,np.PINF,1],dtype=dtype)
    b = np.array([0,np.inf,np.inf,np.PINF,np.NINF,2],dtype=dtype)
    y1 = np_div_no_nan(a,b)
    y2 = np.array([0,0,0,0,0,0.5],dtype=dtype)
    assert y1.dtype==y2.dtype==dtype
    _assert_allclose_according_to_type(y1,y2)


@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
def test_np_zero_close(input_shape,dtype):
    x = _gen_single_input(input_shape,dtype)
    y1 = np_zero_close(x)
    y2 = np_zero_close((x*100)/100)
    y3 = np_zero_close((y1*100)/100)
    assert y1.dtype==x.dtype==dtype
    _assert_allclose_according_to_type(y1.astype(y2.dtype),y2)
    _assert_allclose_according_to_type(y1.astype(y3.dtype),y3)

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
def test_np_nan_to_zero(input_shape,dtype):
    x1 = np.zeros(shape=input_shape,dtype=dtype)
    x2 = np.zeros(shape=input_shape,dtype=dtype)
    x3 = np.zeros(shape=input_shape,dtype=dtype)
    y_nan = x1/x2
    y = np_nan_to_zero(y_nan)
    assert x1.dtype==x2.dtype==x3.dtype==dtype
    assert y.dtype==y_nan.dtype
    np.testing.assert_equal(y,x3)
   

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
def test_np_sequence_reduce_max(input_shape,dtype):
    arrays_list = _gen_sequence_inputs(input_shape,dtype)
    _max = np_sequence_reduce_max(arrays_list)
    assert _max.shape==tuple(input_shape)
    assert _max.dtype==dtype   
    max_buf = _cal_sequence_max(arrays_list)
    np.testing.assert_equal(_max,max_buf)

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
def test_np_sequence_reduce_min(input_shape,dtype):
    arrays_list = _gen_sequence_inputs(input_shape,dtype)
    _min = np_sequence_reduce_min(arrays_list)
    assert _min.shape==tuple(input_shape)
    assert _min.dtype==dtype
    min_buf = _cal_sequence_min(arrays_list)
    np.testing.assert_equal(_min,min_buf)

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
def test_np_sequence_reduce_min_max(input_shape,dtype):
    arrays_list = _gen_sequence_inputs(input_shape,dtype)
    _min,_max = np_sequence_reduce_min_max(arrays_list)
    assert _min.shape==tuple(input_shape)
    assert _min.dtype==dtype
    assert _max.shape==tuple(input_shape)
    assert _max.dtype==dtype
    min_buf,max_buf = _cal_sequence_min_max(arrays_list)
    np.testing.assert_equal(_min,min_buf)
    np.testing.assert_equal(_max,max_buf)


@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('input_dtype',[np.float32,np.float64])
@pytest.mark.parametrize('mdtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('foreground_offset',[-0.3,0.1,0.5,1.0,2.0,4.0])
def test_norm_z_score_with_mask(input_shape,input_dtype,mdtype,dtype,foreground_offset):
    np.random.seed(0)
    x = np.array(np.random.uniform(0,10,size=input_shape),dtype=input_dtype)
    mask = np.array(np.random.uniform(0,10,size=input_shape),dtype=mdtype)
    mask = np.where(mask>5,1,0)
    y = norm_z_score(x=x,mask=mask,foreground_offset=foreground_offset,dtype=dtype)
    assert y.dtype==dtype
    x = x.astype(dtype)
    _where = np.where(mask>0.5,True,False)
    mean = np.mean(x,where=_where)
    std = np.std(x,ddof=0.0,where=_where)
    y2 = ((x-mean)/std+foreground_offset)*(mask.astype(dtype))
    assert y.dtype==y2.dtype
    _assert_allclose_according_to_type(y,y2)

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('input_dtype',[np.float32,np.float64])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('foreground_offset',[-0.3,0.1,0.5,1.0,2.0,4.0])
def test_norm_z_score_without_mask(input_shape,input_dtype,dtype,foreground_offset):
    np.random.seed(0)
    x = np.array(np.random.uniform(0,100,size=input_shape),dtype=input_dtype)
    y = norm_z_score(x=x,mask=None,foreground_offset=foreground_offset,dtype=dtype)
    x = x.astype(dtype)
    mean = np.mean(x)
    std = np.std(x,ddof=0.0)
    y2 = (x-mean)/std
    assert y.dtype==y2.dtype
    _assert_allclose_according_to_type(y,y2)



@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('input_dtype',[np.float32,np.float64])
@pytest.mark.parametrize('mdtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('foreground_offset',[0.1,0.5,0.9])
def test_norm_min_max_with_mask(input_shape,input_dtype,mdtype,dtype,foreground_offset):
  
    arrays_list = _gen_sequence_inputs(input_shape,input_dtype)
    global_min_max= _cal_sequence_min_max(arrays_list)
    x =  arrays_list[0]
    mask = _gen_mask(input_shape,mdtype)
    y = norm_min_max(x=x,global_min_max=global_min_max,mask=mask,foreground_offset=foreground_offset,dtype=dtype)
    assert y.dtype==dtype
    x = x.astype(dtype)
    global_min_max = global_min_max[0].astype(dtype),global_min_max[1].astype(dtype)
    _where = np.where(mask>0.5,True,False)
    _min = np.amin(global_min_max[0],initial=np.PINF,where=_where)
    _max = np.amax(global_min_max[1],initial=np.NINF,where=_where)
    x = (x-_min)/(_max-_min)
    y2 = (x*(1.0-foreground_offset)+foreground_offset)*(mask.astype(dtype))
    assert y.dtype==y2.dtype
    _assert_allclose_according_to_type(y,y2)

    global_min_max = None
    x = _gen_single_input(input_shape,input_dtype)
    mask = _gen_mask(input_shape,mdtype)
    y = norm_min_max(x=x,global_min_max=global_min_max,mask=mask,foreground_offset=foreground_offset,dtype=dtype)
    assert y.dtype==dtype
    x = x.astype(dtype)
    _where = np.where(mask>0.5,True,False)
    _min = np.amin(x,initial=np.PINF,where=_where)
    _max = np.amax(x,initial=np.NINF,where=_where)
    x = (x-_min)/(_max-_min)
    y2 = (x*(1.0-foreground_offset)+foreground_offset)*(mask.astype(dtype))
    assert y.dtype==y2.dtype
    _assert_allclose_according_to_type(y,y2)


@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3],[128,128,128],[64,64,64]])
@pytest.mark.parametrize('input_dtype',[np.float32,np.float64])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_norm_min_max_without_mask(input_shape,input_dtype,dtype):
  
    arrays_list = _gen_sequence_inputs(input_shape,input_dtype)
    global_min_max= _cal_sequence_min_max(arrays_list)
    x =  arrays_list[0]
    y = norm_min_max(x=x,global_min_max=global_min_max,mask=None,dtype=dtype)
    assert y.dtype==dtype
    x = x.astype(dtype)
    global_min_max = global_min_max[0].astype(dtype),global_min_max[1].astype(dtype)
    _min = np.amin(global_min_max[0])
    _max = np.amax(global_min_max[1])
    y2 = (x-_min)/(_max-_min)
    assert y.dtype==y2.dtype
    _assert_allclose_according_to_type(y,y2)

    global_min_max = None
    x = _gen_single_input(input_shape,input_dtype)

    y = norm_min_max(x=x,global_min_max=global_min_max,mask=None,dtype=dtype)
    assert y.dtype==dtype
    x = x.astype(dtype)
    _min = np.amin(x)
    _max = np.amax(x)
    y2 = (x-_min)/(_max-_min)
    assert y.dtype==y2.dtype
    _assert_allclose_according_to_type(y,y2)


@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3]])
@pytest.mark.parametrize('input_dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('domain',[(0,1),(0,1.),(0.0,1.0),(-1,1),(-1,1.),(-1.0,1.0),(0,255),(0,255.),(0.0,255.0)])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_redomain(input_shape,input_dtype,domain,dtype):

    x = _gen_single_input(input_shape,input_dtype,unit=10)
    y = redomain(x=x,domain=domain,dtype=dtype)
    assert y.dtype==dtype
    computed = y.max()-max(domain)
    assert np.isclose(computed,0.0,rtol=1e-05,atol=1e-08,equal_nan=False)
    computed = y.min()-min(domain)
    assert np.isclose(computed,0.0,rtol=1e-05,atol=1e-08,equal_nan=False)