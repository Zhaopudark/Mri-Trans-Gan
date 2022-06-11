import pytest
import numpy as np
from utils.dataset_helper  import np_reduce_min,np_reduce_max,np_min_max_on_sequence,np_zero_close,np_nan_to_zero,np_div_no_nan,norm_min_max,norm_z_score,redomain

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3]])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('mdtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('ignore_nan',[True,False])
def test_np_reduce_min(input_shape,dtype,mdtype,ignore_nan):
    np.random.seed(1000)
    x = np.array(np.random.uniform(0,100,size=input_shape),dtype=dtype)
    mask = np.array(np.random.uniform(0,10,size=input_shape),dtype=mdtype)
    mask = np.where(mask>5,1,0)
    y = np_reduce_min(x,mask=mask,ignore_nan=ignore_nan)
    assert y.dtype==x.dtype==dtype
    y = np_reduce_min(x,mask=None,ignore_nan=ignore_nan)
    assert y.dtype==x.dtype==dtype


@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3]])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('mdtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('ignore_nan',[True,False])
def test_np_reduce_max(input_shape,dtype,mdtype,ignore_nan):
    np.random.seed(1000)
    x = np.array(np.random.uniform(0,100,size=input_shape),dtype=dtype)
    mask = np.array(np.random.uniform(0,10,size=input_shape),dtype=mdtype)
    mask = np.where(mask>5,1,0)
    y = np_reduce_max(x,mask=mask,ignore_nan=ignore_nan)
    assert y.dtype==x.dtype==dtype
    y = np_reduce_max(x,mask=None,ignore_nan=ignore_nan)
    assert y.dtype==x.dtype==dtype


@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3]])
@pytest.mark.parametrize('a_dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('b_dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
def test_np_div_no_nan_dtype(input_shape,a_dtype,b_dtype):
    np.random.seed(1000)
    a = np.array(np.random.uniform(0,100,size=input_shape),dtype=a_dtype)
    b = np.array([1],dtype=b_dtype)
    y1 = np_div_no_nan(a,b)
    y2 = a/b
    assert y1.dtype==y2.dtype
    if (a_dtype in [np.float32,np.int32])and(b_dtype in [np.float32,np.int32]):
        computed = np.mean(y1)-np.mean(y1.astype(np.float32))
        assert np.isclose(computed,0.0,rtol=1e-05,atol=1e-04,equal_nan=False)
        computed = np.mean(a)-np.mean(y1.astype(np.float32))
        assert np.isclose(computed,0.0,rtol=1e-05,atol=1e-04,equal_nan=False)

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3]])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_np_div_no_nan_dtype_same_float_type(input_shape,dtype):
    np.random.seed(1000)
    a = np.array(np.random.uniform(0,100,size=input_shape),dtype=dtype)
    b = np.array([1],dtype=dtype)
    y1 = np_div_no_nan(a,b)
    y2 = a/b
    assert y1.dtype==y2.dtype==dtype
    computed = np.mean(y1)-np.mean(y2)
    assert np.isclose(computed,0.0,rtol=1e-05,atol=1e-08,equal_nan=False)


@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3]])
@pytest.mark.parametrize('dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
def test_np_zero_close_dtype(input_shape,dtype):
    np.random.seed(1000)
    x = np.array(np.random.uniform(0,100,size=input_shape),dtype=dtype)
    y = np_zero_close(x)
    assert y.dtype==x.dtype==dtype

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3]])
@pytest.mark.parametrize('dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
def test_np_nan_to_zero_dtype(input_shape,dtype):
    np.random.seed(1000)
    x = np.array(np.random.uniform(0,100,size=input_shape),dtype=dtype)
    y = np_nan_to_zero(x)
    assert y.dtype==x.dtype==dtype
   

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3]])
@pytest.mark.parametrize('dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('ignore_nan',[True,False])
def test_np_min_max_on_sequence(input_shape,dtype,ignore_nan):
    np.random.seed(1000)
    def gen():
        for _ in range(10):
            yield np.array(np.random.uniform(0,100,size=input_shape),dtype=dtype)
    _min,_max = np_min_max_on_sequence(gen(),ignore_nan=ignore_nan)
    assert _min.shape==_max.shape==tuple(input_shape)
    assert _min.dtype==_max.dtype==dtype

   

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3]])
@pytest.mark.parametrize('input_dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('mdtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
@pytest.mark.parametrize('ignore_nan',[True,False])
def test_norm_min_max_dtype(input_shape,input_dtype,mdtype,dtype,ignore_nan):
    np.random.seed(1000)
    _min = np.array(np.random.uniform(0,100,size=input_shape),dtype=input_dtype)
    _max = np.array(np.random.uniform(0,100,size=input_shape),dtype=input_dtype)
    x = np.array(np.random.uniform(0,100,size=input_shape),dtype=input_dtype)
    mask = np.array(np.random.uniform(0,10,size=input_shape),dtype=mdtype)
    mask = np.where(mask>5,1,0)
    y = norm_min_max(x=x,global_min_max=(_min,_max),mask=mask,ignore_nan=ignore_nan,dtype=dtype)
    assert y.dtype==dtype
    y = norm_min_max(x=x,global_min_max=(_min,_max),mask=None,ignore_nan=ignore_nan,dtype=dtype)
    assert y.dtype==dtype
    y = norm_min_max(x=x,global_min_max=None,mask=mask,ignore_nan=ignore_nan,dtype=dtype)
    assert y.dtype==dtype
    y = norm_min_max(x=x,global_min_max=None,mask=None,ignore_nan=ignore_nan,dtype=dtype)
    assert y.dtype==dtype

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3]])
@pytest.mark.parametrize('input_dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('mdtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_norm_z_score(input_shape,input_dtype,mdtype,dtype):
    np.random.seed(1000)
    x = np.array(np.random.uniform(0,100,size=input_shape),dtype=input_dtype)
    mask = np.array(np.random.uniform(0,10,size=input_shape),dtype=mdtype)
    mask = np.where(mask>5,1,0)
    y = norm_z_score(x=x,mask=mask,dtype=dtype)
    assert y.dtype==dtype
    y = norm_z_score(x=x,mask=None,dtype=dtype)
    assert y.dtype==dtype

@pytest.mark.parametrize('input_shape',[[240,240,155],[128,128,3]])
@pytest.mark.parametrize('input_dtype',[np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64])
@pytest.mark.parametrize('domain',[(0,1),(0,1.),(0.0,1.0),(-1,1),(-1,1.),(-1.0,1.0),(0,255),(0,255.),(0.0,255.0)])
@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_redomain(input_shape,input_dtype,domain,dtype):
    np.random.seed(1000)
    x = np.array(np.random.uniform(0,100,size=input_shape),dtype=input_dtype)
    y = redomain(x=x,domain=domain,dtype=dtype)
    assert y.dtype==dtype
    computed = y.max()-max(domain)
    assert np.isclose(computed,0.0,rtol=1e-05,atol=1e-08,equal_nan=False)
    computed = y.min()-min(domain)
    assert np.isclose(computed,0.0,rtol=1e-05,atol=1e-08,equal_nan=False)