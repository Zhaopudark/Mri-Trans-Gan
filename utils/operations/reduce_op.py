 
import functools
from typing import Any,Iterable,Callable

import numpy as np

def reduce_with_map(reduce_func:Callable,inputs:Iterable[Any],map_func:Callable[[Any],Any]=None):
    if map_func is not None:
        def gen(items):
            for item in items:
                yield map_func(item)
        inputs = gen(inputs)
    return functools.reduce(reduce_func,inputs) 

def reduce_same(inputs:Iterable[Any],map_func:Callable[[Any],Any]=None):
    def _check(x1,x2):
        if x1!=x2:
            raise ValueError(f"Elements in sequence is not the same: `{x1}`!=`{x2}`")
        return x1 
    return reduce_with_map(_check,inputs,map_func=map_func)

# def _fmin_fmax(x1:np.ndarray|tuple[np.ndarray,np.ndarray],x2:np.ndarray):
#     if isinstance(x1,np.ndarray):
#         _min = np.fmin(x1,x2)
#         _max = np.fmax(x1,x2)
#         return (_min,_max)
#     _min = np.fmin(x1[0],x2)
#     _max = np.fmax(x1[1],x2)
#     return (_min,_max)

# def _min_max(x1:np.ndarray|tuple[np.ndarray,np.ndarray],x2:np.ndarray):
#     if isinstance(x1,np.ndarray):
#         _min = np.minimum(x1,x2)
#         _max = np.maximum(x1,x2)
#         return (_min,_max)
#     _min = np.minimum(x1[0],x2)
#     _max = np.maximum(x1[1],x2)
#     return (_min,_max)

# def np_reduce_min(x:np.ndarray,mask:np.ndarray=None,ignore_nan:bool=True):
#     assert x.dtype in [np.float16,np.float32,np.float64]
#     if mask is not None:
#         _where = np.where(mask>0.5,True,False)
#         return np.nanmin(x,initial=np.PINF,where=_where) if ignore_nan else np.amin(x,initial=np.PINF,where=_where)
#     return np.nanmin(x) if ignore_nan else np.amin(x)

# def np_reduce_max(x:np.ndarray,mask:np.ndarray=None,ignore_nan:bool=True):
#     assert x.dtype in [np.float16,np.float32,np.float64]
#     if mask is not None:
#         _where = np.where(mask>0.5,True,False)
#         return np.nanmax(x,initial=np.NINF,where=_where) if ignore_nan else np.amax(x,initial=np.NINF,where=_where)
#     return np.nanmax(x) if ignore_nan else np.amax(x)
    

# def np_reduce_min_max(sequence:Iterable[np.ndarray],ignore_nan:bool=True,wrapper:Callable=None):
#     if ignore_nan:
#         # The net effect is that NaNs are ignored when possible.
#         return functools.reduce(_fmin_fmax,sequence) if wrapper is None else functools.reduce(wrapper(_fmin_fmax),sequence)
#     else:
#         # The net effect is that NaNs are propagated.
#         return functools.reduce(_min_max,sequence)  if wrapper is None else functools.reduce(wrapper(_min_max),sequence)



def np_individual_min(x:np.ndarray,mask:np.ndarray=None):
    assert np.isfinite(x).all()
    assert x.dtype in [np.float16,np.float32,np.float64]
    if mask is not None:
        _where = np.where(mask>0.5,True,False)
        return np.amin(x,initial=np.PINF,where=_where)
    return np.amin(x)

def np_individual_max(x:np.ndarray,mask:np.ndarray=None):
    assert np.isfinite(x).all()
    assert x.dtype in [np.float16,np.float32,np.float64]
    if mask is not None:
        _where = np.where(mask>0.5,True,False)
        return np.amax(x,initial=np.NINF,where=_where)
    return np.amax(x)    

def np_individual_min_max(x:np.ndarray,mask:np.ndarray=None):
    assert np.isfinite(x).all()
    assert x.dtype in [np.float16,np.float32,np.float64]
    if mask is not None:
        _where = np.where(mask>0.5,True,False)
        return  np.amin(x,initial=np.PINF,where=_where),np.amax(x,initial=np.NINF,where=_where)
    return np.amin(x),np.amax(x)    

def _np_reduce_min_func(x1:np.ndarray,x2:np.ndarray):
    assert np.isfinite(x1).all()
    assert np.isfinite(x2).all()
    return np.minimum(x1,x2)
def _np_reduce_max_func(x1:np.ndarray,x2:np.ndarray):
    assert np.isfinite(x1).all()
    assert np.isfinite(x2).all()
    return np.maximum(x1,x2)
def _np_reduce_min_max_func(x1:np.ndarray|tuple[np.ndarray,np.ndarray],x2:np.ndarray):
    if isinstance(x1,np.ndarray):
        assert np.isfinite(x1).all()
        assert np.isfinite(x2).all()
        _min = np.minimum(x1,x2)
        _max = np.maximum(x1,x2)
        return (_min,_max)
    assert np.isfinite(x1).all()
    assert np.isfinite(x2).all()
    _min = np.minimum(x1[0],x2)
    _max = np.maximum(x1[1],x2)
    return (_min,_max)

def np_sequence_reduce_min(sequence:Iterable[np.ndarray]):
    return functools.reduce(_np_reduce_min_func,sequence)
def np_sequence_reduce_max(sequence:Iterable[np.ndarray]):
    return functools.reduce(_np_reduce_max_func,sequence) 
def np_sequence_reduce_min_max(sequence:Iterable[np.ndarray]):
    return functools.reduce(_np_reduce_min_max_func,sequence) 