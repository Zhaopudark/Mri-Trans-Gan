import os
import logging
from typeguard import typechecked 
from typing import Callable,Iterable,Any,Literal,Sequence
import random
import functools
import copy
import itertools
import operator
import nibabel as nib

import numpy as np 
import tensorflow as tf


class DataIter():
    """
    return an datas Iterator that 
        can control random behavior with an inner and individual random.Random() instance
        can iter form the latest counters (step epoch) 
    Args:
        datas: list of data that can be 'shuffle()' by random.Random()
        counters: {"step":step,"epoch":epoch}, contain the quote of global step and epoch
        seed: random seed, `None` means do not random
    Method:
        __iter__(): 
            It will backtrack status from the original dataset by counters before delivering elements.
            Since random.Random().shuffle() will change status after each application, 
            we should backtrack status from scratch everytime.
    """
    @typechecked
    def __init__(self,datas:list,counters:dict[Literal["step","epoch"],tf.Variable]|None=None,seed:int|None=None) -> None:
        if counters is None:
            counters = {"step":0,"epoch":0}
        self._epoch = counters["epoch"] # have passed epoch
        self._step = counters["step"] # have passed step
        self.datas = copy.deepcopy(datas) # do not influence original data, fixing the data
        self.seed = seed
        self._check()
    def _check(self):
        assert (self.step//self.length)==self.epoch
    @property
    def length(self):
        return len(self.datas)
    @property
    def epoch(self):
        return self._epoch.numpy() if hasattr(self._epoch,"numpy") else self._epoch
    @property
    def step(self):
        return self._step.numpy() if hasattr(self._step,"numpy") else self._step
    def __iter__(self):
        datas = copy.deepcopy(self.datas) # do not influence the fixed data
        random = get_random_from_seed(self.seed) 
        for _ in range(self.epoch): # random from scratch
            datas = random_datas(datas,random)
        # self.step%self.length do not need 'minus 1', because it is the start index exactly
        return itertools.islice(datas,self.step%self.length,self.length) 
    def __repr__(self) -> str:
        return f"Static indices is epoch:{self.epoch} step:{self.step}."



class RecordManager():
    """
    This class manager the record files. We can get our expected datas from specific files by `load()`.

    Managing datasets by saving and loading tf-record files has been deprecated for the following reasons:
        1. tf-record files is hard to read and save:
            Need use `tf.train.Feature`, `tf.train.Features`, `tf.io.TFRecordWriter` for loading
                and use `tf.data.TFRecordDataset`,`tf.io.FixedLenFeature` and `tf.io.parse_single_example` for reading.
            In additional, if save and load non-scalar tensors, should use `tf.io.serialize_tensor` to transfer it to bytes and `tf.io.parse_tensor` to get original tensor
        2. tf-record's saving behavior only works in Eager Mode, i.e., can not used in tf.data.Dataset.map() unless wrappered by tf.py_function 
        3. `tf.io.TFRecordDataset` will load all files in memory when shuffle() is used, i.e.,  the mapping of inner segments within `tf.io.TFRecordDataset` is not implemented.
    So, here, we build a new RecordManager with the following keys:
        1. Make a mapping from specific files to file stamps(names)
        2. In disk or remote device, each specific file can be saved separately or in groups, depending on the pressure and flux requirements of the data pipeline.
        3. When the corresponding specific file of a file stamp does not exist, generate and save the file before return.
        4. Use memory and storage consumption as less as possible.
    `numpy`'s NPZ file can provide us with solutions for the above keys.

    Args:
        path: the record file's saved and loaded path
    Methods:
        load:
            load(name:str,structure:dict[str,Any],gen_func:Callable[[],dict[str,np.ndarray]])
            this method works on a group of specific files(ndarrays), that saved in a NPZ file (determined by `name`)
            `structure`'s keys() determines what files(ndarrays) in the saved NPZ file will return.
            For greater compatibility, i.g., it is more convenient to process sequence in posterior procedures, here we just return sequence of ndarray instead of a dict in the same sturcture with `structure`.

    """
    @typechecked
    def __init__(self,path:str="./",) -> None:
        self.path = os.path.normpath(path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        assert os.path.isdir(self.path)
    def _get_target_path(self,name:str):
        return os.path.normpath(f"{self.path}\\{name}.npz") 
    def load(self,name:str,structure:dict[str,Any],gen_func:Callable[[],dict[str,np.ndarray]])->Sequence[np.ndarray]:
        keys = structure.keys()
        files_getter = operator.itemgetter(*keys)
        try:
            saved_file = np.load(self._get_target_path(name))
            output = files_getter(saved_file)
        except KeyError:
            saved_file = dict(**saved_file) | gen_func()
            np.savez_compressed(self._get_target_path(name),**saved_file)
            output = files_getter(saved_file)
        except FileNotFoundError:
            saved_file = gen_func()
            np.savez_compressed(self._get_target_path(name),**saved_file)
            output = files_getter(saved_file)
        return output
#--------------------------------------------------------------------------------#
def _fmin_fmax(x1:np.ndarray|tuple[np.ndarray,np.ndarray],x2:np.ndarray):
    if isinstance(x1,np.ndarray):
        _min = np.fmin(x1,x2)
        _max = np.fmax(x1,x2)
        return (_min,_max)
    _min = np.fmin(x1[0],x2)
    _max = np.fmax(x1[1],x2)
    return (_min,_max)
def _min_max(x1:np.ndarray|tuple[np.ndarray,np.ndarray],x2:np.ndarray):
    if isinstance(x1,np.ndarray):
        _min = np.minimum(x1,x2)
        _max = np.maximum(x1,x2)
        return (_min,_max)
    _min = np.minimum(x1[0],x2)
    _max = np.maximum(x1[1],x2)
    return (_min,_max)
@typechecked
def np_reduce_min(x:np.ndarray,mask:np.ndarray=None,ignore_nan:bool=True):
    assert x.dtype in [np.float16,np.float32,np.float64]
    if mask is not None:
        _where = np.where(mask>0.5,True,False)
        return np.nanmin(x,initial=np.PINF,where=_where) if ignore_nan else np.amin(x,initial=np.PINF,where=_where)
    return np.nanmin(x) if ignore_nan else np.amin(x)
@typechecked
def np_reduce_max(x:np.ndarray,mask:np.ndarray=None,ignore_nan:bool=True):
    assert x.dtype in [np.float16,np.float32,np.float64]
    if mask is not None:
        _where = np.where(mask>0.5,True,False)
        return np.nanmax(x,initial=np.NINF,where=_where) if ignore_nan else np.amax(x,initial=np.NINF,where=_where)
    return np.nanmax(x) if ignore_nan else np.amax(x)
    
@typechecked
def np_min_max_on_sequence(sequence:Iterable[np.ndarray],ignore_nan:bool=True,wrapper:Callable=None):
    if ignore_nan:
        # The net effect is that NaNs are ignored when possible.
        return functools.reduce(_fmin_fmax,sequence) if wrapper is None else functools.reduce(wrapper(_fmin_fmax),sequence)
    else:
        # The net effect is that NaNs are propagated.
        return functools.reduce(_min_max,sequence)  if wrapper is None else functools.reduce(wrapper(_min_max),sequence)
     
@typechecked
def np_zero_close(x:np.ndarray):
    _where = np.isclose(x,0.0,rtol=1e-05,atol=1e-08,equal_nan=False)# 与0比较 其余取默认值(默认nan与nan不相等 返回false,nan与非nan不相等,返回false)
    x[_where]=0
    return x

@typechecked
def np_nan_to_zero(x:np.ndarray):
    _where = np.isnan(x)# 与0比较 其余取默认值(默认nan与nan不相等 返回false,nan与非nan不相等,返回false)
    x[_where]=0
    return x

@typechecked
def np_div_no_nan(a:np,b:np):
    """构建基于numpy的div_no_nan
    input:被除数a,除数b
    output:a/b, Nan值归0
    当a b 为同浮点型时  输出为同浮点型
    当a b 类型不同时 输出类型是一个不定的类型
    """
    out = a/b
    _where = np.isnan(out)
    out[_where] = 0
    return out
@typechecked
def norm_min_max(x:np.ndarray,
                global_min_max:tuple[np.ndarray,np.ndarray]|None=None,
                mask:np.ndarray|None=None,
                foreground_offset:float=0.0,
                ignore_nan:bool=True,
                dtype:type=np.float32,
                **kwargs):
    """
    将输入以min_max归一化到0-1domain
    x:input
    mask:区分前景背景的 0 1mask
    foreground_offset: 前景偏移量
    x = Vaild_Value*(1-foreground_offset)+foreground_offset
    """
    assert dtype in [np.float16,np.float32,np.float64]
    
    if global_min_max is not None:
        assert x.dtype==global_min_max[0].dtype==global_min_max[1].dtype
        _min,_max = np_reduce_min(global_min_max[0].astype(dtype),mask,ignore_nan),np_reduce_max(global_min_max[1].astype(dtype),mask,ignore_nan)
    else:
        _min,_max = np_reduce_min(x.astype(dtype),mask,ignore_nan),np_reduce_max(x.astype(dtype),mask,ignore_nan)
    x = np_div_no_nan(x.astype(dtype)-_min,_max-_min)
    # the above steps support mask  is None or not None
    # the following steps is special procedures when mask is  not None
    assert 0.<=foreground_offset<=1.
    if mask is not None:
        x = x*(1.0-foreground_offset)+foreground_offset # 无mask时 foreground_offset是没有意义的
        x = x*(mask.astype(dtype))
        x = np_zero_close(x)
        x = np_nan_to_zero(x)  #if x*mask has inf*0 = Nan, make Nan to 0
    return x

def norm_z_score(x:np.ndarray,mask:np.ndarray|None=None,foreground_offset:float=0.0,dtype:type=np.float32,**kwargs):
    """
    将输入归一化到 均值为0 方差为1
    img:input
    mask:区分前景背景的 0 1mask
        当mask存在时,计算的是有效区域的均值标准差,将img的有效区域归一化到0均值 将背景归0
        img = (img-mean)/std
    foreground_offset: 前景偏移量 在z_score中,是有效区域整体(等价于有效区域均值)的偏移量 可以用于探究在z_score将背景也归到0或者其他值是否合理
    img = Vaild_Value+foreground_offset
    """
    assert dtype in [np.float32,np.float64] # np.float16 will lead to overflow
    x = x.astype(dtype)
    if mask is not None:
        _where = np.where(mask>0.5,True,False)
        mean = np.mean(x,where=_where)
        std = np.std(x,ddof=0,where=_where) #将一个图像的所有体素视为总体 求的是这个总体的标准差
        x = np_div_no_nan(x-mean,std)
        x = x+foreground_offset
        x = x*(mask.astype(dtype))
        x = np_zero_close(x) 
        return np_nan_to_zero(x) #if x*mask has inf*0 = Nan, make Nan to 0
    else:
        mean = np.mean(x)
        std = np.std(x,ddof=0) #将一个图像的所有体素视为总体 求的是这个总体的标准差
        return np_div_no_nan(x-mean,std)

@typechecked
def redomain(x:np.ndarray,domain:tuple=(0.0,1.0),dtype:type=np.float32):
    """
    将任意值域的输入均匀得映射到目标值域
    domain a tuple consists of min_val and max_val
    """
    x = x.astype(dtype)
    _min = x.min()
    _max = x.max()
    x = np_div_no_nan(x-_min,_max-_min)
    domain_min = min(domain)
    domain_max = max(domain)
    x = x*(domain_max-domain_min)+domain_min
    return np_zero_close(x)
#--------------------------------------------------------------------------------#
@typechecked
def read_nii_file(path:str,dtype=np.int32): #np.int32确保足以承载原始数据 对于norm后的数据采用np.float32
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = nib.load(path)
    affine = img.affine
    header = img.header
    img = np.array(img.dataobj[:,:,:],dtype=dtype)
    return img,affine,header
@typechecked
def _sync_nii_header_dtype(img:np.ndarray,header=None):
    if img.dtype == np.int16:
        header['bitpix'] = np.array(16,dtype=header['bitpix'].dtype)
        header['datatype'] = np.array(4,dtype=header['datatype'].dtype)
    elif img.dtype == np.int32:
        header['bitpix'] = np.array(32,dtype=header['bitpix'].dtype)
        header['datatype'] = np.array(8,dtype=header['datatype'].dtype)
    elif img.dtype == np.float32:
        header['bitpix'] = np.array(32,dtype=header['bitpix'].dtype)
        header['datatype'] = np.array(16,dtype=header['datatype'].dtype)
    else:
        raise ValueError(
            f"Unsupported nii data type {img.dtype}. Only support np.int16 np.int32 np.float32. More dtypes will be supported in the future."
        )
    return header
@typechecked
def save_nii_file(img:np.ndarray,path:str,affine=None,header=None):
    header = _sync_nii_header_dtype(img,header)
    img_ii = nib.Nifti1Image(img,affine=affine,header=header)
    nib.save(img_ii,path)

@typechecked
def random_datas(datas:list|dict[str,list],random:random.Random|None=None):
    if random is not None:
        if isinstance(datas,dict):
            state = random.getstate()
            for key in datas:
                random.setstate(state)
                random.shuffle(datas[key])
        else:
            random.shuffle(datas) # since datas has been shuffled, the next shuffle will not be the same
        logging.getLogger(__name__).info(f"Random complete!,the first data is {datas[0]}")
    return datas
@typechecked
def get_random_from_seed(seed:int|None=None):
    return random.Random(seed) if seed is not None else None

@typechecked
def data_dividing(datas:list,dividing_rates:tuple[float,...],random:random.Random|None=None,selected_all=True)->list[list]:
    assert all(x>=0 for x in dividing_rates)
    assert sum(dividing_rates)<=1.0
    data_range = list(range(len(datas)))
    if random is not None:
        random.shuffle(data_range)
    length = len(data_range)
    end_indexes = []
    for rate in dividing_rates:
        increase_indexes = int(length*rate)
        assert increase_indexes>1
        if not end_indexes:
            end_indexes.append(increase_indexes)
        else:
            end_indexes.append(end_indexes[-1]+increase_indexes)
    if selected_all: # if ture, the last dividing rate will be ignored and the last dividing part will contain all remaining elements
        end_indexes[-1]=length
    else:
        assert end_indexes[-1]<=length
    range_buf = []
    previous_index = 0
    for end_index in end_indexes:
        range_buf.append(sorted(data_range[previous_index:end_index]))
        previous_index = end_index
    return  [[datas[index] for index in ranges] for ranges in range_buf]