import ast
from typing import Iterable,Literal,Sequence,Any
from typeguard import typechecked
import numpy as np
import tensorflow as tf

from utils.managers import RecordManager
from utils.operations import read_nii_file,combine2patches
from . import bratsbase

class BraTSMapping():
    """
    Abstract Data
    """
    @typechecked
    def __init__(self,
        axes_format:tuple[Literal["vertical","sagittal","coronal"],...],
        axes_direction_format:tuple[Literal["S:I","A:P","R:L"],...],
        record_path:str) -> None:
        self._axes_format = axes_format # user desired
        self._axes_direction_format = axes_direction_format  # user desired
        self._record_manager = RecordManager(record_path)
    @property
    def transpose_permutation(self):
        # get transpose_permutation by compare _axes_format with AXES_FORMAT
        if not hasattr(self,"_transpose_permutation"):
            self._transpose_permutation = tuple(bratsbase.AXES_FORMAT.index(item) for item in self._axes_format)
        return self._transpose_permutation
    @property
    def size_range(self): # the size after permutation
        if not hasattr(self,"_size"):
            self._size = tuple([0,bratsbase.SIZE[index]-1] for index in self.transpose_permutation)
        return self._size
    @property
    def flip_axes(self):
        # get flip_axes by compare _axes_direction_format with transposed AXES_DIRECTION_FORMAT
        if not hasattr(self,"_flip_axes"):
            transposed_direction_format = tuple(bratsbase.AXES_DIRECTION_FORMAT[index] for index in self.transpose_permutation)
            def filter_func(index):
                normal_direction = transposed_direction_format[index]
                fliped_direction = transposed_direction_format[index][::-1]
                if self._axes_direction_format[index] == normal_direction:
                    return False # need not flip
                elif self._axes_direction_format[index] == fliped_direction:
                    return True # need flip to match _axes_direction_format
                else:
                    ValueError(f"{self._axes_direction_format[index]} should in `{normal_direction} or {fliped_direction}`")
            self._flip_axes = tuple(filter(filter_func,range(len(self._axes_direction_format))))
        return self._flip_axes
    def _pre_process(self,img:np.ndarray)->np.ndarray:# transpose [Z,Y,X] tensor to [-X,Y,Z]
        img = img.transpose(self.transpose_permutation)
        return np.flip(img,axis=self.flip_axes)  
    def read_from_path(self,path:str)->np.ndarray:
        img,_,_ = read_nii_file(path,dtype=np.float32)
        return self._pre_process(img)
    
    def mapping_whole(self,data:dict[str,tf.Tensor])->dict[str,tf.Tensor]:
        keys = tuple(data.keys())[1::]
        values = tuple(data.values())
        Tout = tuple([tf.TensorSpec(shape=[],dtype=tf.float32),]*len(keys))
   
        def func(*inputs):
            name = ast.literal_eval(str(inputs[0].numpy(),encoding='UTF-8'))
            info = tuple(ast.literal_eval(str(item.numpy(),encoding='utf-8')) for item in inputs[1::])
            def gen_func()->Sequence[tf.Tensor]:
                return tuple(map(self.read_from_path,info))
            return self._record_manager.load(name=name,keys=keys,gen_func=gen_func)
        return dict(zip(keys,tf.py_function(func,inp=[*values],Tout=Tout)))
    def mapping_patches(self,data:dict[str,tf.Tensor])->dict[str,tf.Tensor]:
        keys = tuple(data.keys())[1::]
        values = tuple(data.values())
        keys_for_return = [] 
        for key in keys:
            keys_for_return.extend((f"{key}",f"{key}_ranges"))
        Tout = tuple([tf.TensorSpec(shape=[],dtype=tf.float32),tf.TensorSpec(shape=[],dtype=tf.int32)]*len(keys))   
        def func(*inputs):
            name = ast.literal_eval(str(inputs[0].numpy(),encoding='UTF-8'))
            info = tuple(ast.literal_eval(str(item.numpy(),encoding='utf-8')) for item in inputs[1::])
            paths,ranges = zip(*info)
            keys_for_store = []
            for key,single_ranges in zip(keys,ranges):
                keys_for_return.extend((f"{key}",f"{key}_ranges"))
                keys_for_store.extend((f"{key}_{single_ranges}",f"{key}_{single_ranges}_ranges"))          
            def gen_func()->Sequence[tf.Tensor]:
                outs = []
                for path,single_ranges in zip(paths,ranges):
                    data = self.read_from_path(path)
                    slices = tuple(map(lambda x:slice(x[0],x[1]+1),single_ranges))
                    outs.extend((data[slices],tf.convert_to_tensor(single_ranges,dtype=tf.int32)))
                return tuple(outs)
            return self._record_manager.load(name=name,keys=keys_for_store,gen_func=gen_func)
        return dict(zip(keys_for_return,tf.py_function(func,inp=values,Tout=Tout)))
    
    @classmethod
    def zip_dict_data(cls,data:dict[str,tf.Tensor]):
        key_buf = []
        data_buf = []
        ranges_buf = []
        mask_buf = []
        for key,value in data.items():
            if key.endswith('ranges'):
                ranges_buf.append(value) 
            else:
                key_buf.append(key)
                data_buf.append(value)
                mask_buf.append(tf.ones_like(value,dtype=tf.int32))
        yield from zip(key_buf,data_buf,ranges_buf,mask_buf)
    @classmethod
    def unzip_dict(cls,data:Iterable[Iterable[Any]]):
        buf = {}
        for (kx,x,xr,xm) in data:
            _xm = tf.cast(tf.maximum(xm,1),x.dtype)
            x = tf.divide(x,_xm)
            tf.debugging.assert_all_finite(x,message="assert_all_finite",name=None)
            tf.debugging.assert_type(x,tf.float32,message=None,name=None)
            tf.debugging.assert_type(xr,tf.int32,message=None,name=None)
            buf[kx] = x
            buf[f"{kx}_ranges"] = xr
        return buf

    @classmethod
    def stack_patches(cls,datas:Iterable[dict[str,tf.Tensor]],stack_count=None)-> dict[str,tf.Tensor]:    
        base = None
        for i,data in enumerate(datas):
            if base is None:
                base = cls.zip_dict_data(data)
            else:# combine2patches
                current = cls.zip_dict_data(data)
                buf = [combine2patches(ka,a,ar,am,kb,b,br,bm) for (ka,a,ar,am),(kb,b,br,bm) in zip(base,current)]
                base = buf
            if (i+1)%stack_count == 0:
                yield cls.unzip_dict(base)
                base = None
        if base is not None:
            yield cls.unzip_dict(base)  