"""
Consider here are original BraTS datas' paths:
    t1
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1.nii.gz
    t2
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t2.nii.gz
    flair
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_flair.nii.gz
    t1ce
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1ce.nii.gz
    seg (if exist)
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_seg.nii.gz
    ...
Usually we need to preprocess the above files to get our desired ones. So, naming the preprocessed 
file is inevitable. Therefore, we make the following naming convention:

files from or for single modality, such as:
    some files got form t1 modality {some discription} such as {brain} {brain_mask} {...}
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1_{some discription}_.nii.gz
    ...
files made for all modality, such as:
    `mask` that indicate concerned regions on all registrated modalitis
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_mask_.nii.gz

So, we define `mask` `seg` as `shared` modality (Can be shared by the other modalities),
and attach 4 main TAGS type to BraTS Datas:

    training_type:  'Training'  or 'Validation'
    patient_id:     'BraTS2021_00000' 'BraTS2021_00002' ...
    modality:       'flair','t1','t1ce','t2','shared'
    info:           'main' 'norm' 'mask' 'seg' ...

Only `training_type` and `modality` have limited tags. We use tuple to 
confirm an order of TAGS, ('training_type','patient_id','modality','info')

TAGS ('Training','BraTS2021_00000','t1','main') 
    indicate to
    original `t1` modality file of patient `BraTS2021_00000`, i.e.,
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1.nii.gz

TAGS ('Training','BraTS2021_00000','t1','brain') 
    indicate to
    user difined `t1` modality's `brain` file of patient `BraTS2021_00000`, i.e.,
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1_brain.nii.gz

TAGS ('Training','BraTS2021_00000','t1','norm') 
    indicate to
    user difined `t1` modality's `norm` file of patient `BraTS2021_00000`, i.e.,
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1_norm.nii.gz

TAGS ('Training','BraTS2021_00000','shared','seg') 
    indicate to
    original `shared` modality's `seg` file of patient `BraTS2021_00000`, i.e.,
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_seg.nii.gz

TAGS ('Training','BraTS2021_00000','shared','mask') 
    indicate to
    user difined `shared` modality's `mask` file of patient `BraTS2021_00000`, i.e.,
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_mask.nii.gz
"""
import os
import re 
import ast
import operator
import functools
import itertools

from typing import Iterable,Literal,Callable,Sequence,Any

from typeguard import typechecked
import numpy as np
import tensorflow as tf
from utils.types import DictList,ListContainer
from utils.managers import RecordManager
from utils.operations import read_nii_file
#-----------------------------------config for BraTS------------------------------------------#
FILE_PATTERN = r'.*RSNA_ASNR_MICCAI_BraTS(?P<year>\d*)_(?P<training_type>Training|Validation)Data(?:\\{1}|[/]{1})(?P<patient_id>BraTS\d+_\d+)(?:\\{1}|[/]{1})(?P=patient_id)_(?P<modality>flair|t1ce|t1|t2)?(?:_)?(?P<info>\w*)?(?P<suffix>\.nii\.gz|\.csv|\.*)'
BASE_PATTERN = r'(?P<prefix>.*)(?P<patient_id>BraTS\d+_\d+)(?:_)(?P<modality>flair|t1ce|t1|t2)(?P<suffix>\.nii\.gz)' # used when gen path (by re.sub) from a base existed and correct path 
REPL_BASE_PATTERN = r'\g<prefix>\g<patient_id>_{}\g<suffix>' # used when gen path (by re.sub) from a base existed and correct path 
TAGS = ('training_type','patient_id','modality','info')
TAGS_ORDERS = (('Training','Validation'),None,('flair','t1','t1ce','t2'),None) # used for sort
AXES_FORMAT = ('coronal', 'sagittal', 'vertical')
AXES_DIRECTION_FORMAT = ("R:L", "A:P", "I:S")
SIZE = (240,240,155)
#-----------------------------------functions for BraTS------------------------------------------#
__path_load_prog = re.compile(FILE_PATTERN)
__path_gen_prog = re.compile(BASE_PATTERN)

def get_tags_from_path(path:str):
    _result = __path_load_prog.match(path)
    if _result:
        result = [_result.group(tag_name) for tag_name in TAGS]
    else:
        result = None
    # result = Tags(__path_load_prog.match(path))
    match result:
        case None:
            pass 
        case [_,_,(None|''),_]:
            result[2] = 'shared'
        case [_,_,_,(None|'')]: # info is None or ''
            result[3] = 'main'
        case _:
            pass
    return result

def gen_stamp_from_tags(tags:tuple[str,...]):
    match tags: #rules
        case [training_type,patient_id,_,_]:
            return f"{training_type}_{patient_id}"
        case _:
            raise ValueError(" ") # TODO
def gen_key_tag_from_tags(tags:tuple[str,...]):
    match tags: #rules
        case [_,_,'shared',info]:
            return info
        case [_,_,modality,_]: # [_,_,modality,'main'] or [_,_,modality,other]
            return modality
        case _:
            raise ValueError(" ") # TODO
def get_base_tags_from_tags(tags:tuple[str,...]):
    match tags: #rules
        case [training_type,patient_id,_,_]:
            return (training_type,patient_id,'t1','main')
        case _:
            raise ValueError(" ") # TODO
def gen_path_from_tags(tags:tuple[str,...],base_path:str):
    match tags: #rules
        case [_,_,'shared',info]:
            return __path_gen_prog.sub(REPL_BASE_PATTERN,base_path).format(f"{info}")
        case [_,_,modality,info]:# not shared
            return __path_gen_prog.sub(REPL_BASE_PATTERN,base_path).format(f"{modality}_{info}")
        case _:
            raise ValueError(" ") # TODO
#-------------------------------------------------------------#
def is_affine_euqal(affine1,affine2):
    # 依据brats文件的特性而定义, 比较两个 affine 是否相等
    return (affine1==affine2).all()

def is_header_euqal(header1,header2):
    # 依据brats文件的特性而定义, 比较两个 header 是否相等
    for (key0,value0),(key1,value1) in zip(header1.items(),header2.items()):
        if key0 != key1:
            return False
        dtype_list = [np.uint8,np.uint16,np.uint32,np.int8,np.int16,np.int32,np.float32]
        if (value0.dtype in dtype_list)and(value1.dtype in dtype_list):
            if (np.isclose(value0,value1,equal_nan=True)).all():
                continue
        elif (value0==value1).all():
            continue
    return True



# def tf_py_function_wrapper(func=None):
#     # since tf.py_function can not deal with dict directly, and its using form is not easy
#     # here we make this wrapper, it can trans `func`'s output 
#     # all tensors that a user want to use by calling their `numpy()` functions should become the inputs of the wrapped `func`, otherwise, `numpy()` will not work
#     # since tf.nest.flatten  tf.nest.pack_sequence_as will sort dict structure's `keys` automaticlly, we do not use tf.nest here to avoid unexpected behavior
#     if func is None:
#         return functools.partial(tf_py_function_wrapper,)
#     @functools.wraps(func)
#     def wrappered(inputs:dict[str,tf.Tensor],output_structure:dict[str,tf.TensorSpec])->dict[str,tf.Tensor]:
#         inp = tuple(inputs.values())
#         Tout = tuple(output_structure.values())
#         flattened_output = tf.py_function(func,inp=inp,Tout=Tout)
#         return dict(zip(output_structure.keys(),flattened_output))
#     return wrappered    

class BraTSMapping():
    """
    Abstract Data
    """
    def __init__(self,
        axes_format:tuple[Literal["vertical","sagittal","coronal"],...]=("vertical","sagittal","coronal"),
        axes_direction_format:tuple[Literal["S:I","A:P","R:L"],...]=("S:I","A:P","R:L"),
        record_path = "D:\\Datasets\\BraTS\\BraTS2021_new\\records") -> None:
        self._axes_format = axes_format # user desired
        self._axes_direction_format = axes_direction_format  # user desired
        self._record_manager = RecordManager(record_path)
    @property
    def transpose_permutation(self):
        # get transpose_permutation by compare _axes_format with AXES_FORMAT
        if not hasattr(self,"_transpose_permutation"):
            self._transpose_permutation = tuple(AXES_FORMAT.index(item) for item in self._axes_format)
        return self._transpose_permutation
    @property
    def size_range(self): # the size after permutation
        if not hasattr(self,"_size"):
            self._size = tuple([0,SIZE[index]-1] for index in self.transpose_permutation)
        return self._size
    @property
    def flip_axes(self):
        # get flip_axes by compare _axes_direction_format with transposed AXES_DIRECTION_FORMAT
        if not hasattr(self,"_flip_axes"):
            transposed_direction_format = tuple(AXES_DIRECTION_FORMAT[index] for index in self.transpose_permutation)
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
        Tout = tuple([tf.TensorSpec(shape=[],dtype=tf.float32),]*(len(keys)-1))
   
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
        Tout = tuple([tf.TensorSpec(shape=[],dtype=tf.float32),tf.TensorSpec(shape=[],dtype=tf.int32)]*(len(keys)-1))   
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
            _xm = tf.maximum(xm,1)
            x = tf.divide(x,_xm)
            tf.debugging.assert_all_finite(x)
            tf.debugging.assert_type(x,tf.float32,message=None,name=None)
            tf.debugging.assert_all_finite(xr)
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
                buf = [cls.combine2patches(ka,a,ar,am,kb,b,br,bm) for (ka,a,ar,am),(kb,b,br,bm) in zip(base,current)]
                base = buf
            if (i+1)%stack_count == 0:
                yield cls.unzip_dict(base_data)
                base_data = None
        yield cls.unzip_dict(base_data)  
        
class BraTSData(): # single path
    __slots__ = ['name','data']
    def __init__(self,name:str,data:dict) -> None:
        
        if isinstance(name,(str,ListContainer)):
            self.name = ListContainer(name)
        else:
            raise ValueError(" ") # TODO
        if isinstance(data,(dict,DictList)):
            self.data = DictList(data)
        else:
            raise ValueError(" ") # TODO
    def __add__(self,other): # not in-place
        if isinstance(other,self.__class__):
            return self.__class__(self.name+other.name,self.data+other.data)
        else:
            return self.__class__(self.name,self.data+other)
    def __radd__(self,other):
        raise NotImplementedError
    def __iadd__(self,other):
        raise NotImplementedError
    def serialize(self): # not in-place
        # return the most fundamental information, discard other unnecessary custom data structures
        return {'name':self.name.serialize(),**self.data.serialize()}
    @staticmethod
    def _reduce_func(x,y):
        assert isinstance(x,__class__)
        assert isinstance(y,__class__)
        return x+y
    @classmethod 
    def reduce(cls,datas:Iterable):
        return  functools.reduce(cls._reduce_func,datas)

# class Tags():
#     __match_args__ = TAGS
#     def __init__(self,match_result:Match|tuple):
#         if match_result is None:
#             for attr_name in TAGS:
#                 self.__setattr__(attr_name,None)
#         elif isinstance(match_result,Match):
#             for attr_name in TAGS:
#                 self.__setattr__(attr_name,match_result.group(attr_name))
#         elif isinstance(match_result,tuple):
#             for attr_name,attr_value in zip(TAGS,match_result):
#                 self.__setattr__(attr_name,attr_value)
#         else:
#             raise ValueError(" ") # TODO



# path = "D:\\Datasets\\BraTS\BraTS2021_new\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_flair_z_score_and_min_max_norm.nii.gz"
# print(get_tags_from_path(path))

# print(__path_gen_prog.sub(REPL_BASE_PATTERN,'D:\\Datasets\\BraTS\\BraTS2021_new\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1.nii.gz').format(f"{1}_{2}"))
# with open(".\\datasets\\brats\\brats_config.yaml", 'r', encoding="utf-8")as f:
#     logging_yaml = yaml.load(stream=f, Loader=yaml.SafeLoader)
#     print(logging_yaml['TAGS'])
#     print(type(logging_yaml['TAGS']))