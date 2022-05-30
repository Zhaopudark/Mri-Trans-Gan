import itertools
import functools
import os
from matplotlib.pyplot import bar
from typeguard import typechecked
from dataclasses import dataclass,field
from enum import Enum
from typing import Generator,Callable,Any
import re
import collections
import numpy as np
import copy
from utils.dtype_helper import nested_dict_key_sort,gen_key_value_from_nested_dict,check_nested_dict
from utils.bar_helper import func_bar_injector
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

class BraTSBase(Enum):
    FILE_PATTERN = r'RSNA_ASNR_MICCAI_BraTS(?P<year>\d*)_(?P<training_type>Training|Validation)Data(?:\\{1}|[/]{1})(?P<patient_id>BraTS\d+_\d+)(?:\\{1}|[/]{1})(?P=patient_id)_(?P<modality>flair|t1ce|t1|t2)?(?:_)?(?P<info>\w*)?(?P<suffix>\.nii\.gz|\.csv|\.*)$'
    BASE_PATTERN = r'(?P<patient_id>BraTS\d+_\d+)(?:_)(?P<modality>flair|t1ce|t1|t2)(?P<suffix>\.nii\.gz)$' # used when gen path (by re.sub) from a base existed and correct path 
    REPL_BASE_PATTERN = r'\g<patient_id>_{}\g<suffix>' # used when gen path (by re.sub) from a base existed and correct path 
    KEY_NAMES = ('training_type','patient_id','modality','info')
    KEY_ORDERS = (('Training','Validation'),None,('flair','t1','t1ce','t2'),None)
    AXES_FORMAT = ('coronal', 'sagittal', 'vertical')
    AXES_DIRECTION_FORMAT = ("R:L", "A:P", "I:S")
    SIZE = (240,240,155)

@dataclass(slots=True)
class BraTSData():
    """
    A abstract data class, which contains the target data or datas
    """
    name:str=None
    keys: dict[str,tuple]=field(default_factory=dict) # dict of {identity:paths_collection_key}
    datas:dict[str,str]=field(default_factory=dict) # dict of {identity:paths_collection_value}
    def get_attrs(self)->dict[str,Any]:
        return {'name':copy.deepcopy(self.name),'keys':copy.deepcopy(self.keys),'datas':copy.deepcopy(self.datas)}
#-------------------------------------------------------------#
class BraTSDataPathCollection(collections.UserDict):
    """
    This class describe a paths collection of a dataset.

    1. load all file names (full_path) under a dataset folder
    2. use a `path_match_pattern`, according to specific dataset, to grab path's feature name from each full path
    3. use part of these feature names as level-by-level keys, make a n-level-nested dict and save each matched full_path
       the used feature names can be regarded as full_path's classification label

    when given a tuple of key,
        such as keys = (key1,key2,...)
        d = DataPathCollection() 
        d[keys] will return 
            a single value
            or
            a generator , which will gen all `keys,paths` as long as the key in keys are met

        if one of the key not in the established dict, it means the path is not saved in dict 
        so, the path will be inferred first by a `path_gen_pattern` and then returned 
    """
    __slots__ = ('_input_path','_path_match_pattern','_path_gen_pattern')
    #----------------------------------------------------#
    @typechecked
    def __init__(self,path:str) -> None:
        self._input_path = os.path.normpath(path)
        self._path_match_pattern = re.compile(BraTSBase.FILE_PATTERN.value)
        self._path_gen_pattern = re.compile(BraTSBase.BASE_PATTERN.value)
        super().__init__()
        self._load_paths(self._input_path)
    def _map_path2keyspath(self,path:str):# add specific rules in this func
        matched = self._path_match_pattern.search(path)
        if matched is not None:
            keys = [matched.group(item) for item in BraTSBase.KEY_NAMES.value] # list first
            #rules
            match keys:
                case [_,_,(None|''),_]: # modality is None or ''
                    keys[2] = 'shared'
                case [_,_,_,(None|'')]: # info is None or ''
                    keys[3] = 'main'
                case _:
                    pass
            return tuple(keys),path
        return None,path
    def _map_keys2path(self,keys:tuple[str,...]):# add specific rules in this func
        match list(copy.deepcopy(keys)): # base key rules
            case [training_type,patient_id,_,_]:
                base_key = tuple([training_type,patient_id,'t1','main'])
            case _:
                raise ValueError("") #TODO add some illustration
        try:
            base_path = self.__getitem_basic(base_key)
        except KeyError:
            raise KeyError(f"The base path of key `{base_key}` does not exist. So the path can not be inferred.")

        match keys: #rules
            case [_,_,'shared',info]:
                return self._path_gen_pattern.sub(BraTSBase.REPL_BASE_PATTERN.value,base_path).format(f"{info}")
            case [_,_,modality,info]:# not shared
                return self._path_gen_pattern.sub(BraTSBase.REPL_BASE_PATTERN.value,base_path).format(f"{modality}_{info}")
            case _:
                raise ValueError("")
    @func_bar_injector(bar_name="Path Loading...... ")
    def _load_paths(self,path:str,bar:Callable=None):
        for (dir_name,_,file_list) in os.walk(path):
            for file_name in file_list:
                keys,path = self._map_path2keyspath(os.path.join(dir_name,file_name))
                if keys is not None:
                    self[keys] = path
                if bar is not None:
                    bar()
        self.data = nested_dict_key_sort(self.data,BraTSBase.KEY_ORDERS.value,bar)
    @typechecked
    def __setitem__(self,key:tuple[str,...],item:str) -> None:
        inner_dict = self.data
        for i in range(len(key)-1):
            if key[i] not in inner_dict:
                inner_dict[key[i]] = {}
            inner_dict = inner_dict[key[i]]
        inner_dict[key[-1]] = item
    @typechecked
    def __getitem_basic(self,keys:tuple[str,...]) -> str:
        tmp_dict = self.data
        for key in keys:
            tmp_dict = tmp_dict[key]
        return tmp_dict     
    @typechecked
    def __getitem__(self,keys:tuple[None|int|str|tuple[str,...],...])->str|Generator[tuple[tuple[str,...],str],None,None]: # self[key]
        assert len(keys)==len(BraTSBase.KEY_NAMES.value)
        match keys:
            case tuple(matched_key) if all(isinstance(item,str) for item in matched_key):
                try:
                    return self.__getitem_basic(matched_key)
                except KeyError:
                    self[matched_key] = self._map_keys2path(matched_key)
                    return self.__getitem_basic(matched_key)
            case tuple(matched_key):
                return self.generator(keys)
            case _:
                raise ValueError("") #TODO add some illustration
    @typechecked
    @func_bar_injector(bar_name="Path Checking......")
    def _data_checking(self,keys:tuple[int|str|None|tuple[str,...],...],bar:Callable=None):
        check_nested_dict(self.data,keys,previous_keys=[],value_func=self._map_keys2path,bar=bar)
        self.data = nested_dict_key_sort(self.data,BraTSBase.KEY_ORDERS.value,bar)
    @typechecked
    def generator(self,keys:tuple[int|str|None|tuple[str,...],...])->Generator[tuple[tuple[str,...],str],None,None]:
        self._data_checking(keys)
        return gen_key_value_from_nested_dict(self.data,keys) 
    @typechecked
    def get_individual_datas(self,individual_identities:str,shared_identities:list[str]=None)->list[BraTSData]:
        """
        get_individual_datas of an individual_identities
        add extra datas by shared_identities
        """
        if shared_identities is None:
            shared_identities = []
        searching_key_list = [(None,None,('t1','t2','t1ce','flair'),individual_identities)]+[(None,None,('shared',),shared_identity) for shared_identity in shared_identities]
        # searching_key is used for get `key,value` results in DataPathCollection
        storing_key_list = [('t1','t2','t1ce','flair')]+[(shared_identity,) for shared_identity in shared_identities]
        # storing_key is used as identities for storing searched `key,value` into BraTSData's keys (a dict), datas (a dict) respectively   
        name_groups_list = [itertools.groupby(self[item],key=lambda kv:'_'.join(list(kv[0])[:2])) for item in searching_key_list]
        # `name_groups_list` in type List[Iterable[tuple[name,group]]], in length len(keys_list) | len(identities_list) 
        # `name` in type str
        # `group` in type tuple[tuple[tuple[str,...],str],...] since a single element of self.generator's return is tuple[tuple[str,...],str]
        datas_list  = []
        for name_groups in zip(*name_groups_list): # name_groups is List[tuple[name,group]] in length len(keys_list) | len(identities_list)
            iterable = iter(zip(name_groups,storing_key_list))
            (name,group),identities = next(iterable)
            keys,values = zip(*group) # `keys`in type tuple[tuple[str, ...],...], `values` in type tuple[str,...],
            individual_data = BraTSData(name,dict(zip(identities,keys)),dict(zip(identities,values)))
            for (name,group),identities in iterable:
                keys,values = zip(*group)
                assert individual_data.name == name               
                individual_data.keys |= dict(zip(identities,keys))
                individual_data.datas |= dict(zip(identities,values))
            datas_list.append(individual_data)
        return datas_list
    @classmethod
    @typechecked
    def reduce_datas(cls,datas_list:list[BraTSData|dict[str,Any]],reduce_func:Callable=None)->dict[str,list[Any]|None]:
        """
        reduce list of dict like datas into just one dict
        Args:
            datas_list: list[BraTSData] or list[dict[str,Any]], if datas_list is in type list[BraTSData], 
                        we will focus on each elements's datas, which in type dict[str,Any],
                        else, we just focus on elements of datas_list, which in type dict[str,Any] as well.
                        Each element we focused should have the same keys.( If not, it's better to realize by `reducing`, But not here)
            reduce_func: Values with the same key in the data_list's element are collected into a list as a new value of a new dictionary, their key is the new value's key.
                        Then, we got a combined result in type dict[str,list[Any]].
                        If reduce_func is specified, we apply it as reduce function on each list (new value) of the a combined result, return the reduced result.
                        If reduce_func is None, just return the combined result.
                        The reduce func should support the using form functools.reduce(reduce_func,list[Any],None)
        Return:
            a may be reduced combined result in type dict[str,list[Any]]

        inverse_reduce_datas is the inverse operation
        if x = list(inverse_reduce_datas(reduce_datas(datas)))
           y = list(inverse_reduce_datas(reduce_datas(x)))
        there will be x===y
        if reduce_func is set, we cannot guarantee the reversibility of `inverse_reduce_datas` and `reduce_datas`
        """
        if all(isinstance(item,BraTSData) for item in datas_list):
            datas_list = [item.datas for item in datas_list]
        elif all(isinstance(item,dict) for item in datas_list):
            pass 
        else:
            raise ValueError("All datas in datas_list should have the same type and be one of BraTSData or dict's instance.")
        keys_values = zip(*[datas.items() for datas in datas_list])
        # keys_values in type tuple[tuple[tuple[key,value],...],...] such as ((('t1',value1),('t1',value2),...),(('t2',value1),('t2',value2),...),...)
        buf = {}
        for key_values in keys_values:
            # key_values in type tuple[tuple[key,value],...] such as (('t1',value1),('t1',value2),...)
            key,values = zip(*key_values)
            def check_func(x1,x2):
                assert x1==x2
                return x1
            key = functools.reduce(check_func,key)
            buf[key] = list(values) # buf |= {key:list(values)}python 3.10 supported
        if reduce_func is not None:
            for key,value in buf.items():
                buf[key] = reduce_func(value)
        return buf    
    @classmethod
    @typechecked
    def inverse_reduce_datas(cls,datas:dict[str,list[Any]])->Generator[dict[str,Any],None,None]:
        """
        inverse operation of reduce_datas
        but just return Generator of dicts, i.e., the BraTSData.datas, not the whole BraTSData
        if x = list(inverse_reduce_datas(reduce_datas(datas)))
           y = list(inverse_reduce_datas(reduce_datas(x)))
        there will be x===y
        """
        keys = datas.keys()
        values = zip(*datas.values())
        for value in values:
            yield dict(zip(keys,value))



# data_path_collection = BraTSDataPathCollection(path="D:\\Datasets\\BraTS\\BraTS2021_new")
# input_datas = data_path_collection.get_individual_datas('main',['masksss'])
# length1 = len(input_datas)
# input_datas = BraTSDataPathCollection.reduce_datas(input_datas)
# def check(x1,x2):
#     assert x1==x2 
#     return x2
# length2 = functools.reduce(check,map(len,input_datas.values()))
# mask_paths = input_datas.pop('masksss')
# mask_paths = BraTSDataPathCollection.inverse_reduce_datas({'masksss':mask_paths})
# assert len(list(mask_paths)) == length2 == length1

# data_path_collection = BraTSDataPathCollection(path="D:\\Datasets\\BraTS\\BraTS2021_new")
# input_datas = data_path_collection.get_individual_datas('main',['mask'])
# output_datas = data_path_collection.get_individual_datas('brain_mask',['segggg'])
# x = list(BraTSDataPathCollection.inverse_reduce_datas(BraTSDataPathCollection.reduce_datas(input_datas[:])))
# y = list(BraTSDataPathCollection.inverse_reduce_datas(BraTSDataPathCollection.reduce_datas(x)))
# for item1,item2 in zip(x,y):
#     assert item1 == item2


# for inputs,outputs in zip(input_datas,output_datas):
#     assert inputs.name==outputs.name
# input_datas = BraTSDataPathCollection.reduce_datas(input_datas)
# _ = input_datas.pop('mask')
# input_datas = BraTSDataPathCollection.inverse_reduce_datas(input_datas)
# output_datas = BraTSDataPathCollection.reduce_datas(output_datas)
# _ = output_datas.pop('segggg')
# output_datas = BraTSDataPathCollection.inverse_reduce_datas(output_datas)
# for inputs,outputs in zip(input_datas,output_datas):
#     for (k1,v1),(k2,v2) in zip(inputs.items(),outputs.items()):
#         assert k1==k2 