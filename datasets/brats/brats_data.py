import os
import itertools
import functools
from typing import Callable, Iterable

from typeguard import typechecked

from utils.types import DictList,ListContainer
from utils.types import NestedDict
from utils.managers import func_bar_injector
from . import bratsbase

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

class BraTSDataPathCollector():
    """
    This class describe a paths collection of a dataset.

    1. load all file names (full_path) under a dataset folder
    2. use a `path_match_pattern`, according to specific dataset, to grab path's feature name from each full path
    3. use part of these feature names as level-by-level keys, make a n-level-nested dict and save each matched full_path
       the used feature names can be regarded as full_path's classification label
    """
    @typechecked
    def __init__(self,path:str) -> None:
        self._input_path = os.path.normpath(path)
        self.data = NestedDict()
        self._load_paths(self._input_path)
    @func_bar_injector(bar_name='loading paths...')
    def _load_paths(self,path:str,bar:Callable=None):
        for (dir_name,_,file_list) in os.walk(path):
            for file_name in file_list:
                if bar is not None:
                    bar()
                tags = bratsbase.get_tags_from_path(os.path.join(dir_name,file_name))
                if tags is not None:
                    self.data.append(tags,os.path.join(dir_name,file_name))
        self.data.sort(key_orders=bratsbase.TAGS_ORDERS)
    @typechecked
    @func_bar_injector(bar_name='getting datas...')
    def get_individual_datas(self,tags:tuple[tuple[str,...]|None,...]=None,should_existed=True,bar:Callable=None):
        """
        get_individual_datas of an individual_identities
        add extra datas by shared_identities
        """
        if should_existed:
            tag_datas = self.data.get_items(None,tags,None,bar)
        else:
            def estimate_func(tags):
                base_tags = bratsbase.get_base_tags_from_tags(tags)
                base_path = self.data[base_tags].data
                return bratsbase.gen_path_from_tags(tags,base_path)

            tag_datas = self.data.get_items(None,tags,estimate_func,bar)
        stamp_tag_datas = itertools.groupby(tag_datas,key=lambda tag_data:bratsbase.gen_stamp_from_tags(tag_data[0]))
        buf = []
        for stamp,tag_datas in stamp_tag_datas:
            if bar is not None:
                bar()
            data = {bratsbase.gen_key_tag_from_tags(tag_data[0]):tag_data[-1] for tag_data in tag_datas}
            buf.append(BraTSData(stamp,data))
        return buf