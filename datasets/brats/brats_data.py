import itertools
import functools
import os
from matplotlib.pyplot import bar
from typeguard import typechecked
from dataclasses import dataclass,field
from enum import Enum
from typing import Generator,Callable,Any, Literal
import re
import collections
import numpy as np
import copy
from datasets.brats import bratsbase

from utils.dtype_helper import NestedDict
from utils.bar_helper import func_bar_injector

#-------------------------------------------------------------#
class BraTSDataPathCollection():
    """
    This class describe a paths collection of a dataset.

    1. load all file names (full_path) under a dataset folder
    2. use a `path_match_pattern`, according to specific dataset, to grab path's feature name from each full path
    3. use part of these feature names as level-by-level keys, make a n-level-nested dict and save each matched full_path
       the used feature names can be regarded as full_path's classification label
    """
    #----------------------------------------------------#
    @typechecked
    def __init__(self,path:str) -> None:
        self._input_path = os.path.normpath(path)
        self.data = NestedDict()
        self._load_paths(self._input_path)
    def _load_paths(self,path:str):
        for (dir_name,_,file_list) in os.walk(path):
            for file_name in file_list:
                tags = bratsbase.get_tags_from_path(os.path.join(dir_name,file_name))
                if tags is not None:
                    self.data.append(tags,os.path.join(dir_name,file_name))
        self.data.sort(key_orders=bratsbase.TAGS_ORDERS)
    @typechecked
    def get_individual_datas(self,tags:tuple[tuple[str,...]|None,...]=None,should_existed=True):
        """
        get_individual_datas of an individual_identities
        add extra datas by shared_identities
        """
        if should_existed:
            tag_datas = self.data.get_items(None,tags,None)
        else:
            def estimate_func(tags):
                base_tags = bratsbase.get_base_tags_from_tags(tags)
                base_path = self.data[base_tags].data
                return bratsbase.gen_path_from_tags(tags,base_path)

            tag_datas = self.data.get_items(None,tags,estimate_func)
        stamp_tag_datas = itertools.groupby(tag_datas,key=lambda tag_data:bratsbase.gen_stamp_from_tags(tag_data[0]))
        buf = []
        for stamp,tag_datas in stamp_tag_datas:
            data = {bratsbase.gen_key_tag_from_tags(tag_data[0]):tag_data[-1] for tag_data in tag_datas}
            buf.append(bratsbase.BraTSData(stamp,data))
        return buf
        # if shared_identities is None:
        #     shared_identities = []
        # searching_key_list = [(None,None,('t1','t2','t1ce','flair'),individual_identities)]+[(None,None,('shared',),shared_identity) for shared_identity in shared_identities]
        # # searching_key is used for get `key,value` results in DataPathCollection
        # storing_key_list = [('t1','t2','t1ce','flair')]+[(shared_identity,) for shared_identity in shared_identities]
        # # storing_key is used as identities for storing searched `key,value` into BraTSData's keys (a dict), datas (a dict) respectively   
        # name_groups_list = [itertools.groupby(self[item],key=lambda kv:'_'.join(list(kv[0])[:2])) for item in searching_key_list]
        # # `name_groups_list` in type List[Iterable[tuple[name,group]]], in length len(keys_list) | len(identities_list) 
        # # `name` in type str
        # # `group` in type tuple[tuple[tuple[str,...],str],...] since a single element of self.generator's return is tuple[tuple[str,...],str]
        # datas_list  = []
        # for name_groups in zip(*name_groups_list): # name_groups is List[tuple[name,group]] in length len(keys_list) | len(identities_list)
        #     iterable = iter(zip(name_groups,storing_key_list))
        #     (name,group),identities = next(iterable)
        #     keys,values = zip(*group) # `keys`in type tuple[tuple[str, ...],...], `values` in type tuple[str,...],
        #     individual_data = BraTSData(name,dict(zip(identities,keys)),dict(zip(identities,values)))
        #     for (name,group),identities in iterable:
        #         keys,values = zip(*group)
        #         assert individual_data.name == name               
        #         individual_data.keys |= dict(zip(identities,keys))
        #         individual_data.datas |= dict(zip(identities,values))
        #     datas_list.append(individual_data)
        # return datas_list
    


