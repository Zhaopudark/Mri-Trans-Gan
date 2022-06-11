import os
from typing import Iterable,Callable,Generator,Literal,Any,Sequence,get_type_hints
import operator
from typeguard import typechecked
import itertools
import functools
import tensorflow as tf 
import numpy as np
import re
import copy
from utils.dataset_helper import read_nii_file,data_dividing,random_datas,get_random_from_seed
from utils.dtype_helper import reduce_same
from utils.image.patch_process import index_cal
from datasets.brats.brats_data import BraTSDataPathCollection
from datasets.brats.bratsbase import BraTSData



class BasePipeLine():
    def __init__(self,datas:list[Any]) -> None:
        self.raw_datas = datas # the concerned and processed object
    @property
    def datas(self):
        if not hasattr(self,"_datas"):
            self._datas = self.process(self.raw_datas)
        return self._datas
    def process(self,datas)->list:
        return datas
    def reduce(self,datas):
        raise NotImplementedError
    def __call__(self): 
        return self.reduce(self.datas)

class BraTSBasePipeLine(BasePipeLine):
    """
    """
    @typechecked
    def __init__(self,datas:list[BraTSData]) -> None:
        super().__init__(datas)
    def reduce(self,datas:list[BraTSData]):
        return BraTSData.reduce(datas).serialize()

class BraTSDividingWrapper(BasePipeLine):
    """
    """
    @typechecked  
    def __init__(self,pipeline:BraTSBasePipeLine,
                    dividing_rates:tuple[float,...]=None,
                    dividing_seed:None|int=None):
        self._pipeline = pipeline
        self._dividing_rates = dividing_rates
        self._dividing_seed = dividing_seed
        super().__init__(self._pipeline.raw_datas)
    def process(self,datas)->tuple[list[BraTSData],...]:
        return tuple(data_dividing(datas,dividing_rates=self._dividing_rates,random=get_random_from_seed(self._dividing_seed),selected_all=True))
    def reduce(self,datas:tuple[list[BraTSData],...]):
        return tuple(map(self._pipeline.reduce,datas))

class BraTSPatchesWrapper(BasePipeLine):
    """
    """
    @typechecked  
    def __init__(self,pipeline:BraTSBasePipeLine|BraTSDividingWrapper,
                cut_ranges:tuple[tuple[int,int],...],
                patch_sizes:tuple[int,...],
                patch_nums:tuple[int,...]):
        self._pipeline = pipeline
        self._cut_ranges = cut_ranges
        self._patch_sizes = patch_sizes
        self._patch_nums = patch_nums
        assert len(cut_ranges)==len(patch_sizes)==len(patch_nums)
        super().__init__(self._pipeline.raw_datas)
    @property
    def patch_ranges(self)->tuple[tuple[tuple[int,int],...]]:
        if not hasattr(self,"_patch_ranges"):
            patch_ranges_info = tuple(map(index_cal,self._cut_ranges,self._patch_sizes,self._patch_nums))
            self._patch_ranges = tuple(itertools.product(*patch_ranges_info)) 
        return self._patch_ranges
    
    def process(self,datas)->list[BraTSData]|tuple[list[BraTSData],...]:
        """
        """
        processed_datas = self._pipeline.process(datas)
        if isinstance(processed_datas,list): # BraTSBasePipeLine
            processed = [(data+patch_ranges) for data,patch_ranges in itertools.product(processed_datas,self.patch_ranges)]
        elif isinstance(processed_datas,tuple):
            processed = tuple([(data+patch_ranges) for data,patch_ranges in itertools.product(datas,self.patch_ranges)] for datas in processed_datas)
        else:
            raise ValueError(" ") # TODO
        return processed
    def reduce(self,datas):
        return self._pipeline.reduce(datas)