import copy
import functools
import itertools
from typing import Literal,Callable
from grapheme import length

import tensorflow as tf
from typeguard import typechecked 

from ..operations.random_op import get_random_from_seed,random_datas
from ..types.process_dtype import get_dict_list_length
# class DataIter():
#     """
#     return an datas Iterator that 
#         can control random behavior with an inner and individual random.Random() instance
#         can iter form the latest counters (step epoch) 
#     Args:
#         datas: list of data that can be 'shuffle()' by random.Random()
#         counters: {"step":step,"epoch":epoch}, contain the quote of global step and epoch
#         seed: random seed, `None` means do not random
#     Method:
#         __iter__(): 
#             It will backtrack status from the original dataset by counters before delivering elements.
#             Since random.Random().shuffle() will change status after each application, 
#             we should backtrack status from scratch everytime.
#     """
#     def __init__(self,datas:list,counters:dict[Literal["step","epoch"],tf.Variable]|None=None,seed:int|None=None) -> None:
#         if counters is None:
#             counters = {"step":0,"epoch":0}
#         self._epoch = counters["epoch"] # have passed epoch
#         self._step = counters["step"] # have passed step
#         self._datas = copy.deepcopy(datas) # do not influence original data, fixing the data
#         self.seed = seed
#         self._check()
#     def _check(self):
#         assert (self.step//self.length)==self.epoch
#     @property
#     def length(self):
#         return len(self._datas)
#     @property
#     def epoch(self):
#         return self._epoch.numpy() if hasattr(self._epoch,"numpy") else self._epoch
#     @property
#     def step(self):
#         return self._step.numpy() if hasattr(self._step,"numpy") else self._step
#     def __iter__(self):
#         datas = copy.deepcopy(self._datas) # do not influence the fixed data
#         random = get_random_from_seed(self.seed) 
#         for _ in range(self.epoch): # random from scratch
#             datas = random_datas(datas,random)
#         # self.step%self.length do not need 'minus 1', because it is the start index exactly
#         return itertools.islice(datas,self.step%self.length,self.length) 
#     def __repr__(self) -> str:
#         return f"Static indices is epoch:{self.epoch} step:{self.step}."
class DataIter():
    @typechecked
    def __init__(self,datas:list|dict[str,list],datas_wrapper:Callable=lambda x:x):
        self._datas = copy.deepcopy(datas) # do not influence original data, fixing the data
        self._datas_wrapper = datas_wrapper
    def __len__(self):
        if isinstance(self._datas,list):
            return len(self._datas)
        elif isinstance(self._datas,dict):
            return get_dict_list_length(self._datas)
    def _slice_data(self,s:slice):
        if isinstance(self._datas,list):
            return self._datas[s]
        elif isinstance(self._datas,dict):
            return {k:v[s] for k,v in self._datas.items()}
    def __iter__(self):
        self._latest_pipeline = self._datas_wrapper(self._slice_data(slice(0,len(self),1)))
        return iter(self._latest_pipeline)
    def __next__(self):
        raise NotImplementedError
    def cardinality(self):
        return len(self)


class SynchronizedDataIter():
    """
    Help to get a shuffled data slice accord to outer_counters `step` and `epoch`,
    where `step` and `epoch` means the passed steps and epochs.

    If X = outer_counters['epoch'], Y = outer_counters['step'],
    the data will be shuffled `X+1` times and 
        give out a slice in [Y%length::]

    So if from the scratch (X=0,Y=0), the procedure will be:
        - get original datas
        iter():
            - shuffle datas once, the 1st shuffle from origin
            - give out datas [0::]
        ...
    if from some recover point (X=x,Y=y), the procedure will be:
        - get original datas
        iter():
            - shuffle datas x+1 times, the (x+1)th shuffle from origin
            - give out datas [y%length::]
        ...
    Any time, with a certain X,Y, iter(this_class) will give out a copy slice of a certain shuffled data.
    The output iterator is only influenced by outer_counters

    """
    def __init__(self,datas:list|dict[str,list],counters:dict[Literal["step","epoch"],tf.Variable],seed:int|None=None,datas_wrapper:Callable=lambda x:x) -> None:
        self._inner_counters = {'step':0,'epoch':0} # have passed step epoch
        self._outer_counters = counters # real have passed step epoch
        self._datas = copy.deepcopy(datas) # do not influence original data, fixing the data
        self._random = get_random_from_seed(seed) 
        self._datas_wrapper = datas_wrapper
        self._check()
    def _check(self):
        assert (self._outer_counters['step']//len(self))==self._outer_counters['epoch']
        assert (self._inner_counters['step']-len(self))<=self._outer_counters['step']
        assert (self._inner_counters['epoch']-1)<=self._outer_counters['epoch']
    def __len__(self):
        if isinstance(self._datas,list):
            return len(self._datas)
        elif isinstance(self._datas,dict):
            return get_dict_list_length(self._datas)
    def _slice_data(self,s:slice):
        if isinstance(self._datas,list):
            return self._datas[s]
        elif isinstance(self._datas,dict):
            return {k:v[s] for k,v in self._datas.items()}
    
    def __iter__(self):
        self._check()
        for _ in range((self._outer_counters['epoch']-self._inner_counters['epoch']).numpy()+1): # random from scratch
            self._datas = random_datas(self._datas,self._random)
            self._inner_counters['epoch'] += 1
            self._inner_counters['step'] += len(self)
        self._latest_pipeline = self._datas_wrapper(self._slice_data(slice(self._outer_counters['step'].numpy()%len(self),len(self),1)))
    
        return iter(self._latest_pipeline)
    def __next__(self):
        raise NotImplementedError
    def cardinality(self):
        return len(self)
    


