import copy
import itertools
from typing import Literal

import tensorflow as tf 

from ..operations.random_op import get_random_from_seed,random_datas

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