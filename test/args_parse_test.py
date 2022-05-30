import argparse
from collections import namedtuple
import imp
import re
from tkinter import N
from turtle import stamp

from numpy import iterable, var
import numpy
parser = argparse.ArgumentParser(prog='MRI_Trans_GAN',allow_abbrev=False,fromfile_prefix_chars='@')
args = parser.add_argument('--action',choices=['initial','train','test','debug',"train-and-test"],type=str.lower)
print(parser)
args = parser.add_argument("--workspace",type=str),
args = parser.add_argument("--init",type=str.lower)
args = parser.add_argument("--indicator",type=str.lower)
args = parser.add_argument("--global_random_seed",type=int,help="全局随机种子 其余部分",default=0)
args = parser.parse_args()
all_defaults = {}
for key in vars(args):
    all_defaults[key] = parser.get_default(key)
print(all_defaults)
args = vars(args)
print(args)
defaults = vars(parser.parse_args([]))
print(defaults)

def get_args_stamp(vars:dict,default_vars:dict,passby_keys:list[str]=None):
    if passby_keys is None:
        passby_keys = []
    stamp = ''
    for (k1,v1),(k2,v2) in zip(vars.items(),default_vars.items()):
        assert k1==k2
        if (k1 not in passby_keys) and (v1 != v2):
            stamp += f"_{str(v1)}"
    return stamp.strip("_")
        
print(get_args_stamp(args,defaults))


from matplotlib.cbook import flatten
from tensorflow import nest
from iteration_utilities import deepflatten
a = [1,[2,3],[["4",[5,6]],[7]],{"8","9"},{"10":11},{12:"13"},{"14":15},{16,17},True,{False:None}]
print(list(flatten(a)))
print(nest.flatten(a))
print(list(deepflatten(a)))

from typing import Iterable,NamedTuple
def _flatten(items:Iterable):
    for item in items:
        match item:
            case str(item)|bytes(item)|int(item)|float(item)|complex(item):
                yield item
            case item if isinstance(item,Iterable):
                yield from _flatten(item)
            case item:
                raise ValueError(f"{item} in unexpected type:{type(item)}.")
            
