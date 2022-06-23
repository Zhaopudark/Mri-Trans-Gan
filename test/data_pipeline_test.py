# 想个办法为 tf.data.Dataset.from_generator 传递用于指示数据长度的参数 不限于 __len__

import ast
import logging
import random
import tempfile
import collections
from dataclasses import dataclass,field
import itertools
import functools
from numpy import iterable, outer
import tensorflow as tf
from typing import Any, Callable,Iterable,Mapping,Sequence,Iterator,overload
import copy

from typeguard import typechecked
from zmq import has
from datasets import brats
from datasets.brats.brats_data import BraTSData

# from datasets.brats.brats_pipeline_v2 import BraTSBasePipeLine
# from utils.dtype_helper import nested_dict_key_sort,gen_key_value_from_nested_dict,check_nested_dict
from utils.operations import random_datas

physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os  
   
    
    

if __name__ == "__main__":
    def test():
        # d = BraTSBasePipeLine(
        #             path="D:\\Datasets\\BraTS\\BraTS2021_new",
        #             record_path="D:\\Datasets\\BraTS\\BraTS2021_new\\records",
        #             axes_format=('vertical','sagittal','coronal'),
        #             axes_direction_format=("S:I","A:P","R:L"),
        #             norm_method="individual_min_max",
        #             ) 
        # datas = d.datas
        # [(None,None,('t1','t2','t1ce','flair'),individual_identities)]+[(None,None,('shared',),shared_identity) for shared_identity in shared_identities]
        # for item in datas['t1']:
        #     print(item)

        bdpc = brats.BraTSDataPathCollector("D:\\Datasets\\BraTS\\BraTS2021_new")
        # print(bdpc.data[("Training","BraTS2021_00000",'t1')])
        # print(type(bdpc.data))


        datas = bdpc.get_individual_datas(tags=(None,None,('t1','t2','t1ce','flair','shared'),('main','mask??????????')),should_existed=False)
        # print(type(datas))
        # for item in datas:
        #     print(item,"***********")
        added = [data+(1,2,3) for data in datas]

        serialized = BraTSData.reduce(added).serialize()
        print(type(serialized))
        return serialized

     
    
        # for k,v in serialized.items():
        #     print(k)
        #     print(type(v),len(v))
        #     for i,(item) in enumerate(v):
        #         print(item)
        #         print(type(item))
        #         if i>=3:
        #             break
        # map(lambda x:x,datas)
        # reduced = BraTSData.reduce_datas(map(lambda x:x,datas))
       

        # for k,v in reduced.items():
        #     print(k)
        #     print(type(v),len(v))
        #     for i,(item) in enumerate(v):
        #         print(item)
        #         print(type(item))
        #         if i>=3:
        #             break
        # print(type(reduced))
        # data = tf.data.Dataset.from_tensor_slices(serialized)
        # print(len(Data))

            # print(k,v)
        # print("??????????????????????????????")
        # bdpc.inner_data.sort((('Training','Validation'),None,('t1','flair','t1ce','t2',"share"),None))
        # for item in bdpc.data.get_items(("Training","BraTS2021_00000",None,None)):
        #     print(type(item))
        #     # print(len(item))
        #     print(item[0])
        #     print(type(item[0]))
        # group = itertools.groupby(bdpc[(None,None,('shared',),"mask")],key=lambda kv:bratsbase.get_groupby_stamp(kv[0]))
        # datas = BraTSDatas()
        # for item in group:
        #     # print(item)
        #     datas.update({item[0]:{item[0]:item[1]}})
        # datas.update({item[0]:{item[0]:'item[1]'}})
        # for k,v in datas.items():
        #     print(k)
        #     print(v)


        # for item in datas_list:

        #     print(item.get_extened_datas("jajajaaja"+str(random.randint(0,10))))
        #     pass
    import timeit
    import random
    # print(timeit.timeit(test, number=10))
    serialized = test()
    print(type(serialized))
    def test2():
        random_datas(serialized,random=random.Random(0))
    print(timeit.timeit(test2, number=100))

    a  = {'1':[1,2,3,4,5,6,7]}
    a = [1,2,3,4,5,6,7]
    rd = random.Random(0)

    random_datas(a,random=rd)
    print(a)
    random_datas(a,random=rd)
    print(a)
    random_datas(a,random=rd)
    print(a)
    random_datas(a,random=rd)
    print(a)
    random_datas(a,random=rd)
    print(a)




