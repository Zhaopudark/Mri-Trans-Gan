"""
Towards Specific Experiments
"""
import os 
import sys
import tensorflow as tf
# from ixi.ixi_pipeline import DataPipeLine as IXIDataPipeLine
from typeguard import typechecked
from typing import Callable,Literal

import datasets.brats as brats
from utils.managers import DataIter,SynchronizedDataIter

class DataPipeline():
    @typechecked
    def __init__(self,args:dict,counters:dict[Literal["step","epoch"],tf.Variable]) -> None:
        """
        arg: hyperparameters 

        """
        if args['dataset'].lower() == 'brats':
            
            self.pipe_line = brats.BraTSPipeline(args)
            datas = self.pipe_line()
            self.data_pipeline =  [SynchronizedDataIter(datas[0],counters,args['data_random_seed'],self._pipeline_wrapper2)]+\
                                   list(map(DataIter,datas[1::],[self._pipeline_wrapper2]*len(datas[1::])))
            
                             
        elif args['dataset'].lower() == 'ixi':
            # DataPipeLine = IXIDataPipeLine
            # train_path = "G:\\Datasets\\IXI\\Registration_train"
            # test_path = "G:\\Datasets\\IXI\\Registration_test"
            # seed_path = "G:\\Datasets\\IXI\\Registration_seed"
            # validation_path = "G:\\Datasets\\IXI\\Registration_validate"
            pass
        else:
            raise ValueError(f"Unsupported dataset {args['dataset']}")
        self.batch_size = args['batch_size']
    def _pipeline_wrapper(self,datas:dict[str,list[str]]): # tf.data.Dataset.from_generator 传递的一定是tensor
        return tf.data.Dataset.from_tensor_slices(datas)\
            .map(self._mapping.mapping_patches,num_parallel_calls=4,deterministic=True)\
            .batch(self.batch_size,num_parallel_calls=4,deterministic=True)\
            .prefetch(tf.data.AUTOTUNE)
    def _pipeline_wrapper2(self,datas:dict[str,list[str]]): # tf.data.Dataset.from_generator 传递的一定是tensor
        return tf.data.Dataset.from_tensor_slices(datas)\
            .map(self.pipe_line.patch_map_func,num_parallel_calls=4,deterministic=True)\
            .batch(self.batch_size,num_parallel_calls=4,deterministic=True)\
            .prefetch(tf.data.AUTOTUNE)
    def patch_combine_generator(self,*args,**kwargs):
        yield from brats.BraTSPipeline.stack_patches(*args,**kwargs)
    def __call__(self)->list[SynchronizedDataIter,DataIter]:
        # return map(self.pipeline_wrapper,self.data_pipeline())
        return self.data_pipeline
        # return self.pipeline_wrapper(self.data_pipeline()),None,None

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    import tempfile
    import ast
    import itertools
    args = {}
    args['dataset']= 'braTS'
    args['norm']='min_max_on_z_score'
    args['cut_ranges']=((155//2-8,155//2+7),(0,239),(0,239))
    args['data_dividing_rates'] = (0.7,0.2,0.1)
    args['data_dividing_seed']= 0
    args['use_patch'] = True
    args['axes_format']=("vertical","sagittal","coronal")
    args['axes_direction_format']=("S:I","A:P","R:L")
    args['patch_sizes']=(16,128,128)
    args['overlap_tolerances']=((0.3,0.5),(0.3,0.5),(0.3,0.5))
    args['batch_size'] =1
    args['data_random_seed'] = 1200
    args['global_random_seed'] = 1200
    args['domain'] = (0.0,255.0)
    step1 = tf.Variable(0)
    epoch1 = tf.Variable(0)
    counters1 = {'step':step1,'epoch':epoch1}
    step2 = tf.Variable(0)
    epoch2 = tf.Variable(0)
    counters2 = {'step':step2,'epoch':epoch2}
    pipe_line = DataPipeline(args,counters=counters1)
    pipe_line_2 = DataPipeline(args,counters=counters2)
    import time
    start = time.perf_counter() 


    tf.keras.utils.set_random_seed(args['global_random_seed'])
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args['global_random_seed'])
    train_set,test_set,validation_set = pipe_line()
    print(len(train_set),len(test_set),len(validation_set))
    train_set_2,test_set_2,validation_set_2 = pipe_line_2()
    max_range=[0,0,0]
    for item in itertools.chain(train_set,test_set,validation_set):
 
        print(item["t1"].shape,item["t1"].dtype)
        print(item["t2"].shape,item["t2"].dtype)
        print(item["t1ce"].shape,item["t1ce"].dtype)
        print(item["flair"].shape,item["flair"].dtype)
        print(item["mask"].shape,item["mask"].dtype)
        print(item["total_ranges"].shape,item["total_ranges"].dtype,item["total_ranges"])
        print(item["valid_ranges"].shape,item["valid_ranges"].dtype,item["valid_ranges"])
        print(item["patch_sizes"].shape,item["patch_sizes"].dtype,item["patch_sizes"])
        print(item["patch_ranges"].shape,item["patch_ranges"].dtype,item["patch_ranges"])
        print(item["patch_index"].shape,item["patch_index"].dtype,item["patch_index"])
        print(item["max_index"].shape,item["max_index"].dtype,item["max_index"])
        # print(item["valid_ranges"].shape,item["valid_ranges"].dtype,item["valid_ranges"])
        # for i,_range in enumerate(item["valid_ranges"][0]):
        #     maxed=(_range[-1]-_range[0]).numpy()
        #     if maxed>max_range[i]:
        #         max_range[i]=maxed
        # print(max_range)
        

