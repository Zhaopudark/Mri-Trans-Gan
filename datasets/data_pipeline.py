"""
Towards Specific Experiments
"""
import os 
import sys
from numpy import iterable
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
            """
                In BraTS dataset, if we directly 
                read a ".nii" file and transform it to numpy array,
                we will get a [Z,Y,X] tensor in shape [240,240,155],
                    `Z` dimension: coronal axis, index min to max == R(Right) to L(Left)
                    `Y` dimension: sagittal axis, index min to max == A(Anterior) to P(Posterior)
                    `X` dimension: vertical axis, index min to max == I(Inferior) to S(Superior)
                This can be recorded as `R:L,A:P,I:S` format.

                    [Z,:,:] == a slice of sagittal plane
                    [:,Y,:] == a slice of coronal plane
                    [:,:,X] == a slice of transverse plane or horizontal plane

                Generally, in image-to-image translations, we concern more about transverse plane slices,
                treating it as a 2-D image, with sagittal and coronal axes considered as H(Height) and W(Width) dimensions.
                What's more, we usually use [D(Depth),H(Height),W(Width)] shape to represent a 3D image. 
                So, if D(Depth), H(Height) and W(Width) dimensions are vertical, sagittal and  coronal axes respectively,
                it will be helpful for next procedures.

                So, the [Z,Y,X] tensor from BraTS should be transpose to [X,Y,Z]
                If directly tranpose, we will get `I:S,A:P,R:L` format.
                For more conveniently drawing, we transpose [Z,Y,X] tensor to [-X,Y,Z],
                then, we get `S:I,A:P,R:L` format. Regard  [-X,Y,Z] dimension as [D1,D2,D3]
                    Then [D1,:,:] is  transverse plane or horizontal plane in `A:P,R:L` format
                    Then [:,D2,:] is  coronal plane in `S:I,R:L` format
                    Then [:,:,D3] is  sagittal plane in `S:I,A:P` format
            """
            path_collection = brats.BraTSDataPathCollector("D:\\Datasets\\BraTS\\BraTS2021_new")
            datas = path_collection.get_individual_datas(tags=(None,None,('t1','t2','t1ce','flair','shared'),(args['norm'],'mask')))
            self._mapping = brats.BraTSMapping(
                axes_format=("vertical","sagittal","coronal"),
            axes_direction_format=("S:I","A:P","R:L"),
            record_path="D:\\Datasets\\BraTS\\BraTS2021_new\\records2",)
            data_pipeline = brats.BraTSBasePipeLine(datas)
            data_pipeline = brats.BraTSDividingWrapper(data_pipeline,dividing_rates=tuple(args['data_dividing_rates']),dividing_seed=args['data_dividing_seed'])
            data_pipeline = brats.BraTSPatchesWrapper(data_pipeline,cut_ranges=args['cut_ranges'],patch_sizes=args['patch_sizes'],patch_nums=args['patch_nums'])
            datasets = list(data_pipeline())
            self.data_pipeline =  [SynchronizedDataIter(datasets[0],counters,args['data_random_seed'],self._pipeline_wrapper)]+\
                                   list(map(DataIter,datasets[1::],[self._pipeline_wrapper]*len(datasets[1::])))
                             
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
    def _map_func(self,inputs:dict[str:tf.Tensor]):
        return {key:value[...,tf.newaxis] for key,value in inputs.items()}  
    # @staticmethod
    # def add_channel(inputs:dict[str:tf.Tensor]):
    #     return {key:value[...,tf.newaxis] for key,value in inputs.items()}  
    def _pipeline_wrapper(self,datas:dict[str,list[str]]): # tf.data.Dataset.from_generator 传递的一定是tensor
        return tf.data.Dataset.from_tensor_slices(datas)\
            .map(self._mapping.mapping_patches,num_parallel_calls=4,deterministic=True)\
            .batch(self.batch_size,num_parallel_calls=4,deterministic=True)\
            .prefetch(tf.data.AUTOTUNE)
    def patch_combine_generator(self,*args,**kwargs):
        yield from brats.BraTSMapping.stack_patches(*args,**kwargs)
    def __call__(self)->list[SynchronizedDataIter,DataIter]:
        # return map(self.pipeline_wrapper,self.data_pipeline())
        return self.data_pipeline
        # return self.pipeline_wrapper(self.data_pipeline()),None,None

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    import tempfile
    import ast
    args = {}
    args['dataset']= 'braTS'
    args['norm']='individual_min_max_norm'
    args['cut_ranges']=((155//2-8,155//2+7),(0,239),(0,239))
    args['data_dividing_rates'] = (0.7,0.2,0.1)
    args['data_dividing_seed']= 0
    args['patch_sizes']=(16,128,128)
    args['patch_nums']=(1,3,3)
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
    train_set_2,test_set_2,validation_set_2 = pipe_line_2()
    # print(train_set.cardinality())
    # print(len(train_set))
    # train_set = tf.data.Dataset.range(10).map(lambda x:x).batch(2)
    # print(type(train_set))
    # print(iterable(train_set))
    # print(type(iter(train_set)))
    # print(train_set.cardinality())
    # print(train_set.cardinality())
    # print(train_set.cardinality())
    

    
    
    # def tested(train_set):
        
    #     print(train_set.cardinality())
    #     print(train_set.cardinality())
    #     d1 = iter(train_set)
    #     # # train_set = iter(train_set)
    #     # # train_set = iter(train_set)
    #     d2 = iter(iter(train_set))
        
    #     return d1,d2
    #     # return dataset
    # import timeit
    # print(timeit.timeit('tested(train_set)',number=1,globals=globals()))
    # print(timeit.timeit('tested(train_set)',number=1,globals=globals()))
    # for x1,x2 in zip(*tested(train_set)):
    #     keys1 = tuple(x1.keys())
    #     keys2 = tuple(x1.keys())
    #     assert keys1==keys2
    #     for key in keys1:
    #         tf.debugging.assert_equal(x1[key],x2[key])
    # import progressbar
    # from alive_progress import alive_bar
    # # train_bar = alive_bar(train_set.cardinality(), dual_line=True, title='Alphabet')
    # with alive_bar(train_set.cardinality(), dual_line=True, title='Alphabet') as bar:
    for item in test_set:
        # decoded = ast.literal_eval(str(item["name"].numpy(),encoding='utf-8'))
        print(len(item))
        print(item["t1"].shape,item["t1"].dtype)
        # print(item["t1_ranges"].numpy())
        print(item["t2"].shape,item["t2"].dtype)
        # print(item["t2_ranges"].numpy())
        print(item["t1ce"].shape,item["t1ce"].dtype)
        # print(item["t1ce_ranges"].numpy())
        print(item["flair"].shape,item["flair"].dtype)
        # print(item["flair_ranges"].numpy())
        print(item["mask"].shape,item["mask"].dtype)
        print(item["mask_ranges"].shape,item["mask_ranges"].dtype)
        print(item["mask_ranges"].numpy())
        print(item.keys())
       
        break

    for i,item in enumerate(test_set):
        # decoded = ast.literal_eval(str(item["name"].numpy(),encoding='utf-8'))
        print(len(item))
        print(item["t1"].shape,item["t1"].dtype)
        # print(item["t1_ranges"].numpy())
        print(item["t2"].shape,item["t2"].dtype)
        # print(item["t2_ranges"].numpy())
        print(item["t1ce"].shape,item["t1ce"].dtype)
        # print(item["t1ce_ranges"].numpy())
        print(item["flair"].shape,item["flair"].dtype)
        # print(item["flair_ranges"].numpy())
        print(item["mask"].shape,item["mask"].dtype)
        print(item["mask_ranges"].numpy())
        print(item.keys())
        if i>=2:
            break
    print(len(test_set))
        # print(item["mask"].numpy().min(),item["t1"].numpy().max())
    
    # with tempfile.TemporaryDirectory() as dir_name:
    #     step = counters1['step']
    #     epoch = counters1['epoch']
        
    #     checkpoint = tf.train.Checkpoint(counters=counters1,d=train_set)
    #     ckpt_manager = tf.train.CheckpointManager(checkpoint=checkpoint,directory=dir_name,max_to_keep=3,step_counter=step,checkpoint_interval=10)
    #     buf1 = []
    #     for s,datas in zip(range(step.numpy()+1,25+1),train_set):
    #         step.assign(s)
    #         buf1.append((s,tf.reduce_mean(datas['t1']).numpy()))
    #         ckpt_manager.save(check_interval=True,checkpoint_number=s)
    #         if s>=13:
    #             break
    #     ckpt_manager.restore_or_initialize()
    #     for s,datas in zip(range(step.numpy()+1,25+1),train_set):
    #         step.assign(s)
    #         buf1.append((s,tf.reduce_mean(datas['t1']).numpy()))
    #         ckpt_manager.save(check_interval=True,checkpoint_number=s)
    # with tempfile.TemporaryDirectory() as dir_name:
    #     step = counters2['step']
    #     epoch = counters2['epoch']
    #     checkpoint = tf.train.Checkpoint(counters=counters2)
    #     ckpt_manager = tf.train.CheckpointManager(checkpoint=checkpoint,directory=dir_name,max_to_keep=3,step_counter=step,checkpoint_interval=10)
    #     buf2 = []
    #     for s,datas in zip(range(step.numpy()+1,25+1),train_set_2):
    #         step.assign(s)
    #         buf2.append((s,tf.reduce_mean(datas['t1']).numpy()))
    #         ckpt_manager.save(check_interval=True,checkpoint_number=s)
    #         if s>=13:
    #             break
    #     ckpt_manager.restore_or_initialize()
    #     for s,datas in zip(range(step.numpy()+1,25+1),train_set_2):
    #         step.assign(s)
    #         buf2.append((s,tf.reduce_mean(datas['t1']).numpy()))
    #         ckpt_manager.save(check_interval=True,checkpoint_number=s)


    # for item1,item2 in zip(buf1,buf2):
    #     print(item1,item2)



    
       
