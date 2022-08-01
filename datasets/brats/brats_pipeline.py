"""
Towards Specific Experiments
"""
# from ixi.ixi_pipeline import DataPipeLine as IXIDataPipeLine
import functools
import itertools
import json
import operator
from typing import Callable, Generator,Iterable,Any
from dataclasses import dataclass,field

import numpy as np
import tensorflow as tf

from datasets.brats import bratsbase
from utils.operations import datas_dividing,combine2patches,extend_to


@dataclass(slots=True)
class BraTsSingleData():
    t1:dict[str,str]
    t2:dict[str,str]
    t1ce:dict[str,str]
    flair:dict[str,str]
    mask:dict[str,str]
    meta_data:dict[str,str]
    def get_value(self,x:dict[str,str],key:str):
        try:
            return  x[key]
        except KeyError:
            return  x[f"patch_{key}"]
    def __post_init__(self):
        self.t1 = self.get_value(self.t1,'path')
        self.t2 = self.get_value(self.t2,'path')
        self.t1ce = self.get_value(self.t1ce,'path')
        self.flair = self.get_value(self.flair,'path')
        self.mask = self.get_value(self.mask,'path')
        self.meta_data = self.get_value(self.meta_data,'meta_data')


@dataclass(slots=True)
class BraTsGroupData():
    t1:list[dict[str,str]]
    t2:list[dict[str,str]]
    t1ce:list[dict[str,str]]
    flair:list[dict[str,str]]
    mask:list[dict[str,str]]
    meta_data:list[dict[str,str]]
    def get_single_data(self):
        for item in zip(self.t1,
            self.t2,self.t1ce,self.flair,
            self.mask,self.meta_data):
            yield  BraTsSingleData(*item)
          

class BraTSPipeline():
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

        the data architecture is 
        dict={
            't1':[t1_path_1,t1_path_2,...],
            't2':[t2_path_1,t2_path_2,...],
            't1ce':[t1ce_path_1,t1ce_path_2,...],
            'flair':[flair_path_1,flair_path_2,...],
            'mask':[mask_path_1,mask_path_2,...],
            'meta_data':[meta_data_1,meta_data_1,...]
    }
    """
    def __init__(self,args:dict) -> None:
        """
        arg: hyperparameters 
        """
        self._axes_format = args['axes_format']
        self._axes_direction_format = args['axes_direction_format']
        if args["use_patch"]:
            self.datas = self.gen_patch_datas(remark=args['norm'])
        else:
            self.datas = self.gen_datas(remark=args['norm'])
        self.datas = datas_dividing(self.datas,rates=args['data_dividing_rates'],seed=args['data_dividing_seed'])
        self.datas = list(map(self.data_de_grpup,self.datas))
        self.datas = list(map(self.data_aggregation,self.datas))
    def data_de_grpup(self,datas:list[BraTsSingleData|BraTsGroupData]):
        buf = []
        for item in datas:
            if isinstance(item,BraTsSingleData):
                buf.append(item)
            elif isinstance(item,BraTsGroupData):
                buf.extend(item.get_single_data())
            else:
                raise ValueError(" ") #TODO
        return buf
    def data_aggregation(self,datas:list[BraTsSingleData]):
        buf = []
        for item in datas:
            buf.append([item.t1,item.t2,item.t1ce,item.flair,item.mask,item.meta_data])
        outdict =  dict(zip(('t1','t2','t1ce','flair','mask','meta_data'),zip(*buf)))
        for key in outdict:
            outdict[key] = list(outdict[key])
        return outdict

    def __call__(self)->tuple[dict[str,list[str]]]:
        return self.datas

    def get_path(self,target)->Generator[dict[str,Any],None,None]:
        yield from bratsbase.tb_select_gen('SELECT tb1.patient_id,`path` FROM tb_modalities tb1 JOIN tb_patients '
            ' tb2 ON tb1.patient_id=tb2.patient_id where `modality`=%(modality)s and `remark`=%(remark)s ORDER BY tb1.patient_id',target)
    def get_meta(self,target)->Generator[dict[str,Any],None,None]:
        yield from bratsbase.tb_select_gen('SELECT tb1.patient_id,`meta_data` FROM tb_modalities tb1 JOIN tb_patients '
            ' tb2 ON tb1.patient_id=tb2.patient_id where `modality`=%(modality)s and `remark`=%(remark)s ORDER BY tb1.patient_id',target)
    def get_patch_path(self,target)->Generator[dict[str,Any],None,None]:
        targets = bratsbase.tb_select_gen('SELECT tb1.patient_id,`patch_path` FROM tb_patches tb1 JOIN tb_patients '
            ' tb2 ON tb1.patient_id=tb2.patient_id where `modality`=%(modality)s and `remark`=%(remark)s ORDER BY tb1.patient_id, `patch_index`',target)
        for _,item in itertools.groupby(targets,key=lambda x:x['patient_id']):
            yield list(item)
    def get_patch_meta(self,target)->Generator[dict[str,Any],None,None]:
        targets = bratsbase.tb_select_gen('SELECT tb1.patient_id,`patch_meta_data` FROM tb_patches tb1 JOIN tb_patients '
            ' tb2 ON tb1.patient_id=tb2.patient_id where `modality`=%(modality)s and `remark`=%(remark)s ORDER BY tb1.patient_id, `patch_index`',target)
        for _,item in itertools.groupby(targets,key=lambda x:x['patient_id']):
            yield list(item)
    def gen_datas(self,remark:str):
       return [ BraTsSingleData(*item) 
            for item in 
                zip(
                    self.get_path({'modality':'t1','remark':remark}),
                    self.get_path({'modality':'t2','remark':remark}),
                    self.get_path({'modality':'t1ce','remark':remark}),
                    self.get_path({'modality':'flair','remark':remark}),
                    self.get_path({'modality':'mask','remark':'main'}),
                    self.get_meta({'modality':'mask','remark':'main'})
                )
        ]
    def gen_patch_datas(self,remark:str):
        return [ BraTsGroupData(*item) 
            for item in 
                zip(
                    self.get_patch_path({'modality':'t1','remark':remark}),
                    self.get_patch_path({'modality':'t2','remark':remark}),
                    self.get_patch_path({'modality':'t1ce','remark':remark}),
                    self.get_patch_path({'modality':'flair','remark':remark}),
                    self.get_patch_path({'modality':'mask','remark':'main'}),
                    self.get_patch_meta({'modality':'mask','remark':'main'})
                )
        ]
    def decode(self,tensor_like_str:tf.Tensor):
        return str(tensor_like_str.numpy(),encoding='UTF-8')
    def tensors_map(self,path:str):
        return self._pre_process(self.decode(path))[...,tf.newaxis]
    def ranges_map(self,ranges):
        return tf.convert_to_tensor([ranges[index] for index in self.transpose_permutation]+[[0,0]])
    def sizes_map(self,sizes):
        return tf.convert_to_tensor([sizes[index] for index in self.transpose_permutation]+[1])
    def index_map(self,index):
        return tf.convert_to_tensor(index)
    def patch_map_func(self,data:dict[str,tf.Tensor]):
        input_getter = operator.itemgetter('t1','t2','t1ce','flair','mask','meta_data')
        meta_data_getter = operator.itemgetter('TOTAL_RANGES','VALID_RANGES','PATCH_SIZES','PATCH_RANGES','PATCH_INDEX','MAX_INDEX')
        output_keys = ('t1','t2','t1ce','flair','mask',
        'total_ranges',
        'valid_ranges',
        'patch_sizes',
        'patch_ranges',
        'patch_index',
        'max_index')
        # return {'123123123':self._pre_process(self.decode(data['t1']))}
        
        Tout = [tf.float32,]*5+[tf.int32,]*6
        Fmap = [self.tensors_map,]*5+[
            self.ranges_map,
            self.ranges_map,
            self.sizes_map,
            self.ranges_map,
            self.index_map,
            self.index_map,
            ]
        def py_function(*inputs):
            return [f(x) for f,x in zip(Fmap,itertools.chain(inputs[:-1],meta_data_getter(json.loads(self.decode(inputs[-1])))))]
        return dict(zip(output_keys,tf.py_function(py_function,inp=input_getter(data),Tout=Tout)))
    def _pre_process(self,path:str)->np.ndarray:# transpose [Z,Y,X] tensor to [-X,Y,Z]
        x:np.ndarray = np.load(path)
        x = x.transpose(self.transpose_permutation)
        return np.flip(x,axis=self.flip_axes)  
    @classmethod
    def zip_data(cls,data:dict[str,tf.Tensor]):
        """
        data = {
            ...:...,
            ...:...,
            'total_ranges':...,
            'patch_ranges':...,
        }
        """
        total_ranges = data.pop('total_ranges')
        patch_ranges = data.pop('patch_ranges')
        mask = tf.ones_like(data['mask'],dtype=tf.int32)
        yield from zip( data.keys(),
                        data.values(),
                        [patch_ranges,]*len(data.keys()),
                        [mask,]*len(data.keys())) 
    @classmethod
    def unzip_data(cls,data:Iterable[Iterable[Any]],total_ranges):
        buf = {}
        for (kx,x,xr,xm) in data:
            _xm = tf.cast(tf.maximum(xm,1),x.dtype)
            x = tf.divide(x,_xm)
            tf.debugging.assert_all_finite(x,message="assert_all_finite",name=None)
            tf.debugging.assert_type(x,tf.float32,message=None,name=None)
            tf.debugging.assert_type(xr,tf.int32,message=None,name=None)
            buf[kx] = extend_to(x,xr,total_ranges)
        buf |= {'patch_ranges':total_ranges}
        return buf
    @classmethod
    def stack_patches(cls,datas:Iterable[dict[str,tf.Tensor]])-> dict[str,tf.Tensor]:
        """
        data in datas is 
            data = {
                ...:...,
                'total_ranges':...,
                'valid_ranges':...,
                'patch_sizes':...,
                'patch_ranges':...,
                'patch_index':...,
                'max_index':...,
            }
        """
        base = None
        for data in datas:
            data.pop('patch_sizes')
            dropped_info = {
                    'valid_ranges':data.pop('valid_ranges'),
                    'patch_sizes':tf.convert_to_tensor([max(item)-min(item)+1 for item in data['total_ranges']]),
            }
            total_ranges = data['total_ranges']
            patch_ranges = data['patch_ranges']
            patch_index = data.pop('patch_index')
            max_index = data.pop('max_index')
            # tf.print(patch_index,max_index)
            if base is None:
                base = cls.zip_data(data)
            else: #combine2patches
                current = cls.zip_data(data)
                buf = [combine2patches(ka,a,ar,am,kb,b,br,bm) for (ka,a,ar,am),(kb,b,br,bm) in zip(base,current)]
                base = buf
            datas
            if patch_index == max_index:
                yield cls.unzip_data(base,total_ranges)|dropped_info|{
                    'total_ranges':total_ranges,
                    'patch_index':tf.zeros_like(patch_index,dtype=tf.int32),
                    'max_index':tf.zeros_like(max_index,dtype=tf.int32)}
                base = None
        if base is not None:
            raise ValueError(" ") # TODO
         
    @property
    def transpose_permutation(self):
        # get transpose_permutation by compare _axes_format with AXES_FORMAT
        if not hasattr(self,"_transpose_permutation"):
            self._transpose_permutation = tuple(bratsbase.AXES_FORMAT.index(item) for item in self._axes_format)
        return self._transpose_permutation
    @property
    def flip_axes(self):
        # get flip_axes by compare _axes_direction_format with transposed AXES_DIRECTION_FORMAT
        if not hasattr(self,"_flip_axes"):
            transposed_direction_format = tuple(bratsbase.AXES_DIRECTION_FORMAT[index] for index in self.transpose_permutation)
            def filter_func(index):
                normal_direction = transposed_direction_format[index]
                fliped_direction = transposed_direction_format[index][::-1]
                if self._axes_direction_format[index] == normal_direction:
                    return False # need not flip
                elif self._axes_direction_format[index] == fliped_direction:
                    return True # need flip to match _axes_direction_format
                else:
                    ValueError(f"{self._axes_direction_format[index]} should in `{normal_direction} or {fliped_direction}`")
            self._flip_axes = tuple(filter(filter_func,range(len(self._axes_direction_format))))
        return self._flip_axes

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    args = {}
    args['dataset']= 'braTS'
    args['norm']='individual_min_max_norm'
    args['cut_ranges']=((155//2-8,155//2+7),(0,239),(0,239))
    args['data_dividing_rates'] = (0.7,0.2,0.1)
    args['data_dividing_seed']= 0
    args['use_patch'] = True
    args['patch_sizes']=(16,128,128)
    args['axes_format']=("vertical","sagittal","coronal")
    args['axes_direction_format']=("S:I","A:P","R:L")
    args['overlap_tolerances']=((0.3,0.5),(0.3,0.5),(0.3,0.5))
    args['batch_size'] =1
    args['data_random_seed'] = 1200
    args['global_random_seed'] = 1200
    args['domain'] = (0.0,255.0)
    pipe_line = BraTSPipeline(args)
    train_set,test_set,validation_set = pipe_line()
    print(len(train_set['t1']),len(test_set['t1']),len(validation_set['t1']))
    from matplotlib import pyplot as plt
    # dataset = tf.data.Dataset.from_tensor_slices(test_set).map(pipe_line.patch_map_func)
    # for i,item in enumerate(dataset):
    #     print(i,"********************************")
    #     # plt.imshow(item["t1"][item["t1"].shape[0]//2,...,0])
    #     # plt.show()
    #     tf.debugging.assert_none_equal(tf.reduce_mean(item["t1"]),0.0)
    #     print(item["t1"].shape,item["t1"].dtype,)
    #     print(item["t2"].shape,item["t2"].dtype)
    #     print(item["t1ce"].shape,item["t1ce"].dtype)
    #     print(item["flair"].shape,item["flair"].dtype)
    #     print(item["mask"].shape,item["mask"].dtype)
    #     print(item["total_ranges"].shape,item["total_ranges"].dtype,item["total_ranges"])
    #     print(item["valid_ranges"].shape,item["valid_ranges"].dtype,item["valid_ranges"])
    #     print(item["patch_sizes"].shape,item["patch_sizes"].dtype,item["patch_sizes"])
    #     print(item["patch_ranges"].shape,item["patch_ranges"].dtype,item["patch_ranges"])
    #     print(item["patch_index"].shape,item["patch_index"].dtype,item["patch_index"])
    #     print(item["max_index"].shape,item["max_index"].dtype,item["max_index"])

    

    dataset = tf.data.Dataset.from_tensor_slices(test_set).map(pipe_line.patch_map_func)
    for i,item in enumerate(BraTSPipeline.stack_patches(dataset)):
        print(i,"********************************")
        # plt.imshow(item["t1"][77,...,0])
        # plt.show()
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
    dataset = tf.data.Dataset.from_tensor_slices(validation_set).map(pipe_line.patch_map_func)
    for i,item in enumerate(BraTSPipeline.stack_patches(dataset)):
        print(i,"********************************")
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