import os 
from typing import Iterable,Callable,Generator,Literal,Any,Sequence
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
from datasets.brats.brats_data import BraTSBase
from datasets.brats.brats_data import BraTSData
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
    @typechecked
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

class RecordManager():
    """
    This class manager the record files. We can get our expected datas from specific files by `load()`.

    Managing datasets by saving and loading tf-record files has been deprecated for the following reasons:
        1. tf-record files is hard to read and save:
            Need use `tf.train.Feature`, `tf.train.Features`, `tf.io.TFRecordWriter` for loading
                and use `tf.data.TFRecordDataset`,`tf.io.FixedLenFeature` and `tf.io.parse_single_example` for reading.
            In additional, if save and load non-scalar tensors, should use `tf.io.serialize_tensor` to transfer it to bytes and `tf.io.parse_tensor` to get original tensor
        2. tf-record's saving behavior only works in Eager Mode, i.e., can not used in tf.data.Dataset.map() unless wrappered by tf.py_function 
        3. `tf.io.TFRecordDataset` will load all files in memory when shuffle() is used, i.e.,  the mapping of inner segments within `tf.io.TFRecordDataset` is not implemented.
    So, here, we build a new RecordManager with the following keys:
        1. Make a mapping from specific files to file stamps(names)
        2. In disk or remote device, each specific file can be saved separately or in groups, depending on the pressure and flux requirements of the data pipeline.
        3. When the corresponding specific file of a file stamp does not exist, generate and save the file before return.
        4. Use memory and storage consumption as less as possible.
    `numpy`'s NPZ file can provide us with solutions for the above keys.

    Args:
        path: the record file's saved and loaded path
    Methods:
        load:
            load(name:str,structure:dict[str,Any],gen_func:Callable[[],dict[str,np.ndarray]])
            this method works on a group of specific files(ndarrays), that saved in a NPZ file (determined by `name`)
            `structure`'s keys() determines what files(ndarrays) in the saved NPZ file will return.
            For greater compatibility, i.g., it is more convenient to process sequence in posterior procedures, here we just return sequence of ndarray instead of a dict in the same sturcture with `structure`.

    """
    @typechecked
    def __init__(self,path:str="./",) -> None:
        self.path = os.path.normpath(path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        assert os.path.isdir(self.path)
    def _get_target_path(self,name:str):
        return os.path.normpath(f"{self.path}\\{name}.npz") 
    def load(self,name:str,structure:dict[str,Any],gen_func:Callable[[],dict[str,np.ndarray]])->Sequence[np.ndarray]:
        keys = structure.keys()
        files_getter = operator.itemgetter(*keys)
        try:
            saved_file = np.load(self._get_target_path(name))
            output = files_getter(saved_file)
        except KeyError:
            saved_file = dict(**saved_file) | gen_func()
            np.savez_compressed(self._get_target_path(name),**saved_file)
            output = files_getter(saved_file)
        except FileNotFoundError:
            saved_file = gen_func()
            np.savez_compressed(self._get_target_path(name),**saved_file)
            output = files_getter(saved_file)
        return output
class BraTSBasePipeLine():
    """
    Here, we define an abstract data.
        An abstract data is a piece of information.
        This piece of information a is combination of raw data info and descriptor parameters
            raw data info: 
                the original data info,
                may be directly loaded from single data file or a group of data files. The file can be `nii` file, numpy save files(npy npz) or other files.
                or may be a single path or group of paths that lead to specific file or files store in local host or remote host
            descriptor parameters:
                the parameters for descriptor,
                tells the posterior descriptor how to deal with the raw data info
    Then, we define a descriptor (descriptor class). see https://docs.python.org/zh-cn/3/howto/descriptor.html
        This descriptor class just realize `__get__` method, mapping abstract data to our expected data by a mapping function and return the expected one.
        The mapping function takes both abstract data's raw data info and descriptor parameters as inputs.
        The raw data info is the mapping function's operation object, the descriptor parameters is the operation's required parameters. 
        Our descriptor only realizes equivalent `__get__` method by mapping function.
            We do not need to realize  `__set__` and `__delete__`methods, so the descriptor is `Non-data descriptor` 
            Non-data descriptor: Instance's attribut has the priority over the descriptor
    For simplicity, we use @property 
        we define `datas` (decorated by property) as the abstract datas, contains all abstract data's `raw data info` and `descriptor parameters` in a dict structure.
        we define map_function as the `mapping function` of descriptor
        we do not realize the descriptor in this class, beacuse the map_function require huge memory consumption. Posterior procudure will 
        apply the map_function on `datas` to get it's desired datas.

    
    This class read all files in BraTS dataset and categorize them by patient files's name.       
    
    Consider a procedure form data_raw --> data_needed
        basic procedure is 
            read data_raw (maybe from path, maybe raw_data, maybe npz(not npy))
            give out a dict of numpy ndarray (support npz)
        post_read   
    """
    @typechecked
    def __init__(self,
                 path:str,
                 record_path:str,
                 norm_method:str="individual_min_max_norm",
                 axes_format:tuple[Literal["vertical","sagittal","coronal"],...]=("vertical","sagittal","coronal"),
                 axes_direction_format:tuple[Literal["S:I","A:P","R:L"],...]=("S:I","A:P","R:L"),
                 counters:dict[Literal["step","epoch"],tf.Variable]=None,
                 seed:int|None=None
                 ) -> None:
        self._data_path_collection = BraTSDataPathCollection(path)
        self._record_manager = RecordManager(record_path)
        self._norm_method = norm_method
        self._axes_format = axes_format
        self._axes_direction_format  = axes_direction_format
        self._counters = counters
        self._seed = seed
    @property
    def datas(self)->list[BraTSData]:
        """
        When dealt with by `tf.data.Dataset.from_tensor_slice()`, we will get each item(slice) in type `dict[str,tf.Tensor]`.
        This will be the input of `map_function`
        """
        if not hasattr(self,"_datas"):
            self._datas = self._data_path_collection.get_individual_datas(self._norm_method,["mask",])
            # self._datas = [{"name":item.name}|item.datas for item in datas]
        return self._datas
    @property
    def data_iter(self):
        if not hasattr(self,"_data_iter"):
            self._data_iter = DataIter(self.datas,counters=self._counters,seed=self._seed)
        return self._data_iter
    @property
    def transpose_permutation(self):
        if not hasattr(self,"_transpose_permutation"):
            self._transpose_permutation = tuple(BraTSBase.AXES_FORMAT.value.index(item) for item in self._axes_format)
        return self._transpose_permutation
    @property
    def size(self): # the size after permutation
        if not hasattr(self,"_size"):
            self._size = tuple(BraTSBase.SIZE.value[index] for index in self.transpose_permutation)
        return self._size
    @property
    def flip_axes(self):
        if not hasattr(self,"_flip_axes"):
            transposed_direction_format = tuple(BraTSBase.AXES_DIRECTION_FORMAT.value[index] for index in self.transpose_permutation)
            def filter_func(index):
                assert self._axes_direction_format[index] in [transposed_direction_format[index], transposed_direction_format[index][::-1]]
                return self._axes_direction_format[index] == transposed_direction_format[index][::-1]
            self._flip_axes = tuple(filter(filter_func,range(len(self._axes_direction_format))))
        return self._flip_axes
    def _pre_process(self,img:np.ndarray)->np.ndarray:# transpose [Z,Y,X] tensor to [-X,Y,Z]
        img = img.transpose(self.transpose_permutation)
        return np.flip(img,axis=self.flip_axes)  
    def read_from_path(self,path:str)->np.ndarray:
        img,_,_ = read_nii_file(path,dtype=np.float32)
        return self._pre_process(img)
    def _gen_func(self,datas:dict[str,Any]):
        return {key:self.read_from_path(value) for key,value in datas.items()}
    def _get_storage_structure(self,datas)->dict[str,tf.TensorSpec]: # corresponding with _gen_func
        return {key:tf.TensorSpec(shape=None,dtype=tf.float32) for key in datas}
    @property
    def output_structure(self)->dict[str,tf.TensorSpec]: # corresponding with _gen_func, may be the same storage_structure 
        if not hasattr(self,"_output_structure"):
            datas = self.datas[0].datas
            self._output_structure = {key:tf.TensorSpec(shape=None,dtype=tf.float32) for key in datas}
        return self._output_structure
    def map_function(self,data:BraTSData):
        name = data.name
        datas = data.datas
        storage_structure = self._get_storage_structure(datas)
        output_sequence = self._record_manager.load(name=name,structure=storage_structure,gen_func=functools.partial(self._gen_func,datas=datas))
        return dict(zip(self.output_structure.keys(),output_sequence))
    def generator(self,data_iter)->Generator: # a non-parameter generator is usefull for next procedures(such as tf.data.Dataset.from_generator)
        for data in data_iter:
            yield self.map_function(data)
    @typechecked
    def __call__(self)->Callable[[],Generator]: 
        return functools.partial(self.generator,data_iter=self.data_iter)
    
class BraTSDividingWrapper():
    """
    This class wrapped a basicpipeline, 
    dividing datas into several pipelines by `dividing_rates` that given.

    `dividing_rates` indicates the number of parts and proportion of each part in the total datas.

    if `dividing_rates` is a tuple, (a1,a2,...,an)
        there should be a1+a2+...+an<=1.0, a1,a2,...,an>=0

    we will get n piplines (pipeline_1,pipeline_2,...,pipeline_n) 
    consider there are N elements in total datas
        pipeline_1 will give out a1*N elements
        pipeline_2 will give out a2*N elements
        ...
        pipeline_n will give out an*N elements
    The elements between these pipelines are mutually exclusive. 

    Only the first pipline support random behavior (Usually, it's the train set pipeline)
    """
    @typechecked  
    def __init__(self,pipeline:BraTSBasePipeLine,
                    dividing_rates:tuple[float,...]=None,
                    dividing_seed:None|int=None):
        self.pipeline = pipeline
        self._dividing_rates = dividing_rates
        self._dividing_seed = dividing_seed
        self._record_manager = pipeline._record_manager
        self._counters = self.pipeline._counters
        self._seed = self.pipeline._seed
    #------------------------------maybe changeable when wrapper----------------------------------#
    @property
    def datas(self)->tuple[list[BraTSData]]:
        if not hasattr(self,"_datas"):
            self._datas = tuple(data_dividing(self.pipeline.datas,dividing_rates=self._dividing_rates,random=get_random_from_seed(self._seed),selected_all=True))
        return self._datas
    @property
    def output_structure(self)->dict[str,tf.TensorSpec]:
        return self.pipeline.output_structure
    @property
    def data_iters(self):
        if not hasattr(self,"_data_iters"):
            seeds = tuple([self._seed]+[None,]*(len(self.datas)-1))
            counters = tuple([self._counters]+[None]*(len(self.datas)-1))
            self._data_iters = list(map(DataIter,self.datas,counters,seeds))
        return self._data_iters
    def read_from_path(self,path:str)->np.ndarray:
        return self.pipeline.read_from_path(path)
    def generator(self,data_iter:DataIter)->Generator: # a non-parameter generator is usefull for next procedures(such as tf.data.Dataset.from_generator)
        yield from self.pipeline.generator(data_iter)
    @typechecked
    def __call__(self)->list[Callable[[],Generator]]: # return callable funcs, which will return generators when call
        def gen(data_iter):
            return functools.partial(self.generator,data_iter=data_iter)
        return list(map(gen,self.data_iters))

class BraTSPatchesWrapper():
    """
    This class wrapped a pipeline, 
    dividing datas into several patches.
    """
    @typechecked  
    def __init__(self,pipeline:BraTSBasePipeLine|BraTSDividingWrapper,
                cut_ranges:tuple[tuple[int,int],...],
                patch_sizes:tuple[int,...],
                patch_nums:tuple[int,...]):
        """
        datas:接受 _Data()的实例列表 data in datas中 data的target_path_dict 存放我们希望的数据路径(一般为真正处理好的数据)
        """
        self.pipeline = pipeline
        self._cut_ranges = cut_ranges
        self._patch_sizes = patch_sizes
        self._patch_nums = patch_nums
        assert len(cut_ranges)==len(patch_sizes)==len(patch_nums)
        self._record_manager = pipeline._record_manager
        self._seed = pipeline._seed
        self._counters = pipeline._counters
    @property
    def patch_ranges(self)->Iterable[str]:
        if not hasattr(self,"_patch_ranges"):
            patch_ranges_info = list(map(index_cal,self._cut_ranges,self._patch_sizes,self._patch_nums))
            int_results = list(itertools.product(*patch_ranges_info)) # list() is uesd to save the results
            str_results = [' '.join([str(item) for item in tf.nest.flatten(int_result)]) for int_result in int_results]
            # int_results2 = [tf.nest.pack_sequence_as(self._cut_ranges,[int(item) for item in str_result.split(' ')]) for str_result in str_results]
            # assert int_results == int_results2
            self._patch_ranges = str_results
        return self._patch_ranges
    @property
    def datas(self)->list[BraTSData]|tuple[list[BraTSData]]:
        """
        When dealt with by`tf.data.Dataset.from_tensor_slice()`, we will get each item(slice) in type 
        `dict[str,tf.Tensor]`.
        This will be the input of `map_function`
        """
        if not hasattr(self,"_datas"):
            if isinstance(self.pipeline,BraTSBasePipeLine):
                self._datas = []
                for data,patch_ranges in itertools.product(self.pipeline.datas,self.patch_ranges):
                    data_attrs = data.get_attrs()
                    data_attrs["datas"] = {key:f"{{{value}}}{{{patch_ranges}}}" for key,value in data_attrs["datas"].items()}
                    self._datas.append(BraTSData(**data_attrs))
            elif isinstance(self.pipeline,BraTSDividingWrapper):
                out_buf = []
                for datas in self.pipeline.datas:
                    buf = []
                    for data,patch_ranges in itertools.product(datas,self.patch_ranges):
                        data_attrs = data.get_attrs()
                        data_attrs["datas"] = {key:f"{{{value}}}{{{patch_ranges}}}" for key,value in data_attrs["datas"].items()}
                        buf.append(BraTSData(**data_attrs))
                    out_buf.append(buf)
                self._datas = tuple(out_buf)
        return self._datas
    @property
    def data_iter(self):
        assert isinstance(self.pipeline,BraTSBasePipeLine)
        if not hasattr(self,"_data_iter"):
            self._data_iter = DataIter(self.datas,self._counters,self._seed)
        return self._data_iter
    @property
    def data_iters(self):
        assert isinstance(self.pipeline,BraTSDividingWrapper)
        if not hasattr(self,"_data_iters"):
            seeds = tuple([self._seed]+[None,]*(len(self.datas)-1))
            counters = tuple([self._counters]+[None,]*(len(self.datas)-1))
            self._data_iters = list(map(DataIter,self.datas,counters,seeds))
        return self._data_iters
    def _get_path_and_ranges(self,path_and_ranges:str)->tuple[str,str]:
        matched = re.match(r'{(.*)}{(.*)}',path_and_ranges)
        path,str_ranges = matched[1],matched[2]
        return path,str_ranges
    def _read_from_path_and_ranges(self,path_and_ranges:str)->np.ndarray:
        path,str_ranges = self._get_path_and_ranges(path_and_ranges)
        img = self.pipeline.read_from_path(path)
        ranges = tf.nest.pack_sequence_as(self._cut_ranges,[int(item) for item in str_ranges.split(' ')])
        return img[tuple(map(lambda x:slice(x[0],x[1]+1),ranges))]
    def _get_str_ranges(self,datas:dict[str,str]):
        def get_ranges(path_and_ranges):
            return self._get_path_and_ranges(path_and_ranges)[-1]
        return reduce_same(datas.values(),map_func=get_ranges)
    def _gen_func(self,datas:dict[str,Any]):
        str_ranges = self._get_str_ranges(datas)
        outputs = {f"{{{str_ranges}}}_{key}":self._read_from_path_and_ranges(value) for key,value in datas.items()}
        # add mask and padding_vector
        ranges = tf.nest.pack_sequence_as(self._cut_ranges,[int(item) for item in str_ranges.split(' ')])
        padding_vector = tuple(map(lambda x,y:[x[0]-y[0],y[1]-x[1]],ranges,self._cut_ranges)) 
        mask = np.pad(np.ones(self._patch_sizes,dtype=np.float32),padding_vector,mode='constant',constant_values=0.)
        padding_vector = np.array(padding_vector,dtype=np.int32)
        outputs |= {f"{{{str_ranges}}}_patch_mask":mask,f"{{{str_ranges}}}_patch_padding_vector":padding_vector}
        return outputs
    def _get_storage_structure(self,datas:dict[str,str],)->dict[str,str]: # corresponding with _gen_func
        str_ranges = self._get_str_ranges(datas)
        return {f"{{{str_ranges}}}_{key}":tf.TensorSpec(shape=None,dtype=tf.float32) for key in datas}|\
               {f"{{{str_ranges}}}_patch_mask":tf.TensorSpec(shape=None,dtype=tf.float32),f"{{{str_ranges}}}_patch_padding_vector":tf.TensorSpec(shape=None,dtype=tf.int32)}
    @property
    def output_structure(self)->dict[str,tf.TensorSpec]:
        if not hasattr(self,"_output_structure"):
            if isinstance(self.pipeline,BraTSBasePipeLine):
                datas = self.datas[0].datas
            elif isinstance(self.pipeline,BraTSDividingWrapper):
                datas = self.datas[0][0].datas
            self._output_structure = {key:tf.TensorSpec(shape=None,dtype=tf.float32) for key in datas}|\
                                     {"patch_mask":tf.TensorSpec(shape=None,dtype=tf.float32),
                                      "patch_padding_vector":tf.TensorSpec(shape=None,dtype=tf.int32)}
        return self._output_structure
    def map_function(self,data:BraTSData):
        name = data.name
        datas = data.datas
        storage_structure = self._get_storage_structure(datas)
        output_sequence = self._record_manager.load(name=name,structure=storage_structure,gen_func=functools.partial(self._gen_func,datas=datas))
        return dict(zip(self.output_structure.keys(),output_sequence))
    def generator(self,data_iter)->Generator: # a non-parameter generator is usefull for next procedures(such as tf.data.Dataset.from_generator)
        for data in data_iter:
            yield self.map_function(data)
    def __call__(self)->Generator|list[Generator]:
        if isinstance(self.pipeline,BraTSBasePipeLine):
            return functools.partial(self.generator,data_iter=self.data_iter)
        elif isinstance(self.pipeline,BraTSDividingWrapper):
            def gen(data_iter):
                return functools.partial(self.generator,data_iter=data_iter)
            return list(map(gen,self.data_iters))
        else:
            raise ValueError(f"pipeline should be an instance of {BraTSBasePipeLine} or {BraTSDividingWrapper}")
    @property
    def patch_count(self):
        if not hasattr(self,"_patch_count"):
            self._patch_count = functools.reduce(lambda x1,x2:x1*x2,self._patch_nums,1)
        return self._patch_count
    @typechecked
    def patch_combine_generator(self,datas:Iterable[dict[str,dict[str,Any]]])->Generator[dict[str,dict[str,Any]],None,None]: 
        """
        since functools.reduce will consume the Iterable datas immediately
        here we can not use `reduce`
        iter-1  {"x":{"img":tensor_1
                       "mask":tensor_1
                       "padding_vector":tensor_1},
                 "y":{"img":tensor_1
                       "mask":tensor_1
                       "padding_vector":tensor_1},
                ...
                }
        iter-2  {"x":{"img":tensor_2
                       "mask":tensor_2
                       "padding_vector":tensor_2},
                 "y":{"img":tensor_2
                       "mask":tensor_2
                       "padding_vector":tensor_2},
                ...
                }
        """
        buf = {}
        for i,(data_dict) in enumerate(datas):
            for key,value in data_dict.items():
                if key not in buf:
                    img = np.pad(value["img"],value["padding_vector"],mode='constant',constant_values=(0,))
                    mask = value["mask"]
                    buf[key] = {"img":img,"mask":mask}
                else:
                    inverse_mask = np.where(value["mask"]>=0.5,0.0,1.0)
                    img = buf[key]["img"]*inverse_mask+np.pad(value["img"],value["padding_vector"],mode='constant',constant_values=(0,))*value["mask"]
                    mask = np.where((buf[key]["mask"]+value["mask"])>=0.5,1.0,0.0)
                    buf[key] |= {"img":img,"mask":mask} # only support python 3.10
            if (i%self.patch_count)==(self.patch_count-1):
                yield buf
                buf = {}