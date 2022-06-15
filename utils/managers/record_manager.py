import os
import operator
from typing import Callable,Sequence

import numpy as np

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
    def __init__(self,path:str="./",) -> None:
        self.path = os.path.normpath(path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        assert os.path.isdir(self.path)
    def _get_target_path(self,name:str):
        return os.path.normpath(f"{self.path}\\{name}.npz") 
    def load(self,name:str,keys:tuple[str,...],gen_func:Callable[[],Sequence[np.ndarray]])->Sequence[np.ndarray]:
        files_getter = operator.itemgetter(*keys)
        try:
            saved_file = np.load(self._get_target_path(name))
            output = files_getter(saved_file)
        except KeyError:
            saved_file = dict(**saved_file).update(zip(keys,gen_func()))
            np.savez_compressed(self._get_target_path(name),**saved_file)
            output = files_getter(saved_file)
        except FileNotFoundError:
            saved_file = dict(zip(keys,gen_func()))
            np.savez_compressed(self._get_target_path(name),**saved_file)
            output = files_getter(saved_file)
        return output