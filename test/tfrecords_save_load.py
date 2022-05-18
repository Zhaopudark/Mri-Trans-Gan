import os 
import sys 
from typing import Iterable,Callable,Generator,Literal,Any
from typeguard import typechecked
import random
import math
import itertools
import functools
import numpy as np
import tensorflow as tf 
import logging
import copy
from utils.dataset_helper import read_nii_file,data_dividing,get_random_from_seed,path_norm
from utils.image.patch_process import index_cal
from datasets.brats.brats_data import BraTSDataPathCollection
from datasets.brats.brats_data import BraTSBase
import tempfile
with tempfile.NamedTemporaryFile(suffix=".tfrecords") as example_path:
    np.random.seed(0)
    # Write the records to a file.
    with tf.io.TFRecordWriter(example_path) as file_writer:
        for _ in range(4): #for _ in range(4000):
            x, y = tf.random.normal(shape=[155,240,240,3]), tf.random.normal(shape=[155,240,240,3])
            x = tf.io.serialize_tensor(x)
            y = tf.io.serialize_tensor(y)
            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.numpy()])),
                'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y.numpy()])),
            })).SerializeToString()

            file_writer.write(record_bytes)
    #Read the data back out.
    def decode_fn(record_bytes):
        return tf.io.parse_single_example(
            record_bytes, 
            {'x': tf.io.FixedLenFeature([], dtype=tf.string),
            'y': tf.io.FixedLenFeature([], dtype=tf.string)})

    for batch in tf.data.TFRecordDataset([example_path,]).map(decode_fn).shuffle(4):
        print(tf.io.parse_tensor(batch['x'],out_type=tf.float32).shape)
        print(tf.io.parse_tensor(batch['y'],out_type=tf.float32).shape)


class TFRecordManager():
    """
    """
    @typechecked
    def __init__(self,path:str="./",) -> None:
        self.path = path_norm(path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        assert os.path.isdir(self.path)
    def _get_target_path(self,name:str):
        # since typecheck may slow down, use the __wrapped__ one
        return path_norm(f"{self.path}\\{name}.tfrecords") 
    def _bytes_feature(self,value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _save(self,name:str,datas:dict[str:tf.Tensor])->None:
        with tf.io.TFRecordWriter(self._get_target_path(name)) as file_writer:
            feature_buf = {key:self._bytes_feature(tf.io.serialize_tensor(value)) for key,value in datas.items()}
            record_bytes = tf.train.Example(features=tf.train.Features(feature=feature_buf)).SerializeToString()
            file_writer.write(record_bytes)
    def load(self,name:str,data_types:dict[str,tf.DType],gen_func:Callable)->tuple[np.ndarray,...]:
        try:
            buf = {}
            for record_bytes in tf.data.TFRecordDataset(self._get_target_path(name)):
                feature_buf = {key:tf.io.FixedLenFeature([],dtype=tf.string) for key in data_types}
                batch = tf.io.parse_single_example(record_bytes,feature_buf)
                for key,dtype in data_types.items():
                    buf[key] = tf.io.parse_tensor(batch[key],out_type=dtype)
            return buf
        except Exception:
            arrays = gen_func()
            self._save(name,arrays)
            return arrays
tf_record_manager= TFRecordManager()
def gen():
    keys = ('img','mask','padding_vector')
    arrays = (tf.random.normal(shape=[16,128,128]),
              tf.random.normal(shape=[16,240,240]),
              tf.constant([[0,0],[0,1],[3,12]]),
                )
    return dict(zip(keys,arrays))
for _ in range(100):
    keys = ('img','mask','padding_vector')
    dtypes = (tf.float32,tf.float32,tf.int32)
    name = 'test_record'
    out = tf_record_manager.load(name,dict(zip(keys,dtypes)),gen_func=gen)
    print(out)




