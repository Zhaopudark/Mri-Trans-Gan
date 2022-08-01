from collections import UserDict
import os 
import re
from typing import Any 
import yaml
import ast
import copy
import logging
import logging.handlers
import logging.config
from typeguard import typechecked
import pathlib
import datetime
from PIL import Image
import pandas as pd
import tensorflow as tf 

class SummaryDataCollections(UserDict):
    """
    self.data = {
        sample_name1:sample_value,
        sample_name2:...,
        ...,
    }
    """
    _supported_typs = {'image','scalar','text'}
    def __init__(self,summary_type:str,name:str):
        assert summary_type in self._supported_typs
        self.summary_type = summary_type
        self.name = name
        super().__init__()

class _TensorboardLogMaker():
    def __init__(self,path:pathlib.Path) -> None:
        self.path = path/'tensorboard'
    def __call__(self,inputs:list[SummaryDataCollections],step:tf.Variable)->None:
        for summary_data_collections in inputs:
            summary_type = summary_data_collections.summary_type
            name = summary_data_collections.name
            file_writer = tf.summary.create_file_writer(str(self.path/f"{name}")) 
            with file_writer.as_default(step=step):
                for sample_name,sample_value in summary_data_collections.items():
                    match summary_type:
                        case 'image':
                            tf.summary.image(name=sample_name,data=sample_value)
                        case 'text':
                            tf.summary.text(name=sample_name,data=sample_value)
                        case 'scalar':
                            tf.summary.scalar(name=sample_name,data=sample_value)
                        case _:
                            raise ValueError(f"Unsupported summary type: `{summary_type}`")
                
class _ImageLogMaker():
    def __init__(self,path:pathlib.Path) -> None:
        self.path = path/'images'
    def __call__(self,inputs:list[SummaryDataCollections],step:tf.Variable)->None:
        for images in inputs:
            current_path = self.path/f"step_{int(step)}"/f"{images.name}"
            current_path.mkdir(parents=True,exist_ok=True)
            for sample_name,sample_value in images.items():
                tf.debugging.assert_shapes([(sample_value,(1,'H','W','C'))])
                img = Image.fromarray(sample_value[0,...].numpy())
                img.save(current_path/f"{sample_name}.png")

class _ScalarLogMaker():
    """
    将list[SummaryDataCollections] 转置为行列表
    增加'sample_names'一列
    默认list[SummaryDataCollections]中各个SummaryDataCollections的`sample_names`完全一致
    """
    def __init__(self,path:pathlib.Path) -> None:
        self.path = path/'scalars.xlsx'
    def __call__(self,inputs:list[SummaryDataCollections],step:tf.Variable)->None:
        sheet_name = f"step_{int(step)}"
        names = ['sample_names']+[item.name for item in  inputs]
        assert all(item.summary_type=='scalar' for item in  inputs)
        datas = [list(inputs[0].keys())]+[list(item.values()) for item in  inputs]
        df = pd.DataFrame(dict(zip(names,datas)))
        try:
            with pd.ExcelWriter(self.path,mode='a',engine="openpyxl") as  writer:
                df.to_excel(writer, sheet_name=sheet_name,index=False)
        except FileNotFoundError:#(FileNotFoundError,ValueError)
            with pd.ExcelWriter(self.path,mode='w',engine="openpyxl") as  writer:
                df.to_excel(writer, sheet_name=sheet_name,index=False)   

class _SummaryManager():
    """
    Record high-level logs information,
    such as tables, figures, tensorboard-logs ...
    summary/
        train/
            tensorboard/
            images/...
            'scalars.xlsx'
        test/
            datatime/
                tensorboard/
                images/...
                'scalars.xlsx'
    """    
    def __init__(self,path:pathlib.Path) -> None:
        self.path = path
        self.tensorboard_log_maker =  _TensorboardLogMaker(path)
        self.image_log_maker =  _ImageLogMaker(path)
        self.scalar_log_maker =  _ScalarLogMaker(path)
    def __call__(self,inputs:list[SummaryDataCollections],step:tf.Variable) -> Any:
        self.tensorboard_log_maker(inputs,step)
        self.image_log_maker([item for item in inputs if item.summary_type=='image'],step)
        self.scalar_log_maker([item for item in inputs if item.summary_type=='scalar'],step)

class TestSummaryMaker(_SummaryManager):
    def __init__(self,path) -> None:
        super().__init__(pathlib.Path(path)/'summary'/'test'/datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
class TrainSummaryMaker(_SummaryManager):
    def __init__(self,path) -> None:
        super().__init__(pathlib.Path(path)/'summary'/'train')


@typechecked
def set_global_loggers(config_path:str='logging_config.yaml',target_prefix:str='.'):
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r', encoding="utf-8")as f:
        logging_yaml = yaml.load(stream=f, Loader=yaml.CLoader)
        copyed_yaml = copy.deepcopy(logging_yaml)
        copyed_yaml.pop('helper')
        for handler_name,handler_keys in logging_yaml['handlers'].items():
            if 'filename' in handler_keys:
                compiled_pattern = re.compile(ast.literal_eval(logging_yaml['helper']['filePattern']))
                repl = ast.literal_eval(logging_yaml['helper']['fileReplPattern'])
                file_path = compiled_pattern.sub(repl,handler_keys['filename']).format(target_prefix) # add prefix | locate the log's file output dir
                file_path = os.path.normpath(file_path)
                dir_path = os.path.dirname(os.path.abspath(file_path))
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                copyed_yaml['handlers'][handler_name]['filename'] = file_path
    logging.config.dictConfig(config=copyed_yaml)

@typechecked
def get_simple_logger(logger_name:str,console_level=logging.WARNING,file_level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.handlers.RotatingFileHandler('logs/preprocess.log',maxBytes=2*1024*1024, backupCount=3)
    file_handler.setLevel(file_level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt="%Y/%m/%d %I:%M:%S")
    file_handler.setFormatter(formatter)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt="%Y/%m/%d %I:%M:%S")
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

if __name__ == "__main__":
    import datetime
    print(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
