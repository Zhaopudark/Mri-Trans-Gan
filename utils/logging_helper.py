import os 
import re 
import yaml
import ast
import copy
import logging
import logging.config
from typeguard import typechecked


class MyLogging(logging.getLoggerClass()):
    ...
    

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
