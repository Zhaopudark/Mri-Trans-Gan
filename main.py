import tensorflow as tf
import platform
from implementations.experiments2 import experiment_runer
from implementations.experiments_linux import experiment_runer_linux
if __name__=="__main__":

    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if platform.system() == 'Linux':
        arg_path = './implementations/args_linux.txt'
        experiment_runer_linux(arg_path=arg_path,logging_config_path='logging_config.yaml')
    elif platform.system() == 'Windows':
        arg_path = 'implementations\\args.txt'
        experiment_runer(arg_path=arg_path,logging_config_path='logging_config.yaml')
    else:
        raise ValueError(f"Unsupported platform :{platform.system()}.")
