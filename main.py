import tensorflow as tf
from implementations.experiments2 import experiment_runer
if __name__=="__main__":

    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    experiment_runer(arg_path='implementations\\args.txt',logging_config_path='logging_config.yaml')