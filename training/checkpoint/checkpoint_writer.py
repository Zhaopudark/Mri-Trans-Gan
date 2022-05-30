
import time
import logging

import tensorflow as tf 
__all__ = [
    'Checkpoint',
]
class Checkpoint():
    def __init__(self,counters_dict,path,max_to_keep,checkpoint_interval,**kwargs):
        self.__counters_dict = counters_dict
        self.path = path
        self.__max_to_keep = max_to_keep
        self.__checkpoint_interval = checkpoint_interval
        self.__epoch = self.__counters_dict['epoch']
        self.__step = self.__counters_dict['step']
        self.__ckpt = tf.train.Checkpoint(**self.__counters_dict,**kwargs)
        self.__directory = self.path+'/tf_ckpts'
        self.__ckpt_manager = tf.train.CheckpointManager(checkpoint=self.__ckpt,
                                                         directory=self.__directory,
                                                         max_to_keep=self.__max_to_keep,
                                                         step_counter=self.__step,
                                                         checkpoint_interval=self.__checkpoint_interval)
        self.__start = time.perf_counter() 
        logging.getLogger(__name__).info(f"counters_dict have:{self.__counters_dict.keys()}")
    def save(self):
        save_path = self.__ckpt_manager.save(check_interval=True)
        if save_path is not None:
            logging.getLogger(__name__).debug(f"Ckpt saved when step/epoch:{self.__step.numpy()}/{(self.__epoch).numpy()} completed!")
            logging.getLogger(__name__).debug(f"{self.__checkpoint_interval} steps take {time.perf_counter()-self.__start} sec.")
            logging.getLogger(__name__).debug(f"Path:{save_path}")
            self.__start = time.perf_counter() 
        return save_path
    def restore_or_initialize(self):
        return self.__ckpt_manager.restore_or_initialize()
    @property
    def checkpoints(self):
        return  self.__ckpt_manager.checkpoints
    def restore(self,kept_point):
        self.__ckpt.restore(kept_point)
        logging.getLogger(__name__).info(f"current_checkpoint:{kept_point}")
 
if __name__=='__main__':
    step = tf.Variable(0,dtype=tf.int64,trainable=False)
    epoch = tf.Variable(0,dtype=tf.int64,trainable=False)
    counters_dict={'step':step,'epoch':epoch}

    ckpt = Checkpoint(counters_dict=counters_dict,path=".",max_to_keep=3,checkpoint_interval=10)
    ckpt.restore_or_initialize()
    for _ in range(100):
        step.assign_add(1)
        ckpt.save()
    