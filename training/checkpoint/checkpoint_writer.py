import tensorflow as tf 
import time
__all__ = [
    "Checkpoint",
]
class Checkpoint():
    def __init__(self,counters_dict,path,max_to_keep,checkpoint_interval,**kwargs):
        self.__counters_dict = counters_dict
        self.path = path
        self.__max_to_keep = max_to_keep
        self.__checkpoint_interval = checkpoint_interval
        self.__epoch = self.__counters_dict["epoch"]
        self.__step = self.__counters_dict["step"]
        self.__ckpt = tf.train.Checkpoint(**self.__counters_dict,**kwargs)
        self.__directory = self.path+'/tf_ckpts'
        self.__ckpt_manager = tf.train.CheckpointManager(checkpoint=self.__ckpt,
                                                         directory=self.__directory,
                                                         max_to_keep=self.__max_to_keep,
                                                         step_counter=self.__step,
                                                         checkpoint_interval=self.__checkpoint_interval)
        self.__start = time.time()
        print("counters_dict have:",self.__counters_dict.keys())
    def save(self):
        save_path = self.__ckpt_manager.save(check_interval=True)
        if save_path is None:
            pass 
        else:
            print("Ckpt saved when step/epoch:{}/{} completed!".format(self.__step.numpy(),(self.__epoch+1).numpy()))
            print("{} steps take {} sec.".format(self.__checkpoint_interval,time.time()-self.__start))
            print("Path:{}".format(save_path))
            self.__start = time.time()
        info = save_path
        return info
    def restore_or_initialize(self):
        info = self.__ckpt_manager.restore_or_initialize()
        return info
    @property
    def checkpoints(self):
        return  self.__ckpt_manager.checkpoints
    def restore(self,kept_point):
        self.__ckpt.restore(kept_point)
        print("current_checkpoint:{}".format(kept_point))
 
if __name__=="__main__":
    step = tf.Variable(0,dtype=tf.int64,trainable=False)
    epoch = tf.Variable(0,dtype=tf.int64,trainable=False)
    counters_dict={"step":step,"epoch":epoch}

    ckpt = Checkpoint(counters_dict=counters_dict,path=".",max_to_keep=3,checkpoint_interval=10)
    ckpt.restore_or_initialize()
    for _ in range(100):
        step.assign_add(1)
        ckpt.save()
    