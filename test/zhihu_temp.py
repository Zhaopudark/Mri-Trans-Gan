import tensorflow as tf 
from typing import Literal
import tempfile
import copy
import random 
import logging
from typeguard import typechecked
import itertools
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_random_from_seed(seed:int|None=None):
    return random.Random(seed) if seed is not None else None
def random_datas(datas:list,random:random.Random|None=None):
    if random is not None:
        random.shuffle(datas) # since datas has been shuffled, the next shuffle will not be the same
        logging.info(f"Random complete!,the first data is {datas[0]}")
    return datas
class DataIter():
    @typechecked
    def __init__(self,datas:list,counters:dict[Literal['step','epoch'],tf.Variable],seed:int|None=None) -> None:
        # self.count = 0
        self._epoch = counters['epoch'] # have passed epoch
        self._step = counters['step'] #  have passed step
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
        return self._epoch.numpy()
    @property
    def step(self):
        return self._step.numpy()
    def __iter__(self):
        datas = copy.deepcopy(self.datas) # do not influence the fixed data
        random = get_random_from_seed(self.seed)
        for _ in range(self.epoch+1): # self.epoch+1 means current epoch
            datas = random_datas(datas,random)
        # self.step%self.length do not need 'minus 1', because it is the start index exactly
        return itertools.islice(datas,self.step%self.length,self.length) 
    def __repr__(self) -> str:
        return f"Static indices is epoch:{self.epoch} step:{self.step}."
    
datas = [{'test':str(item)} for item in  range(100)]
def mapfunc(x):
    k = list(x.keys())
    v = list(x.values())
    y = list(map(lambda inp:int(inp),v))
    return dict(zip(k,y))
with tempfile.TemporaryDirectory() as dir_name:
    
    step = tf.Variable(0)
    epoch = tf.Variable(0)
    checkpoint = tf.train.Checkpoint(step=step,epoch=epoch)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=checkpoint,directory=dir_name,max_to_keep=3,step_counter=step,checkpoint_interval=10)
    counters = {'step':step,'epoch':epoch}
    di  = DataIter(datas,counters=counters,seed=0)
    def generator(): # can be overwrited 
        for item in di:
            yield mapfunc(item)
    dataset = tf.data.Dataset.from_generator(generator,output_signature=({'test':tf.TensorSpec(shape=[])}))

    buf1 = []
    for e in range(epoch.numpy()+1,5+1):
        
        stop_flag = True
        for s,item  in zip(range(step.numpy()+1,25+1),dataset):
            # train step
            step.assign(s)
            buf1.append((step.numpy(),tf.reduce_mean(item['test']).numpy()))
            ckpt_manager.save(check_interval=True,checkpoint_number=step)
            if step.numpy()>=13:
                break
        else:
            stop_flag = False
        if stop_flag:
            break
        epoch.assign(e)

    print(step,epoch)
    ckpt_manager.restore_or_initialize()
    print(step,epoch)
    for e in range(epoch.numpy()+1,5+1):
        for s,item  in zip(range(step.numpy()+1,25+1),dataset):
            step.assign(s)
            buf1.append((step.numpy(),tf.reduce_mean(item['test']).numpy()))
            ckpt_manager.save(check_interval=True,checkpoint_number=step)
        epoch.assign(e)
    print(buf1)
b = list(range(100))
r = random.Random(0)
a = b[:]
r.shuffle(a)
print(a[:25])


