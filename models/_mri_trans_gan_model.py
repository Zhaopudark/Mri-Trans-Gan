import os
import re 
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import datetime
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base,'../'))
from models.networks.network_selector import NetworkSelector
from training.optimizers.optimizer import Optimizer
from training.losses.gan_losses import GanLoss
from training.losses.image_losses import DualGanReconstructionLoss
from training.losses.image_losses import CycleConsistencyLoss
from training.process.train_process import TrainProcess 
from training.checkpoint.checkpoint_writer import Checkpoint
from training.logs.logs_maker import LogsMaker
from utils.image.image_process import Drawer
from training.metrics.metrics_conductor import MetricsConductor
from datasets.data_pipeline import DataPipeline
import itertools
from typeguard import typechecked
__all__ = [ 
    'MriTransGan',
]
class MriTransGan():
    def __init__(self,args):
        #-----------------------counters-dict---------------------#
        # 计数器 在模型中定义 同时给检查点(checkpoint) 日志 (logs_maker) 和模型网络(gan_process)
        self.steps = int(args.steps) # 目标训练步数
        self.epochs = int(args.epochs) # 目标训练周期
        self.step = tf.Variable(0,dtype=tf.int64,trainable=False)
        self.epoch = tf.Variable(0,dtype=tf.int64,trainable=False)
        self.counters_dict = {'step':self.step,'epoch':self.epoch}
        #------------------------data------------------------#
        self.dataset = DataPipeline(args,counters=self.counters_dict)
        self.train_set,self.test_set,self.validation_set = self.dataset()
        self.patch_combiner = self.dataset.data_pipeline.patch_combine_generator
        self.input_shape = [1,16,128,128,1]
        #------------------------model------------------------#
        _mixed_precision = bool(args.mixed_precision)
        if _mixed_precision:
            _policy = tf.keras.mixed_precision.Policy('mixed_float16')
        else:
            _policy = None
        #定义模型
        _GAN = NetworkSelector(args=args).architectures
        self.G0 = _GAN['generator'](args=args,name='G_0_x2y',dtype=_policy)
        self.G1 = _GAN['generator'](args=args,name='G_1_y2x',dtype=_policy)
        self.D0 = _GAN['discriminator'](args=args,name='D_0_y',dtype=_policy)
        self.D1 = _GAN['discriminator'](args=args,name='D_1_x',dtype=_policy)
        self.models_list = [self.D0,self.D1,self.G0,self.G1]
        #------------------------optimizer-training-process------------------------# 
        _G_learning_rate = float(args.G_learning_rate)
        _D_learning_rate = float(args.D_learning_rate)
        # _Do = Optimizer(_D_learning_rate,args=args,name='Do')
        # _Go = Optimizer(_G_learning_rate,args=args,name='Go')
        _Do0 = Optimizer(_D_learning_rate,args=args,name='Do0')
        _Do1 = Optimizer(_D_learning_rate,args=args,name='Do1')
        _Go0 = Optimizer(_G_learning_rate,args=args,name='Go0')
        _Go1 = Optimizer(_G_learning_rate,args=args,name='Go1')

        self.Do0 = _Do0.optimier
        self.Do1 = _Do1.optimier
        self.Go0 = _Go0.optimier
        self.Go1 = _Go1.optimier
        # self.Go_shadow = _Go_shadow.optimier
        self.optimizers_list = [self.Do0,self.Do1,self.Go0,self.Go1]
        self.train_process = TrainProcess(args=args)
        
        #-----------------------losses---------------------#
        _gan_loss_name = args.gan_loss_name
        self.gan_loss = GanLoss(_gan_loss_name,args=args,counters_dict=self.counters_dict) 
        self.cycle_loss = CycleConsistencyLoss(args=args)
        self.rec_loss = DualGanReconstructionLoss(args=args)
        #----------------logs-checkpoint-config--------------------#
        self.weight_path = args.weight_path # 初始化时加载
        self.logs_path = args.logs_path
        self.logs_interval = int(args.logs_interval)
        _checkpoint_interval = int(args.checkpoint_interval)
        _checkpoint_max_keep = int(args.checkpoint_max_keep)
        _metrics_list = args.metrics_list
        self.checkpoint = Checkpoint(counters_dict=self.counters_dict,
                                     path=self.logs_path,
                                     max_to_keep=_checkpoint_max_keep,
                                     checkpoint_interval=_checkpoint_interval,
                                     optimizers = self.optimizers_list,
                                     models = self.models_list)
        self.metrics = MetricsConductor(_metrics_list) # NOTE 独立于LogsMaker之外 迫使LogsMaker只负责记录而避免设计具体的计算或者绘图细节
        self.drawer = Drawer() # NOTE 独立于LogsMaker之外 迫使LogsMaker只负责记录而避免设计具体的计算或者绘图细节           
        self.logs_maker = LogsMaker(counters_dict=self.counters_dict,path=self.logs_path)
    #-----------------build-------------------------#
    def build(self): 
        """
        input_shape必须切片 因为在底层会被当做各层的输出shape而被改动
        """
        self.G0.build(input_shape=self.input_shape)#G0 x->y
        self.D0.build(input_shape=self.input_shape)#D0 y or != y
        self.G1.build(input_shape=self.input_shape)#G1 y->x
        self.D1.build(input_shape=self.input_shape)#D1 x or != x

        self.built = True
        self.G0.summary()
        self.D0.summary()
        self.G1.summary()
        self.D1.summary()
    def combination(self,dataset,predict_func): # NOTE 这部分内容很繁杂 但必须写在这里 这是模型面向具体任务的细化 写在别处将导致维护和调试的时间成本太高 风险太大
        def test_step_wrapper(dataset):
            for item in dataset:
                x = item['t1']
                y = item['t2']
                mask = item['mask']
                m = item['patch_mask']
                v = item['patch_padding_vector']
                y_,x_ = predict_func(x=x,y=y,mask=item['mask'],m=item['patch_mask'])
                yield {'x':{'img':x[0,...,0],
                            'mask':m[0,...,0],
                            'padding_vector':v[0,...,0]},
                        'y_':{'img':y_[0,...,0],
                            'mask':m[0,...,0],
                            'padding_vector':v[0,...,0]},
                        'x_':{'img':x_[0,...,0],
                            'mask':m[0,...,0],
                            'padding_vector':v[0,...,0]},
                        'y':{'img':y[0,...,0],
                            'mask':m[0,...,0],
                            'padding_vector':v[0,...,0]},
                        'mask':{'img':mask[0,...,0],
                            'mask':m[0,...,0],
                            'padding_vector':v[0,...,0]},
                       }
        def combiner_wrapper(inner_gen):
            def out_gen():
                yield from self.patch_combiner(inner_gen)
            def unit(x,m):
                x = x*m
                _min = x.numpy().min()
                _max = x.numpy().max()
                x = (x-_min)/(_max-_min)
                x = x*m
                _min = x.numpy().min()
                _max = x.numpy().max()
                return x
            for i,item in enumerate(out_gen()):
                x  = item['x']['img']
                y_ = item['y_']['img']
                x_ = item['x_']['img']
                y  = item['y']['img']
                m  = item['mask']['img']
                print('1x',x.shape,x.numpy().min(),x.numpy().max())
                print('1y_',y_.shape,y_.numpy().min(),y_.numpy().max())
                print('1x_',x_.shape,x_.numpy().min(),x_.numpy().max())
                print('1y',y.shape,y.numpy().min(),y.numpy().max())
                x = unit(x,m)
                y_ = unit(y_,m)
                x_ = unit(x_,m)
                y = unit(y,m)
                print('2x',x.shape,x.numpy().min(),x.numpy().max())
                print('2y_',y_.shape,y_.numpy().min(),y_.numpy().max())
                print('2x_',x_.shape,x_.numpy().min(),x_.numpy().max())
                print('2y',y.shape,y.numpy().min(),y.numpy().max())
                yield {'x':x,'y_':y_,'x_':x_,'y':y}
        def dict_wrapper(iterable):
            for item in iterable:
                yield {k: v[tf.newaxis,...,tf.newaxis] for k,v in item.items()}
        yield from dict_wrapper(combiner_wrapper(test_step_wrapper(dataset)))
    def _draw_pre_process(self,x):# 可扩展的 支持医学图像 RGB自然图像的
        if len(x.shape)==5:#'BDHWC':
            slice_len = x.shape[1]
            t= slice_len//2 # 以序列中部元素或者中部偏右元素为中心
            return x[0,t,:,:,0]
        elif len(x.shape)==4:
            return x[0,:,:,0]
        else:
            raise ValueError(f"Unexpected shape {x.shape}!")
    class _metric_apply():
        def __init__(self,indicate):
            self.name = indicate[0]+" and "+indicate[1]
            self.indicate = indicate
        def __call__(self,metric,images) :
            return metric(images[self.indicate[0]],images[self.indicate[1]]).numpy()
    def make_logs(self,dataset,predict_func,loss=None):
        # [loss metric image]
        log_types = []
        log_contents = []
        if loss is None:
            pass 
        else:
            log_types.append('loss')
            if isinstance(loss,dict):
                log_contents.append(loss)
            else:
                raise ValueError()
        log_types.append('image')
        image_dict = {}
        for i,out_put in enumerate(self.combination(dataset,predict_func)):
            image_dict[str(i)] = out_put
        log_contents.append(self.drawer.draw_from_dict(image_dict,process_func=self._draw_pre_process))
        log_types.append('metric')
        log_contents.append(self.metrics.calculate(dict_in_dict=image_dict,func_apply_list=[self._metric_apply(['x','x_']),self._metric_apply(['y','y_'])],mean=True))
        return log_types,log_contents
    #------------------predict-------------------------#
    def _predict_func(self,x,y,mask,m):
        y_ = self.G0(in_put=[x,m,mask],training=False,step=self.step,epoch=self.epoch)
        x_ = self.G1(in_put=[y,m,mask],training=False,step=self.step,epoch=self.epoch)
        return y_,x_
    #------------------train-------------------------#
    def _loss_func(self,x,y,mask,m):
        # tf.print(self.step)
        y_       = self.G0(in_put=[x,m,mask],training=True,step=self.step,epoch=self.epoch)
        # tf.print(tf.reduce_mean(y_),self.step)
        D_real_0,buf_real_0 = self.D0(in_put=[y,m,mask],buf_flag=True,training=True,step=self.step,epoch=self.epoch)
        D_fake_0,buf_fake_0 = self.D0(in_put=[y_,m,mask],buf_flag=True,training=True,step=self.step,epoch=self.epoch)
        # tf.print(tf.reduce_mean(D_real_0),self.step)
        # tf.print(tf.reduce_mean(D_fake_0),self.step)

        x_       = self.G1(in_put=[y,m,mask],training=True,step=self.step,epoch=self.epoch)
        D_real_1,buf_real_1 = self.D1(in_put=[x,m,mask],buf_flag=True,training=True,step=self.step,epoch=self.epoch)
        D_fake_1,buf_fake_1 = self.D1(in_put=[x_,m,mask],buf_flag=True,training=True,step=self.step,epoch=self.epoch)

        x__      = self.G1(in_put=[y_,m,mask],training=True,step=self.step,epoch=self.epoch)
        y__      = self.G0(in_put=[x_,m,mask],training=True,step=self.step,epoch=self.epoch)

        cycle_loss = self.cycle_loss.call(x=x,x__=x__,y=y,y__=y__)
        rec_loss = self.rec_loss.call(x=x,x_=x_,y=y,y_=y_,xd=buf_real_1,x_d=buf_fake_1,yd=buf_real_0,y_d=buf_fake_0)
        
        G_loss = self.gan_loss.generator_loss(D_real=D_real_0,D_fake=D_fake_0)+\
                    self.gan_loss.generator_loss(D_real=D_real_1,D_fake=D_fake_1)+\
                    cycle_loss+rec_loss
                    
        D_loss = self.gan_loss.discriminator_loss(D_real=D_real_0,D_fake=D_fake_0,real_samples=y,fake_samples=y_,D=self.D0,condition=m)+\
                    self.gan_loss.discriminator_loss(D_real=D_real_1,D_fake=D_fake_1,real_samples=x,fake_samples=x_,D=self.D1,condition=m)
        # tf.print(D_loss,G_loss)
        return [D_loss,G_loss] 
    def train_step(self):
        raise ValueError("train_step must be reload!!!")
    def test_step(self):
        raise ValueError("test_step must be reload!!!")
    def train(self):
        self._checkpoint_check()
        # tf.profiler.experimental.start(self.logs_path)
        _train_step = self.train_process.train_wrapper(
                          self._loss_func,
                          optimizer_list=[self.Do0,self.Go0],
                          variable_list=[self.D0.trainable_variables+self.D1.trainable_variables,self.G0.trainable_variables+self.G1.trainable_variables])
        _predict_step = self.train_process.predict_wrapper(self._predict_func)
        # self.step 代表已完成的step
        # self.epoch 代表已完成的epoch
        for epoch in range(self.epoch.numpy()+1,self.epochs+1):
            for step,item in zip(range(self.step.numpy()+1,self.steps+1),self.train_set):
                t1,t2,mask,m,v = item['t1'],item['t2'],item['mask'],item['patch_mask'],item['patch_padding_vector']
                x = t1
                y = t2
                mask = mask
                m_m = m
                # tf.print(tf.reduce_mean(x),tf.reduce_mean(y),tf.reduce_mean(mask),tf.reduce_mean(m_m))
                D_loss,G_loss = _train_step(x=x,y=y,mask=mask,m=m_m) # NOTE _train_step
                # tf.print(self.step.numpy(),D_loss.numpy(),G_loss.numpy())
                if (int(self.step.numpy())%self.logs_interval==0)and(int(self.step.numpy())>=1):
                    log_types,log_contents = self.make_logs(self.validation_set,_predict_step,loss={'D_loss':D_loss.numpy(),'G_loss':G_loss.numpy()})# NOTE _predict_step
                    self.logs_maker.wirte(training=True,log_types=log_types,log_contents=log_contents)
                # tf.print(f"{self.step.numpy()} {step} t1 {tf.reduce_mean(t1).numpy()}")
                self.step.assign(step) 
                self.checkpoint.save() # 依据checkpoint 自身规则自动选择与保存 
            if self.step.numpy()>=self.steps:
                break
            self.epoch.assign(epoch)
        # while(True):
        #     if (int(self.epoch.numpy())>=self.epochs)or(int(self.step.numpy())>=self.steps):#已经训练超过epochs或超过steps
        #         self.checkpoint.save()
        #         break
        #     # with tf.profiler.experimental.Trace('train', step_num=self.step, _r=1):
        #     try:
        #         # imgs,img_masks,padding_vectors = next(train_set)
        #         item = next(train_set)
        #         t1,t2,mask,m,v = item['t1'],item['t2'],item['mask'],item['patch_mask'],item['patch_padding_vector']
        #         if self._position_check():
        #             x = t1
        #             y = t2
        #             mask = mask
        #             m_m = m
        #             # tf.print(tf.reduce_mean(x),tf.reduce_mean(y),tf.reduce_mean(mask),tf.reduce_mean(m_m))
        #             D_loss,G_loss = _train_step(x=x,y=y,mask=mask,m=m_m) # NOTE _train_step
        #             # tf.print(self.step.numpy(),D_loss.numpy(),G_loss.numpy())
        #             self.step.assign_add(1)
        #             if int(self.step.numpy())%self.logs_interval==0:
        #                 log_types,log_contents = self.make_logs(self.validation_set,_predict_step,loss={'D_loss':D_loss.numpy(),'G_loss':G_loss.numpy()})# NOTE _predict_step
        #                 self.logs_maker.wirte(training=True,log_types=log_types,log_contents=log_contents)
        #             self.checkpoint.save() # 依据checkpoint 自身规则自动选择与保存  
        #         self._shadow_step.assign_add(1)
        #     except StopIteration:
        #         self.epoch.assign_add(1)
        #         self.train_dataset_iterator = iter(self.train_set)
        # tf.profiler.experimental.stop()
    #-----------------test-------------------------#
    def _checkpoint_check(self):
        status = self.checkpoint.restore_or_initialize()
        if status is not None: 
            # 实验目录logs_path已经存在且有checkpoint内容了  实验已经跑了若干次
            # 不应当再进行weight_load加载初始化权重
            return True 
        else:
            # 实验目录logs_path不存在checkpoint内容 根据需要加载权重
            self._weight_load()
            return False 
    #-----------------initial-------------------------#
    def weight_initial(self):
        status = self._weight_load()
        if status:
            print("Weight is Existed already and will not be saved!!")
            # 很重要 防止训练程序不小心改写weight
        else:
            print("Weight is not Existed and will be saved!")
            self._weight_save() #确保仅有initial可以保存初始化权重 宁可报错也不可轻易改写
    def _weight_save(self):
        time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for item in self.models_list:
            _path = self.weight_path+"/"+item.name+"/"
            item.save_weights(_path)
        for item in self.models_list:
            _path = self.weight_path+"-"+time_stamp+"/"+item.name+"/" 
            # 很重要 防止训练程序不小心改写weight的双保险
            item.save_weights(_path)
        print("Weight saved Successfully!")
    def _weight_load(self): # 该方法不应当被其余文件调用 
        try:
            for item in self.models_list:
                _path = self.weight_path+"/"+item.name+"/"
                item.load_weights(_path)
            print("Weight loaded Successfully!")
            return True
        except tf.errors.NotFoundError:
            print("Weight loading Failed! Not Found")
            return False
        except ValueError:
            raise ValueError("Weight loading Failed! Unmatched Weight. Please rename init flag and retry!")
        except:
            raise ValueError("Weight loading Failed! Unmatched Weight. Please rename init flag and retry!")
    #-----------------test-------------------------#
    def test(self):
        status = self._checkpoint_check() 
        _predict_step = self.train_process.predict_wrapper(self._predict_func)
        if status: # 存在已保存检查点
            for kept_point in self.checkpoint.checkpoints: # 2 
                self.checkpoint.restore(kept_point)
                log_types,log_contents = self.make_logs(self.test_set,_predict_step)
                self.logs_maker.wirte(training=False,log_types=log_types,log_contents=log_contents)
        else:# 不存在已保存检查点 初始测试
            log_types,log_contents = self.make_logs(self.test_set,_predict_step)
            self.logs_maker.wirte(training=False,log_types=log_types,log_contents=log_contents)
#--------------------------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    a = 10
    b = float(a)
    print(a,b)