import datetime
import logging

import tensorflow as tf

from models.networks.network_selector import NetworkSelector
from training.optimizers.optimizer import Optimizer
from training.losses.gan_losses import GanLoss
from training.losses.image_losses import DualGanReconstructionLoss
from training.losses.image_losses import CycleConsistencyLoss
from training.process.train_process import TrainProcess 
from training.checkpoint.checkpoint_writer import Checkpoint
# from training.summary.summary_maker import LogsMaker
from utils.managers import SummaryDataCollections,TrainSummaryMaker,TestSummaryMaker
from utils.image.image_process import Drawer
# from training.metrics.metrics_conductor import MetricsConductor
from training.metrics.psnr import PeakSignal2NoiseRatio3D,PeakSignal2NoiseRatio2D
from training.metrics.ssim import StructuralSimilarity3D,StructuralSimilarity2D
from datasets.data_pipeline import DataPipeline

from typeguard import typechecked

__all__ = [ 
    'MriTransGan',
]
class MriTransGan():
    @typechecked
    def __init__(self,args:dict):
        #-----------------------counters-dict---------------------#
        # 计数器 在模型中定义 同时给检查点(checkpoint) 日志 (logs_maker) 和模型网络(gan_process)
        self.steps = int(args['steps']) # 目标训练步数
        self.epochs = int(args['epochs']) # 目标训练周期
        self.step = tf.Variable(0,dtype=tf.int64,trainable=False)
        self.epoch = tf.Variable(0,dtype=tf.int64,trainable=False)
        self.counters_dict = {'step':self.step,'epoch':self.epoch}
        #------------------------data------------------------#
        self.dataset = DataPipeline(args,counters=self.counters_dict)
        self.train_set,self.test_set,self.validation_set = self.dataset()
        self.patch_combiner = self.dataset.patch_combine_generator
        self.input_shape = [1,64,64,64,1]
        #------------------------model------------------------#
        _mixed_precision = bool(args['mixed_precision'])
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
        _G_learning_rate = float(args['G_learning_rate'])
        _D_learning_rate = float(args['D_learning_rate'])
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
        _gan_loss_name = args['gan_loss_name']
        self.gan_loss = GanLoss(_gan_loss_name,args=args,counters_dict=self.counters_dict) 
        self.cycle_loss = CycleConsistencyLoss(args=args)
        self.rec_loss = DualGanReconstructionLoss(args=args)
        #----------------logs-checkpoint-config--------------------#
        self.weight_path = args['weight_path'] # 初始化时加载
        self.logs_path = args['logs_path']
        self.logs_interval = int(args['logs_interval'])
        _checkpoint_interval = int(args['checkpoint_interval'])
        _checkpoint_max_keep = int(args['checkpoint_max_keep'])
        _metrics = args['metrics']
        self.checkpoint = Checkpoint(counters_dict=self.counters_dict,
                                     path=self.logs_path,
                                     max_to_keep=_checkpoint_max_keep,
                                     checkpoint_interval=_checkpoint_interval,
                                     optimizers = self.optimizers_list,
                                     models = self.models_list)
        #--------------------------metrics-------------------------#
        self.metrics = {
            'psnr2d':PeakSignal2NoiseRatio2D(),
            'psnr3d':PeakSignal2NoiseRatio3D(),
            'ssim2d':StructuralSimilarity2D(),
            'ssim3d':StructuralSimilarity3D(),
        }
        # self.metrics = MetricsConductor(_metrics) # NOTE 独立于LogsMaker之外 迫使LogsMaker只负责记录而避免设计具体的计算或者绘图细节
        self.drawer = Drawer() # NOTE 独立于LogsMaker之外 迫使LogsMaker只负责记录而避免设计具体的计算或者绘图细节           
        # self.logs_maker = LogsMaker(counters_dict=self.counters_dict,path=self.logs_path)
        self.train_summary_maker = TrainSummaryMaker(self.logs_path)
        self.test_summary_maker = TestSummaryMaker(self.logs_path)

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
        def gen(dataset):
            bar = tf.keras.utils.Progbar(dataset.cardinality(),width=30,verbose=1,interval=0.5,stateful_metrics=None,unit_name='test_or_init_step')
            for item in dataset:
                x = item['t1']
                y = item['t2']
                mask = item['mask']
                y_,x_ = predict_func(x=x,y=y,mask=mask)
                bar.add(1)
                yield {
                        'x':tf.squeeze(x,axis=0),
                        'y':tf.squeeze(y,axis=0),
                        'y_':tf.squeeze(y_,axis=0),
                        'x_':tf.squeeze(x_,axis=0),
                        'mask':tf.squeeze(mask,axis=0), 
                        'total_ranges':tf.squeeze(item['total_ranges'],axis=0),
                        'valid_ranges':tf.squeeze(item['valid_ranges'],axis=0),
                        'patch_sizes':tf.squeeze(item['patch_sizes'],axis=0),
                        'patch_ranges':tf.squeeze(item['patch_ranges'],axis=0),
                        'patch_index':tf.squeeze(item['patch_index'],axis=0),
                        'max_index':tf.squeeze(item['max_index'],axis=0),       
                }
        def combiner_wrapper(gen):
            yield from self.patch_combiner(gen)
        def post_precess_wrapper(gen):
            def unit(x,mask):
                x = x*mask
                _min = x.numpy().min()
                _max = x.numpy().max()
                x = (x-_min)/(_max-_min)
                x = x*mask
                _min = x.numpy().min()
                _max = x.numpy().max()
                return x
            for i,item in enumerate(gen):
                x  = item['x']
                y_ = item['y_']
                x_ = item['x_']
                y  = item['y']
                mask = item['mask']
                logging.getLogger(__name__).debug(f'{i} 1x %s %s %s',x.shape,x.numpy().min(),x.numpy().max())
                logging.getLogger(__name__).debug(f'{i} 1y_ %s %s %s',y_.shape,y_.numpy().min(),y_.numpy().max())
                logging.getLogger(__name__).debug(f'{i} 1x_ %s %s %s',x_.shape,x_.numpy().min(),x_.numpy().max())
                logging.getLogger(__name__).debug(f'{i} 1y %s %s %s',y.shape,y.numpy().min(),y.numpy().max())
                x = unit(x,mask=mask)
                y_ = unit(y_,mask=mask)
                x_ = unit(x_,mask=mask)
                y = unit(y,mask=mask)
                logging.getLogger(__name__).debug(f'{i} 2x %s %s %s',x.shape,x.numpy().min(),x.numpy().max())
                logging.getLogger(__name__).debug(f'{i} 2y_ %s %s %s',y_.shape,y_.numpy().min(),y_.numpy().max())
                logging.getLogger(__name__).debug(f'{i} 2x_ %s %s %s',x_.shape,x_.numpy().min(),x_.numpy().max())
                logging.getLogger(__name__).debug(f'{i} 2y %s %s %s',y.shape,y.numpy().min(),y.numpy().max())
                yield {'x':x[tf.newaxis,...],'y_':y_[tf.newaxis,...],'x_':x_[tf.newaxis,...],'y':y[tf.newaxis,...]}
        yield from post_precess_wrapper(combiner_wrapper(gen(dataset)))
    def metrics_conductor(self,inputs:dict[str,tf.Tensor])->dict[str,int|float]:
        buf = {}
        self.metrics['psnr3d'](inputs['x'],inputs['x_'],max_val=1.0)
        buf['x_psnr3d'] = self.metrics['psnr3d'].result().numpy()
        self.metrics['psnr3d'](inputs['y'],inputs['y_'],max_val=1.0)
        buf['y_psnr3d'] = self.metrics['psnr3d'].result().numpy()
        self.metrics['ssim3d'](inputs['x'],inputs['x_'],max_val=1.0)
        buf['x_ssim3d'] = self.metrics['ssim3d'].result().numpy()
        self.metrics['ssim3d'](inputs['y'],inputs['y_'],max_val=1.0)
        buf['y_ssim3d'] = self.metrics['ssim3d'].result().numpy()
        return buf

    def _draw_pre_process(self,x):# 可扩展的 支持医学图像 RGB自然图像的
        if len(x.shape)==5:#'BDHWC':
            slice_len = x.shape[1]
            t= slice_len//2 # 以序列中部元素或者中部偏右元素为中心
            return x[0,t,...,0]
        else:
            raise ValueError(f"Unexpected shape {x.shape}!")
    class _metric_apply():
        def __init__(self,indicate):
            self.name = indicate[0]+" and "+indicate[1]
            self.indicate = indicate
        def __call__(self,metric,images) :
            return metric(images[self.indicate[0]],images[self.indicate[1]]).numpy()
    def validate_or_test_step(self,dataset,predict_func)->list[SummaryDataCollections]:
        image_record = SummaryDataCollections(summary_type='image',name='slice')
        metrics_records = None
        for i,out_put in enumerate(self.combination(dataset,predict_func)):
            grabed = tf.nest.map_structure(self._draw_pre_process,out_put)
            image_record[f"{i}"] = self.drawer.dict2img(grabed)
            metrics_single_dict = self.metrics_conductor(out_put)
            if metrics_records is None:
                metrics_records = [SummaryDataCollections(summary_type='scalar',name=f"{item}") for item in metrics_single_dict.keys()]
            for metrics_record,value in zip(metrics_records,metrics_single_dict.values()):
                metrics_record[f"{i}"] = value
        return [image_record]+metrics_records
    #------------------predict-------------------------#
    def _predict_func(self,x,y,mask):
        y_ = self.G0(in_put=[x,mask],training=False,step=self.step,epoch=self.epoch)
        x_ = self.G1(in_put=[y,mask],training=False,step=self.step,epoch=self.epoch)
        return y_,x_
    #------------------train-------------------------#
    def _loss_func(self,x,y,mask):
        # tf.print(self.step)
        y_       = self.G0(in_put=[x,mask],training=True,step=self.step,epoch=self.epoch)
        # tf.print(tf.reduce_mean(y_),self.step)
        D_real_0,buf_real_0 = self.D0(in_put=[y,mask],buf_flag=True,training=True,step=self.step,epoch=self.epoch)
        D_fake_0,buf_fake_0 = self.D0(in_put=[y_,mask],buf_flag=True,training=True,step=self.step,epoch=self.epoch)
        # tf.print(tf.reduce_mean(D_real_0),self.step)
        # tf.print(tf.reduce_mean(D_fake_0),self.step)

        x_       = self.G1(in_put=[y,mask],training=True,step=self.step,epoch=self.epoch)
        D_real_1,buf_real_1 = self.D1(in_put=[x,mask],buf_flag=True,training=True,step=self.step,epoch=self.epoch)
        D_fake_1,buf_fake_1 = self.D1(in_put=[x_,mask],buf_flag=True,training=True,step=self.step,epoch=self.epoch)

        x__      = self.G1(in_put=[y_,mask],training=True,step=self.step,epoch=self.epoch)
        y__      = self.G0(in_put=[x_,mask],training=True,step=self.step,epoch=self.epoch)

        cycle_loss = self.cycle_loss.call(x=x,x__=x__,y=y,y__=y__)
        rec_loss = self.rec_loss.call(x=x,x_=x_,y=y,y_=y_,xd=buf_real_1,x_d=buf_fake_1,yd=buf_real_0,y_d=buf_fake_0)
        
        G_loss = self.gan_loss.generator_loss(D_real=D_real_0,D_fake=D_fake_0)+\
                    self.gan_loss.generator_loss(D_real=D_real_1,D_fake=D_fake_1)+\
                    cycle_loss+rec_loss
                    
        D_loss = self.gan_loss.discriminator_loss(D_real=D_real_0,D_fake=D_fake_0,real_samples=y,fake_samples=y_,D=self.D0)+\
                    self.gan_loss.discriminator_loss(D_real=D_real_1,D_fake=D_fake_1,real_samples=x,fake_samples=x_,D=self.D1)
        # tf.print(D_loss,G_loss)
        return [D_loss,G_loss] 
    def train_step(self):
        raise ValueError("train_step must be reload!!!")
    def test_step(self):
        raise ValueError("test_step must be reload!!!")
    def train(self):
        self._checkpoint_check()
        # tf.profiler.experimental.start(self.logs_path) # TODO
        _train_step = self.train_process.train_wrapper(
                          self._loss_func,
                          optimizer_list=[self.Do0,self.Go0],
                          variable_list=[self.D0.trainable_variables+self.D1.trainable_variables,self.G0.trainable_variables+self.G1.trainable_variables])
        _predict_step = self.train_process.predict_wrapper(self._predict_func)
        # self.step 代表已完成的step
        # self.epoch 代表已完成的epoch
        for epoch in range(self.epoch.numpy()+1,self.epochs+1):
            train_bar = tf.keras.utils.Progbar(self.train_set.cardinality()*self.epochs,width=30,verbose=1,interval=0.5,stateful_metrics=None,unit_name='train_step')
            for step,item in zip(range(self.step.numpy()+1,self.steps+1),self.train_set):
                t1,t2,mask = item['t1'],item['t2'],item['mask']
                x = t1
                y = t2
                mask = mask
                # tf.print(tf.reduce_mean(x),tf.reduce_mean(y),tf.reduce_mean(mask),tf.reduce_mean(m_m))
                D_loss,G_loss = _train_step(x=x,y=y,mask=mask) # NOTE _train_step
                # tf.print(self.step.numpy(),D_loss.numpy(),G_loss.numpy())
                if (int(self.step.numpy())%self.logs_interval==0)and(int(self.step.numpy())>=1):
                    records = self.validate_or_test_step(self.validation_set,_predict_step)
                    self.train_summary_maker(records,step=self.step)

                    # log_types,log_contents = self.make_logs(self.validation_set,_predict_step,loss={'D_loss':D_loss.numpy(),'G_loss':G_loss.numpy()})# NOTE _predict_step
                    # self.logs_maker.wirte(training=True,log_types=log_types,log_contents=log_contents)
                # tf.print(f"{self.step.numpy()} {step} t1 {tf.reduce_mean(t1).numpy()}")
                self.step.assign(step) 
                self.checkpoint.save() # 依据checkpoint 自身规则自动选择与保存 
                train_bar.update(step)
            if self.step.numpy()>=self.steps:
                break
            self.epoch.assign(epoch)
       
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
            logging.getLogger(__name__).info("Weight is Existed already and will not be saved!!")
            # 很重要 防止训练程序不小心改写weight
        else:
            logging.getLogger(__name__).info("Weight is not Existed and will be saved!")
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
        logging.getLogger(__name__).info("Weight saved Successfully!")
    def _weight_load(self): # 该方法不应当被其余文件调用 
        try:
            for item in self.models_list:
                _path = self.weight_path+"/"+item.name+"/"
                item.load_weights(_path)
            logging.getLogger(__name__).info("Weight loaded Successfully!")
            return True
        except tf.errors.NotFoundError:
            logging.getLogger(__name__).info("Weight loading Failed! Not Found")
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
                records = self.validate_or_test_step(self.test_set,_predict_step)
                self.test_summary_maker(records,step=self.step)

                # log_types,log_contents = self.make_logs(self.test_set,_predict_step)
                # self.logs_maker.wirte(training=False,log_types=log_types,log_contents=log_contents)
        else:# 不存在已保存检查点 初始测试
            records = self.validate_or_test_step(self.test_set,_predict_step)
            self.test_summary_maker(records,step=self.step)
            # log_types,log_contents = self.make_logs(self.test_set,_predict_step)
            # self.logs_maker.wirte(training=False,log_types=log_types,log_contents=log_contents)
