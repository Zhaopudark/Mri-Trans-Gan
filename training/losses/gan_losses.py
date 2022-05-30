import logging
import tensorflow as tf
   
"""
一个好的loss计算模块
计算时对batch 维度独立
返回时按照需求返回 默认在batch维度取均值
一般的 求导永远是对无维度的单值loss求导
"""
__all__ = [
    'GanLoss',
]

class GanLoss():
    """
    原则
    返回的loss必定被minimize 所以对应于公式中的符号在本类中直接修改
    """
    def __init__(self,loss_name,args,counters_dict={}):
        self.loss_name = loss_name
        self.__counters_dict = counters_dict
        if loss_name.lower() == 'vanilla':
            self.gan_loss = _VanillaGanLoss(loss_name,counters_dict=self.__counters_dict)
        elif loss_name.lower() == 'wgan':
            self.gan_loss = _WGanLoss(loss_name,counters_dict=self.__counters_dict)
        elif loss_name.lower() == "wgan-gp":
            penalty_l = float(args['wgp_penalty_l'])
            initial_seed = int(args['wgp_initial_seed'])
            random_always = bool(args['wgp_random_always'])
            random_generator = tf.random.Generator.from_seed(initial_seed)
            self.__counters_dict['wgp_random_generator'] = random_generator # 会被模型函数中的check point 记录
            self.gan_loss = _WGanGpLoss(loss_name,counters_dict=self.__counters_dict,penalty_l=penalty_l,initial_seed=initial_seed,random_always=random_always)
        elif loss_name.lower() == 'lsgan':
            self.gan_loss = _LsGanLoss(loss_name,counters_dict=self.__counters_dict)
        elif loss_name.lower() == 'hinge':
            self.gan_loss = _HingeLoss(loss_name,counters_dict=self.__counters_dict)
        elif loss_name.lower() == 'rsgan':#判别器输出范围不明 暂时不讨论
            self.gan_loss = _RsGanLoss(loss_name,counters_dict=self.__counters_dict)
        else:
            raise ValueError("Do not support the loss: "+loss_name)
    def discriminator_loss(self,D_real,D_fake,**kwargs):
        return self.gan_loss.discriminator_loss(D_real,D_fake,**kwargs)
    def generator_loss(self,D_real,D_fake,**kwargs):
        return self.gan_loss.generator_loss(D_real,D_fake,**kwargs)
##########################################################################################################
class _VanillaGanLoss():
    def __init__(self,loss_name,counters_dict={}):
        self.__loss_name = loss_name 
        self.__counters_dict = counters_dict
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)# 输入的值已经被二值化处理
    def discriminator_loss(self,D_real,D_fake,**kwargs):
        real_part = self.cross_entropy(tf.ones_like(D_real),D_real)  # -1*log(D_real)+(-0*log(1-D_real))
        fake_part = self.cross_entropy(tf.zeros_like(D_fake),D_fake) # -0*log(D_fake)+(-1*log(1-D_fake))
        D_loss = real_part + fake_part
        return D_loss
    def generator_loss(self,D_real,D_fake,**kwargs):
        G_loss = self.cross_entropy(tf.ones_like(D_fake),D_fake)
        return G_loss
class _LsGanLoss():
    def __init__(self,loss_name,counters_dict={}):
        self.__loss_name = loss_name 
        self.__counters_dict = counters_dict
    def discriminator_loss(self,D_real,D_fake,**kwargs):
        D_loss = (tf.reduce_mean(tf.math.squared_difference(D_real,1))+\
                      tf.reduce_mean(tf.math.squared_difference(D_fake,0)))*0.5
        return D_loss
    def generator_loss(self,D_real,D_fake,**kwargs):
        G_loss = tf.reduce_mean(tf.math.squared_difference(D_fake,1))#*0.5
        return G_loss

class _HingeLoss():
    def __init__(self,loss_name,counters_dict={}):
        self.__loss_name = loss_name 
        self.__counters_dict = counters_dict
    def discriminator_loss(self,D_real,D_fake,**kwargs):
        #要求判别器的输出在-1 1区间 sigmoid变成tanh
        D_loss = tf.reduce_mean(tf.nn.relu(1.0-D_real))+tf.reduce_mean(tf.nn.relu(1.0+D_fake))
        return D_loss
    def generator_loss(self,D_real,D_fake,**kwargs):
        G_loss = -tf.reduce_mean(D_fake)
        return G_loss

class _RsGanLoss():
    def __init__(self,loss_name,counters_dict={}):
        self.__loss_name = loss_name 
        self.__counters_dict = counters_dict
    def discriminator_loss(self,D_real,D_fake,**kwargs):
        D_loss = -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(D_real-D_fake)+1e-5))
        return D_loss
    def generator_loss(self,D_real,D_fake,**kwargs):
        G_loss = -tf.reduce_mean(tf.math.log(tf.nn.sigmoid(D_fake-D_real)+1e-5))
        return G_loss

class _WGanLoss():
    def __init__(self,loss_name,counters_dict={}):
        self.__loss_name = loss_name 
        self.__counters_dict = counters_dict
    def discriminator_loss(self,D_real,D_fake,**kwargs):
        D_loss = -tf.reduce_mean(D_real)+tf.reduce_mean(D_fake)#用batch 均值逼近期望 然后依据公式 max  所以取反  -E(real)+E(fake)  做min
        # tf.print("D_loss",D_loss)
        D_loss = tf.clip_by_value(D_loss,-10.,+10.)
        return D_loss
    def generator_loss(self,D_real,D_fake,**kwargs):
        G_loss = -tf.reduce_mean(D_fake)
        # tf.print("G_loss",G_loss)
        G_loss = tf.clip_by_value(G_loss,-10.,+10.)
        return G_loss
    
class _WGanGpLoss():
    def __init__(self,loss_name,counters_dict={},penalty_l=10.0,initial_seed=None,random_always=True):
        self.__loss_name = loss_name
        self.__counters_dict = counters_dict
        self.penalty_l = penalty_l
        self.noise_shape = None
        self.step = self.__counters_dict['step']
        self.epoch = self.__counters_dict['epoch']
        self.random_generator = self.__counters_dict['wgp_random_generator']
        self.initial_seed = initial_seed
        self.random_always = random_always
    def discriminator_loss(self,D_real,D_fake,real_samples=None,fake_samples=None,D=None,condition=None):
        penalty = self.penalty_loss(real_samples,fake_samples,D,condition)
        D_loss = -tf.reduce_mean(D_real)+tf.reduce_mean(D_fake)+tf.reduce_mean(penalty) #用batch 均值逼近期望 然后依据公式 max  所以取反  -E(real)+E(fake)  做min
        # logging.getLogger(__name__).info("not cliping")

        D_loss = tf.clip_by_value(D_loss,-10.,+10.)
        logging.getLogger(__name__).info("cliping")
        # print("cliping")
        return D_loss
    def generator_loss(self,D_real,D_fake,**kwargs):
        G_loss = -tf.reduce_mean(D_fake)
        G_loss = tf.clip_by_value(G_loss,-10.,+10.)
        return G_loss
    def penalty_loss(self,real_samples,fake_samples,D,condition=None):
        if self.noise_shape is None:
            noise_shape = [real_samples.shape[0]]
            for _ in range(1,len(real_samples.shape)):
                noise_shape.append(1)
            self.noise_shape = noise_shape
        else:
            pass
        if self.noise_shape[0]!=real_samples.shape[0]:
            self.noise_shape[0] = real_samples.shape[0]
        if self.random_always:
            e = self.random_generator.uniform(shape=self.noise_shape,minval=0.0,maxval=1.0)
        else:
            self.random_generator.reset_from_seed(self.initial_seed)
            e = self.random_generator.uniform(shape=self.noise_shape,minval=0.0,maxval=1.0)
        # tf.print(tf.reduce_mean(e))

        mid_samples = e*real_samples+(1-e)*fake_samples
        if condition is not None:
            in_put = [mid_samples,condition]
        else:
            in_put = [mid_samples]
        with tf.GradientTape() as gradient_penalty:
            gradient_penalty.watch(mid_samples)
            D_mid = D(in_put=in_put,training=True,step=self.step,epoch=self.epoch)
        penalty = gradient_penalty.gradient(D_mid,mid_samples)

        # penalty_norm = self.penalty_l*tf.math.square(tf.norm(tf.reshape(penalty,shape=[self.noise_shape[0],-1]),ord=2,axis=-1)-1) #******** # 2
        penalty_norm = self.penalty_l*tf.math.square(tf.maximum(tf.norm(tf.reshape(penalty,shape=[self.noise_shape[0],-1]),ord=2,axis=-1),1.0)-1.0) # 3 是最好的wgp loss
        # penalty_norm=self.penalty_l*tf.math.square(tf.norm(tf.maximum(tf.reshape(penalty,shape=[self.noise_shape[0],-1]),1.0)-1.0,ord=2,axis=-1)) # 4
        # tf.print(penalty_norm)
        return penalty_norm

if __name__=='__main__':
    import random
    import os
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    seed = 0 
    def rrr(x):
        random.seed(x)
    random.seed(seed)
    # state = random.getstate()
    for _ in range(2000):
        state = random.getstate()
        rrr(10)
        random.setstate(state)
        print(random.randint(0,2**31-1))
        state = random.getstate()
        random.setstate(state)
    g = tf.random.Generator.from_seed(0)
    print(g.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32))
    print(g.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32))
    print(g.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32))
    g2 = tf.random.Generator.from_seed(0)
    print(g2.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32))
    print(g2.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32))
    print(g2.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32))
    # print(tf.random.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32,seed=0))
    g3 = tf.random.Generator.from_seed(0)
    print(g3.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32))
    state = g3.state
    algorithm = g3.algorithm    
    g4 = tf.random.Generator.from_state(state,algorithm)
    print(g4.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32))
    print(g4.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32))

    g = tf.random.Generator.from_seed(0)
    g.reset_from_seed(0)
    print(g.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32))
    g.reset_from_seed(0)
    print(g.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32))
    g.reset_from_seed(0)
    print(g.uniform(shape=(1,),minval=0,maxval=2**31-1,dtype=tf.int32))
    # state = g.state
    def change_dic(dic):
        dic['999'] = 10000
    dic = {}
    dic['11'] = 11
    change_dic(dic.copy())
    print(dic)

    def change_dic(__dic):
        __dic['999'] = 10000
    dic = {}
    dic['11'] = 11
    change_dic(dic)
    print(dic)

    buf1 = []
    for i in range(3):
        rg = tf.random.Generator.from_seed(0)
        checkpoint_directory = "./tmp/training_checkpoints"
        checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
        checkpoint = tf.train.Checkpoint(optimizer=rg)
        # checkpoint.write(file_prefix=checkpoint_prefix)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
        for _ in range(10):
            # rg.reset_from_seed(3)
            random_result  = rg.uniform(shape=(1,),minval=0.0,maxval=1.0)
            buf1.append(random_result)
            print(random_result)
        print("!!!!!!!!!!!!!!!!!!!!!!")
        # status.assert_consumed()  # Optional sanity checks.
        checkpoint.save(file_prefix=checkpoint_prefix)
    print("*****************************")
    buf2 = []
    rg = tf.random.Generator.from_seed(0)
    for _ in range(30):
        # rg.reset_from_seed(3)
        random_result  = rg.uniform(shape=(1,),minval=0.0,maxval=1.0)
        buf2.append(random_result)
        print(random_result)
    _sum = tf.constant(0.0)
    for item1,item2 in zip(buf1,buf2):
        _sum += tf.reduce_sum(item1-item2)
    print(_sum)
    print("测试schedule是否可以落实 如果sum=0 表示带有schedule的optimizers可以顺利保存内部的step(即iteration)并在下次求导时正确应用schedule",_sum)

