import tensorflow as tf
__all__ = [ 
    'CustomDecay',
]
class CustomDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,initial_learning_rate,decay_steps,decay_rate,start_step=0,staircase=False,name=None):
        super(CustomDecay,self).__init__()
        self.initial_learning_rate = tf.convert_to_tensor(initial_learning_rate,dtype=tf.float32)
        self.decay_steps = tf.convert_to_tensor(decay_steps,dtype=tf.int32)
        self.decay_rate = tf.convert_to_tensor(decay_rate,dtype=tf.float32)
        self.staircase = staircase
        self.start_step = start_step
        self.name = name
    def __call__(self,step):
        over_steps = tf.maximum(tf.constant(0,dtype=tf.int32),tf.cast(step-self.start_step,tf.int32)) #step-self.start_step类型未知 所以不提前定义start_step类型未知类型
        if self.staircase:
            index = tf.cast(tf.math.floor(over_steps/self.decay_steps),tf.float32)
        else:
            index = tf.cast(over_steps/self.decay_steps,tf.float32)
        # lr*tf.cast(tf.pow(self.lr_exp_base,over),tf.float32)
        decayed_learning_rate = self.initial_learning_rate * tf.pow(self.decay_rate,index)
        return decayed_learning_rate


if __name__ == '__main__':
    import os 
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=2,
        decay_rate=0.3,
        staircase=True)
    # lr_schedule = CustomDecay(
    #     initial_learning_rate,
    #     decay_steps=2,
    #     decay_rate=0.3,
    #     start_step=10,
    #     staircase=True)
    buf1 = []
    for i in range(3):
        # opt = tf.keras.optimizers.SGD(learning_rate=1.0,name='SGD')
        opt = tf.keras.optimizers.Adam(learning_rate=1.0,beta_1=0.0,beta_2=0.9)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        opt.learning_rate = lr_schedule
        w1 = tf.Variable(10.0,trainable=True) 
        # NOTE 定义了3遍 所以如果buf1记录的是变量 
        # buf1[10个第一w1引用(是同一个Variable,值为第10次的值),
        #         10个第二w1引用(是同一个Variable,值为第20次的值),
        #         10个第三w1引用(是同一个Variable,值为第30次的值)]
        # 如果buf2记录的是变量 
        # buf2=[30个第三w2引用(是同一个Variable,值为第30次的值)] 所以buf1-buf2 != 0 
        # 因此buf1 buf2 应当记录constant
        checkpoint_directory = "./tmp/training_checkpoints"
        checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
        checkpoint = tf.train.Checkpoint(optimizer=opt,model=w1)
        # checkpoint.write(file_prefix=checkpoint_prefix)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
        for _ in range(100):
            with tf.GradientTape(persistent=True) as tape:
                loss  = 1.33*w1
                scaled_loss = opt.get_scaled_loss(loss)
            scaled_gradient = tape.gradient(scaled_loss,[w1])
            gradient = opt.get_unscaled_gradients(scaled_gradient)
            # gradient = tape.gradient(loss,[w1])
            opt.apply_gradients(zip(gradient,[w1]))
            buf1.append(w1*1)
            print(w1*1,opt.learning_rate)
        print("!!!!!!!!!!!!!!!!!!!!!!")
        # status.assert_consumed()  # Optional sanity checks.
        checkpoint.save(file_prefix=checkpoint_prefix)
    print("*****************************")
    buf2 = []
    # opt = tf.keras.optimizers.SGD(learning_rate=1.0,name='SGD')
    opt = tf.keras.optimizers.Adam(learning_rate=1.0,beta_1=0.0,beta_2=0.9)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    opt.learning_rate = lr_schedule
    w2 = tf.Variable(10.0,trainable=True)
    for _ in range(300):
        with tf.GradientTape(persistent=True) as tape:
            loss  = 1.33*w2
            scaled_loss = opt.get_scaled_loss(loss)
        scaled_gradient = tape.gradient(scaled_loss,[w2])
        gradient = opt.get_unscaled_gradients(scaled_gradient)
        # gradient = tape.gradient(loss,[w2])
        opt.apply_gradients(zip(gradient,[w2]))
        buf2.append(w2*1)
        print(w2*1,opt.learning_rate)
    _sum = tf.constant(0.0)
    for item1,item2 in zip(buf1,buf2):
        _sum += tf.reduce_sum(item1-item2)
    print(_sum)
    print("测试schedule是否可以落实 如果sum=0 表示带有schedule的optimizers可以顺利保存内部的step(即iteration)并在下次求导时正确应用schedule",_sum)
    
