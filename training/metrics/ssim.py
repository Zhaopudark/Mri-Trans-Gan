import tensorflow as tf

class StructuralSimilarity2D(tf.keras.metrics.Metric):
    def __init__(self,name='structural_similarity_2d',**kwargs) -> None:
        super(StructuralSimilarity2D,self).__init__(name=name,**kwargs)
        self.ssim = self.add_weight(name='ssim',initializer='zeros')
    def update_state(self,
        y_true,
        y_pred,
        max_val,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        compensation=1.0,
        sample_weight=None):

        tf.debugging.assert_shapes(
            shapes=[
                    (y_true,('N','H','W','C')),
                    (y_pred,('N','H','W','C'))
                ]
        )

        values = _ssim2d(y_true,y_pred,max_val,filter_size,filter_sigma,k1,k2,compensation)
        if sample_weight is not None:
            sample_weight = tf.convert_to_tensor(sample_weight)
            sample_weight = tf.cast(sample_weight,self.dtype)
            tf.debugging.assert_shapes(shapes=[
                (values,('N',)),
                (sample_weight,('N',))
                ])
            values = tf.multiply(values,sample_weight)
        self.ssim.assign(tf.reduce_mean(values))
    def result(self):
        return self.ssim

class StructuralSimilarity3D(tf.keras.metrics.Metric):
    def __init__(self,name='structural_similarity_3d',**kwargs) -> None:
        super(StructuralSimilarity3D,self).__init__(name=name,**kwargs)
        self.ssim = self.add_weight(name='ssim',initializer='zeros')
    def update_state(self,
        y_true,
        y_pred,
        max_val,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        compensation=1.0,
        sample_weight=None):

        tf.debugging.assert_shapes(
            shapes=[
                    (y_true,('N','D','H','W','C')),
                    (y_pred,('N','D','H','W','C'))
                ]
        )

        values = _ssim3d(y_true,y_pred,max_val,filter_size,filter_sigma,k1,k2,compensation)
        if sample_weight is not None:
            sample_weight = tf.convert_to_tensor(sample_weight)
            sample_weight = tf.cast(sample_weight,self.dtype)
            tf.debugging.assert_shapes(shapes=[
                (values,('N',)),
                (sample_weight,('N',))
                ])
            values = tf.multiply(values,sample_weight)
        self.ssim.assign(tf.reduce_mean(values))
    def result(self):
        return self.ssim

class Ssim3D():
    def __init__(self):
        self.name = 'ssim3D'
        self.mean = tf.keras.metrics.Mean(dtype=tf.float32)
    def __call__(self,x,y):
        self.mean.reset_states()
        if x.numpy().min()<0:
            raise ValueError("SSIM cal out of range!")
        if y.numpy().min()<0:
            raise ValueError("SSIM cal out of range!")
        if x.numpy().max()>1.0:
            raise ValueError("SSIM cal out of range!")
        if y.numpy().max()>1.0:
            raise ValueError("SSIM cal out of range!")
        if len(x.shape)!=len(y.shape):
            raise ValueError("x and y must in the same shape!")
        if x.shape[0]!=1:
            raise ValueError("Calculation error! Only suppotr [1,x,x,x] or [1,x,x,x,x] shape")#本方法支持
        if len(x.shape)==5:
            self.mean(_ssim3d(x,y)) 
        if len(x.shape)==4:
            self.mean(_ssim2d(x,y))
        result = self.mean.result()
        self.mean.reset_states()
        return result


def _ssim2d(y_true,y_pred,max_val=1.0,filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03,compensation=1.0):
    """
    The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
    but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
    即做无偏估计时,传入(1 - \sum_i w_i ^ 2) 一般情况下为1.0

    SSIM计算
    对于一张有限大小的二维单通道图像A,B 拥有X Y轴
    X Y是为两个独立的随机变量,取值是一个合适的范围内的离散整数,可以视X,Y为图像每个像素点的坐标
    函数 Q=g(X,Y) 确定了新的随机变量, Q值代表对应X Y下的像素值.
    Q的分布列由X Y的联合分布列导出 
    一般的 对于采集的函数  我们认为Q是均匀分布, 即X Y各自也是均匀分布.
    分布列必然包含了所有变量取值与取值概率
    但在计算SSIM时 考虑到视觉特性 我们认为X Y是独立的高斯分布 Q分布列满足二维高斯分布
    由此, 我们在已知图像A与图像B的Q分布列(包括X Y分布列)时,计算某个值,反应图像A B的结构相似性SSIM.

    SSIM 由亮度(流明)(Luminance) 对比度(contrast) 结构(structure)进行联合计算 
    亮度由平均灰度(像素值)计算
    对比度由标准差或方差计算
    结构由均值标准差进行计算定义
    定义可见 https://blog.csdn.net/ecnu18918079120/article/details/60149864

    依据计算公式 常数C1=(k1 * max_val)**2 C2=(k2 * max_val)**2 用于保持数值稳定
    亮度计算(2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
    由于卷积(深度卷积)的特性, 以及期望可以等价于概率值与随机变量值乘积求和,这一求和的过程恰好可以用指定高斯核的卷积操作描述
    必须由深度卷积(逐通道)的原因是,正常卷积会在卷积的时候就计算出全通道的和,等价于计算出了通道均值的某个常数倍,不应该在此处取得均值,只应该在最后对通道和各尺度取均值
    所以,以下计算可以与亮度计算等价
    mean0 = _reducer(x)
    mean1 = _reducer(y)
    num0 = mean0 * mean1 * 2.0
    den0 = tf.math.square(mean0) + tf.math.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    然后,取SSIM中对比度核结构的特殊情况,即其指数为1,数值稳定系数C3=C2/2
    则可提前将对比度与结构相乘而合并,记为CS,CS计算公式为 
    SSIM contrast-structure measure is
    (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
             = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    通过深度卷积又可以与下式等价

    num1 = _reducer(x*y) * 2.0
    den1 = _reducer(tf.math.square(x)+tf.math.square(y))
    compensation = 1
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    luminance*cs 即为图像A与B的ssim取值(对称性,有界性,最大值唯一性).
    当考察超过高斯核大小的,以及多个通道的二维图像时,
    SSIM会在X Y轴上扩展为padding=VALID的大小,在通道上扩展为通道尺度
    实际计算时,会将高斯核在通道上复制从而保证一次卷积即可计算
    计算完成后,需要将SSIM在X Y 轴以及通道尺度上全部取均值 
    """
    tf.debugging.assert_greater_equal(y_true,0.0)
    tf.debugging.assert_less_equal(y_true,max_val)
    tf.debugging.assert_greater_equal(y_pred,0.0)
    tf.debugging.assert_less_equal(y_pred,max_val)

    w = _fspecial_gauss_2d(size=filter_size,sigma=filter_sigma) # 获得高斯核
    w = tf.tile(w,multiples=[1,1,y_true.shape[-1],1])# 在输入通道上扩展 高斯核的最后一维是输出倍数,专门用于深度卷积 默认就为1不变
    def _reducer(x):
        return tf.nn.depthwise_conv2d(x,filter=w,strides=[1,1,1,1],padding='VALID')
    c1 = (k1*max_val)**2
    c2 = (k2*max_val)**2
    mean0 = _reducer(y_true)
    mean1 = _reducer(y_pred)
    num0 = mean0*mean1*2.0
    den0 = tf.math.square(mean0)+tf.math.square(mean1)
    luminance = (num0+c1)/(den0+c1)
    num1 = _reducer(y_true*y_pred)*2.0
    den1 = _reducer(tf.math.square(y_true)+tf.math.square(y_pred))
    compensation = 1 #
    c2 *= compensation 
    cs = (num1-num0+c2)/(den1-den0+c2)
    return tf.reduce_mean(luminance*cs,[-3,-2,-1])

def _ssim3d(y_true,y_pred,max_val=1.0,filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03,compensation=1.0):
    """
    The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
    but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
    即做无偏估计时,传入(1 - \sum_i w_i ^ 2) 一般情况下为1.0

    SSIM计算 针对3D图像
    对于一张有限大小的三维单通道图像A,B 拥有X Y Z轴
    X Y Z是为三个独立的随机变量,取值是一个合适的范围内的离散整数,可以视X,Y,Z为图像每个像素点的坐标
    函数 Q=g(X,Y,Z) 确定了新的随机变量, Q值代表对应X Y Z下的像素值(体素值).
    Q的分布列由X Y Z的联合分布列导出 
    一般的 对于采集的函数  我们认为Q是均匀分布, 即X Y Z各自也是均匀分布.
    分布列必然包含了所有变量取值与取值概率
    但在计算SSIM时 考虑到视觉特性 我们认为X Y Z是独立的高斯分布 Q分布列满足三维高斯分布
    由此, 我们在已知图像A与图像B的Q分布列(包括X Y Z分布列)时,计算某个值,反应图像A B的结构相似性SSIM.

    3D SSIM的计算过程与2D一致,仅仅需要3D高斯核,并将2D卷积转为3D卷积
    luminance*cs 即为图像A与B的ssim取值(对称性,有界性,最大值唯一性).
    当考察超过高斯核大小的,以及多个通道的3维图像时,
    SSIM会在X Y Z轴上扩展为padding=VALID的大小,在通道上扩展为通道尺度
    实际计算时,会将高斯核在通道上复制从而保证一次卷积即可计算
    计算完成后,需要将SSIM在X Y Z轴以及通道尺度上全部取均值 
    """

    w = _fspecial_gauss_3d(size=filter_size,sigma=filter_sigma) # 获得高斯核
    w = tf.tile(w,multiples=[1,1,1,y_true.shape[-1],1])# 在输入通道上扩展 高斯核的最后一维是输出倍数,专门用于深度卷积 默认就为1不变
    # 但是 3D的深度卷积并不支持  目前不涉及3D多通道 因此只有一个深度 我们稍加限制以安全稳定地实现暂时是实验目的
    # def _reducer(x):
    #     return tf.nn.depthwise_conv3d(x,filter=w,strides=[1,1,1,1,1],padding='VALID')
    assert w.shape[-2]==1
    def _reducer(x):
        return tf.nn.conv3d(x,filters=w,strides=[1,1,1,1,1],padding='VALID') #输入通道恒为1时,卷积结果等价于深度卷积的结果
    c1 = (k1*max_val)**2
    c2 = (k2*max_val)**2
    mean0 = _reducer(y_true)
    mean1 = _reducer(y_pred)
    num0 = mean0*mean1*2.0
    den0 = tf.math.square(mean0)+tf.math.square(mean1)
    luminance = (num0+c1)/(den0+c1)

    num1 = _reducer(y_true*y_pred)*2.0
    den1 = _reducer(tf.math.square(y_true)+tf.math.square(y_pred))
    compensation = 1 #
    c2 *= compensation 
    cs = (num1-num0+c2)/(den1-den0+c2)
    return tf.reduce_mean(luminance*cs,[-4,-3,-2,-1])

def _fspecial_gauss_2d(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function. 复现二维高斯核的计算过程"""
    size = tf.convert_to_tensor(size, tf.int32)
    sigma = tf.convert_to_tensor(sigma)
    coords = tf.cast(tf.range(size), sigma.dtype)
    coords -= tf.cast(size - 1, sigma.dtype) / 2.0
    g = tf.math.square(coords)
    g *= -0.5 / tf.math.square(sigma) # 生成1维高斯核的指数部分
    g = tf.reshape(g, shape=[1, -1]) + tf.reshape(g, shape=[-1, 1]) # 由1维合成2维高斯核的指数部分 类似于meshgrid的点对点但是已经是具体有指数意义的值
    g = tf.reshape(g, shape=[1, -1])  # For tf.nn.softmax()
    g = tf.nn.softmax(g) # 2维高斯分布的1维向量形式 (softmax经过两步, 第一是加入了exp(·)完成了高斯分布的基本形式,第二是使得有限的矩阵大小使得元素和为1,即估计出了一个指定大小的标准3维正太分布)
    # print(g.shape)
    return tf.reshape(g, shape=[size, size, 1, 1]) # 拉成二维高斯核形式,并赋予卷积需要的输入输出维度 默认为1

def _fspecial_gauss_3d(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    size = tf.convert_to_tensor(size, tf.int32)
    sigma = tf.convert_to_tensor(sigma)
    coords = tf.cast(tf.range(size), sigma.dtype)
    coords -= tf.cast(size - 1, sigma.dtype) / 2.0
    g = tf.math.square(coords)
    g *= -0.5/ tf.math.square(sigma) # 生成1维高斯核的指数部分 需要注意0.5系数不变 3维甚至多维高斯分布中 公式的指数部分的分母中依旧是0.5  https://www.cnblogs.com/bingjianing/p/9117330.html
    g = tf.reshape(g,shape=[1,1,-1])+tf.reshape(g, shape=[1,-1,1])+tf.reshape(g,shape=[-1,1,1]) # 由1维合成3维高斯核的指数部分 类似于meshgrid的点对点但是已经是具体有指数意义的值
    g = tf.reshape(g, shape=[1,-1])  # For tf.nn.softmax()
    g = tf.nn.softmax(g) # 3维高斯分布的1维向量形式 (softmax经过两步, 第一是加入了exp(·)完成了高斯分布的基本形式,第二是使得有限的矩阵大小使得元素和为1,即估计出了一个指定大小的标准3维正太分布)
    # print(g.shape)
    return tf.reshape(g, shape=[size,size,size,1,1]) # 拉成三维高斯核形式,并赋予卷积需要的输入输出维度 默认为1
#--------------------------------------------------------------------------------------------------------------------------------------#
if __name__ =='__main__':
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    x1 = tf.random.uniform(shape=[1,128,128,128],minval=0,maxval=1)
    x2 = tf.random.uniform(shape=[1,128,128,128],minval=0,maxval=1)
    # w  = tf.random.normal(shape=[3,3,3,1])
    # _ssim(x1,x2)
    # print(tf.nn.depthwise_conv2d(x1,filter=w,strides=[1,1,1,1],padding='SAME'))

    ssim = Ssim3D()
    print(m1:=ssim(x1,x2))
    # print(m2:=_ssim2d(x1,x2))
    print(m3:=tf.image.ssim(x1,x2,max_val=1.0,filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03))
    # x1 = tf.reshape(x1,x1.shape[:]+[1])
    # x2 = tf.reshape(x2,x2.shape[:]+[1])
    # print(m4:=_ssim3d(x1,x2))

    # for i in range(100):
        
    #     x1 = tf.random.uniform(shape=[1,128,128,128],minval=0,maxval=1)
    #     x2 = tf.random.uniform(shape=[1,128,128,128],minval=0,maxval=1)
    #     m2 = _ssim2d(x1,x2)
    #     x1 = tf.reshape(x1,x1.shape[:]+[1])
    #     x2 = tf.reshape(x2,x2.shape[:]+[1])
    #     m4 =_ssim3d(x1,x2)
    #     if m2 < m4:
    #         print(i,m2,m4)





   
    
