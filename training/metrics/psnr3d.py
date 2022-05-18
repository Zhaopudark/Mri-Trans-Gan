import tensorflow as tf 
from typeguard import typechecked
def _psnr(x,y,max_val=1.0):
    """
    psnr的一般定义式为
    10.0*_log10(max_val**2/_mse(x,y))
    为了和tf原生psnr保持一致的计算数值并减少计算量
    将公式变为
    20.0*_log10(max_val)-10*_log10(_mse(x,y))
    tf.math.log(10.0)需要输入浮点型
    """
    mse = tf.reduce_mean(tf.math.square(x-y),axis=list(range(1,len(x.shape))))
    a = 20*tf.math.log(max_val)/tf.math.log(10.0)
    b = 10/tf.math.log(10.0)*tf.math.log(mse)
    return tf.math.subtract(a,b,name='psnr')
class Psnr():
    """
    Calculate the psnr of 
    For simplicity, in
    """
    @typechecked
    def __init__(self,domain:tuple[int|float,int|float]):
        self.name = 'psnr'
        self.domain = domain
    @property
    def domain_min(self):
        if not hasattr(self,'_domain_min'):
            self._domain_min = min(self.domain)
        return self._domain_min
    @property
    def domain_max(self):
        if not hasattr(self,'_domain_max'):
            self._domain_max = max(self.domain)
        return self._domain_max
    @property
    def domain_scale(self):
        if not hasattr(self,'_domain_scale'):
            self._domain_scale = self.domain_max-self.domain_min
        return self._domain_scale
    @property
    def domain_offset(self):
        if not hasattr(self,'_domain_offset'):
            self._domain_offset = self.domain_min
        return self._domain_offset
    def _re_domain(self,x):
        if tf.reduce_min(x)<min(self.domain) or tf.reduce_max(x)>max(self.domain):
            raise ValueError(f"PSNR inputs x should within domain:{self.domain}")
        return (x-self.domain_offset)/self.domain_scale
    def __call__(self,x,y):
        return _psnr(self._re_domain(x),self._re_domain(y))


class Psnr3D():
    """
    Calculate the psnr of 
    For simplicity, in
    """
    def __init__(self):
        self.name = 'psnr3D'
        self.mean = tf.keras.metrics.Mean(dtype=tf.float32)
    def __call__(self,x,y):
        self.mean.reset_states()
        if x.numpy().min()<0:
            raise ValueError("PSNR cal out of range!")
        if y.numpy().min()<0:
            raise ValueError("PSNR cal out of range!")
        if x.numpy().max()>1.0:
            raise ValueError("PSNR cal out of range!")
        if y.numpy().max()>1.0:
            raise ValueError("PSNR cal out of range!")
        if len(x.shape)!=len(y.shape):
            raise ValueError("x and y must in the same shape!")
        if x.shape[0]!=1:
            raise ValueError("Calculation error! Only suppotr [1,x,x,x] or [1,x,x,x,x] shape")#本方法支持
            # raise ValueError("Calculation error! Only suppotr [1,x,x,x] or [x,x,x] shape") tf.image.ssim仅支持
        if len(x.shape)==5:
            self.mean(_psnr(x,y,max_val=1.0))
        if len(x.shape)==4:
            self.mean(_psnr(x,y,max_val=1.0))
        result = self.mean.result()
        self.mean.reset_states()
        return result

# def _mse(x,y): # 均方差的计算采用更直接的squared_difference更加接近psnr真值
#     # return tf.reduce_mean(tf.math.square(x-y),axis=list(range(1,len(x.shape))))
#     return tf.reduce_mean(tf.math.squared_difference(x,y),axis=list(range(1,len(x.shape))))
# def _log10(x):
#     return tf.math.log(x)/tf.math.log(10.0)




if __name__ =='__main__':
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    x1 = tf.random.uniform(shape=[1,128,128,128],minval=0,maxval=1)
    x2 = tf.random.uniform(shape=[1,128,128,128],minval=0,maxval=1)
    # psnr = Psnr(domain=(0.0,1.0))
    psnr = Psnr3D()
    print(psnr(x1,x2))
    print(_psnr(x1,x2,max_val=1.0))
    print(tf.image.psnr(x1,x2,max_val=1.0))
    x1 = tf.reshape(x1,x1.shape[:]+[1])
    x2 = tf.reshape(x2,x2.shape[:]+[1])
    print(_psnr(x1,x2,max_val=1.0))

    
