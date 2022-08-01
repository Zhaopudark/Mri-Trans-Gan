import tensorflow as tf 
from utils.operations import mse

def _psnr(y_true,y_pred,max_val=1.0):
    """
    psnr的一般定义式为
    10.0*_log10(max_val**2/_mse(x,y))
    为了和tf原生psnr保持一致的计算数值并减少计算量
    将公式变为
    20.0*_log10(max_val)-10*_log10(_mse(x,y))
    tf.math.log(10.0)需要输入浮点型
    """
    tf.debugging.assert_greater_equal(y_true,0.0)
    tf.debugging.assert_less_equal(y_true,max_val)
    tf.debugging.assert_greater_equal(y_pred,0.0)
    tf.debugging.assert_less_equal(y_pred,max_val)

    a = 20*tf.math.log(max_val)/tf.math.log(10.0)
    b = 10/tf.math.log(10.0)*tf.math.log(mse(y_true,y_pred))
    return tf.math.subtract(a,b,name='psnr')

class _PeakSignal2NoiseRatio(tf.keras.metrics.Metric):
    def __init__(self,name='peak_signal_to_noise_ratio',**kwargs) -> None:
        super(_PeakSignal2NoiseRatio,self).__init__(name=name,**kwargs)
        self.psnr = self.add_weight(name='psnr',initializer='zeros')
    def update_state(self,y_true,y_pred,max_val,sample_weight=None):
        values = _psnr(y_true,y_pred,max_val)
        if sample_weight is not None:
            sample_weight = tf.convert_to_tensor(sample_weight)
            sample_weight = tf.cast(sample_weight,self.dtype)
            tf.debugging.assert_shapes(shapes=[
                (values,('N',)),
                (sample_weight,('N',))
                ])
            values = tf.multiply(values,sample_weight)
        self.psnr.assign(tf.reduce_mean(values))
    def result(self):
        return self.psnr

class PeakSignal2NoiseRatio2D(_PeakSignal2NoiseRatio):
    def __init__(self,name='peak_signal_to_noise_ratio_2D',**kwargs) -> None:
        super().__init__(name=name,**kwargs)
    def update_state(self,y_true,y_pred,max_val,sample_weight=None):
        tf.debugging.assert_shapes(
            shapes=[
                    (y_true,('N','H','W','C')),
                    (y_pred,('N','H','W','C'))
                ]
        )
        super().update_state(y_true,y_pred,max_val,sample_weight=None)

class PeakSignal2NoiseRatio3D(_PeakSignal2NoiseRatio):
    """
    In PSNR calculation, there is no difference between `2D` and `3D`.
    """
    def __init__(self,name='peak_signal_to_noise_ratio_3D',**kwargs) -> None:
        super().__init__(name=name,**kwargs) 
    def update_state(self,y_true,y_pred,max_val,sample_weight=None):
        tf.debugging.assert_shapes(
            shapes=[
                    (y_true,('N','D','H','W','C')),
                    (y_pred,('N','D','H','W','C'))
                ]
        )
        super().update_state(y_true,y_pred,max_val,sample_weight=None)

if __name__ =='__main__':
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    x1 = tf.random.uniform(shape=[7,128,128,5],minval=0,maxval=1)
    x2 = tf.random.uniform(shape=[7,128,128,5],minval=0,maxval=1)
    # psnr = Psnr(domain=(0.0,1.0))
    # psnr = Psnr3D()
    # print(psnr(x1,x2))
    # print(_psnr(x1,x2,max_val=1.0))
    print(tf.image.psnr(x1,x2,max_val=1.0))
    print(tf.image.psnr(x1*255.0,x2*255.0,max_val=255.0))
    # x1 = tf.reshape(x1,x1.shape[:]+[1])
    # x2 = tf.reshape(x2,x2.shape[:]+[1])
    # print(_psnr(x1,x2,max_val=1.0))

    psnr = PeakSignal2NoiseRatio2D()
    sample_weight = tf.ones(shape=[7])
    psnr(x1,x2,1.0,sample_weight=sample_weight)
    print(psnr.psnr)

    psnr.reset_state()
    print(psnr.psnr)
    print(psnr.psnr.trainable)

