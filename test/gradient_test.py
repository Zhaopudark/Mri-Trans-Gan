"""
测试了tensorflow中
keras.optimizer  minimiz方法内部的求导的grad_loss参数
以及tf.Graditent  gradient方法求导output_gradients参数
的作用与使用场景
结论是该参数一般用于手动链式求导中,用以承载上一级的梯度,在数值上与本级求导结果相乘,支持broadcasting 
但由于要处理嵌套结构 broadcasting 机制非常奇怪 升至可能是bug 因此最好不要让梯度输出为多值 而仅仅是单值 
"""
import tensorflow as tf 
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

optimizer =  tf.keras.optimizers.SGD(1.0)
w = [tf.Variable([0.0,0.1]),tf.Variable([0.0,0.2])]
x = 2.0
with tf.GradientTape(persistent=True) as tape:
    y = x*(w[0]+w[1])
tf.print(y) # [0 0.6]
grads = tape.gradient(y,w)
tf.print(grads) # [[2 2], [2 2]]
optimizer.minimize(y,w,grad_loss=[tf.constant([1.,2.]),tf.constant([2.,2.])],tape=tape)
tf.print(w) # [[-2 -3.9], [-2 -3.8]]

w = tf.Variable([1.0,0.0])
x = 2.0
with tf.GradientTape() as tape:
    y = x*w
optimizer.minimize(y,w,grad_loss=tf.constant([2.0,5.0]),tape=tape)
tf.print(w) # [-3 -10]

w = [tf.Variable([[0.0,0.1],[0.7,0.8]]),tf.Variable([0.0,0.2])]
x = 2.0
with tf.GradientTape(persistent=True) as tape:
    y = x*(w[0]+w[1])
    y2 = 7.0*y
grads = tape.gradient(y2,w)
tf.print(grads,type(grads),type(grads[0]),grads[0].shape) 
# [[[14 14]
#  [14 14]], [28 28]] <class 'list'> <class 'tensorflow.python.framework.ops.EagerTensor'> TensorShape([2, 2])
grads = tape.gradient(y,w,output_gradients=tape.gradient(y2,y)) # 手动链式求导
tf.print(grads)
# [[[14 14]
#  [14 14]], [28 28]]
output_gradients = [tf.constant([[-1.0,-2.0],[-3.0,.4]]),tf.constant([[-1.01111,-22222.0],[-3333.0,.4444]])]
# output_gradients = [[-1.0,-2.0],[-3.0,-4.0]]
# output_gradients = [-1.0,-2.0,-3.0,-4.0]
# output_gradients = [tf.constant([-1.0]),tf.constant([-3.0,-4.0])]
# output_gradients = -1.0
# output_gradients = -1.0,2.0
grads = tape.gradient(y2,w,output_gradients=output_gradients) # 人为干预
tf.print(grads) 
# [[[-14 -28]
#  [-42 5.6]], [-56 -22.4]] 明显是对前一个梯度的逐维度求sum