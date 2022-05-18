import tensorflow as tf 
import tempfile
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True) 
import functools
def tf_py_function_wrapper(func=None):
    # since tf.py_function can not deal with dict directly, and its using form is not easy
    # here we make this wrapper, it can trans `func`'s output 
    # all tensors that a user want to use by calling their `numpy()` functions should become the inputs of the wrapped `func`, otherwise, `numpy()` will not work
    # since tf.nest.flatten  tf.nest.pack_sequence_as will sort dict structure's `keys` automaticlly, we do not use tf.nest here to avoid unexpected behavior
    if func is None:
        return functools.partial(tf_py_function_wrapper,)
    @functools.wraps(func)
    def wrappered(inputs:dict[str,tf.Tensor],output_structure:dict[str,tf.TensorSpec])->dict[str,tf.Tensor]:
        inp = tuple(inputs.values())
        Tout = tuple(output_structure.values())
        flattened_output = tf.py_function(func,inp=inp,Tout=Tout)
        return dict(zip(output_structure.keys(),flattened_output))
    return wrappered  

inputs = {'k1':tf.convert_to_tensor(str(1)),
          'k2':tf.convert_to_tensor(str(2)),
          'k3':tf.convert_to_tensor(str(3)),
          'k4':tf.convert_to_tensor(str(4))}
#@tf.function 
def original_func(inputs:dict[str,tf.Tensor]): # 各元素*2
    # @tf.function会导致函数运行在图模式, 此时tf.Tensor不存在numpy()方法, 因此当前函数尽可能运行在Eager模式
    for key in inputs:
        inputs[key] = tf.convert_to_tensor(int(str(inputs[key].numpy(),encoding='UTF-8')))*2
    inputs['k5'] = tf.random.normal(shape=[],dtype=tf.float32)
    inputs['k6'] = tf.random.normal(shape=[],dtype=tf.float16)
    return inputs
print(original_func(inputs))


inputs = {'k1':tf.convert_to_tensor(str(1)),
          'k2':tf.convert_to_tensor(str(2)),
          'k3':tf.convert_to_tensor(str(3)),
          'k4':tf.convert_to_tensor(str(4))}
@tf_py_function_wrapper
def modified_func(*inputs): # 各元素*2
    return [tf.convert_to_tensor(int(str(item.numpy(),encoding='UTF-8')))*2 for item in inputs]+[tf.random.normal(shape=[],dtype=tf.float32),tf.random.normal(shape=[],dtype=tf.float16)]
output_structure = {key:tf.TensorSpec(shape=None,dtype=tf.int32) for key in inputs}|\
                   {'k5':tf.TensorSpec(shape=None,dtype=tf.float32),'k6':tf.TensorSpec(shape=None,dtype=tf.float16)}
print(modified_func(inputs,output_structure))
  