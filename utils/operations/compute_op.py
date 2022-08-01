import numpy as np
import tensorflow as tf  


def np_zero_close(x:np.ndarray):
    _where = np.isclose(x,0.0,rtol=1e-05,atol=1e-08,equal_nan=False)# 与0比较 其余取默认值(默认nan与nan不相等 返回false,nan与非nan不相等,返回false)
    x[_where]=0
    return x

def np_nan_to_zero(x:np.ndarray):
    _where = np.isnan(x)# 与0比较 其余取默认值(默认nan与nan不相等 返回false,nan与非nan不相等,返回false)
    x[_where]=0
    return x

def np_div_no_nan(a:np,b:np):
    """构建基于numpy的div_no_nan
    input:被除数a,除数b
    output:a/b, Nan值归0
    当a b 为同浮点型时  输出为同浮点型
    当a b 类型不同时 输出类型是一个不定的类型
    """
    out = a/b
    _where = np.isnan(out)
    out[_where] = 0
    return out

def redomain(x:np.ndarray,domain:tuple=(0.0,1.0),dtype:type=np.float32):
    """
    将任意值域的输入均匀得映射到目标值域
    domain a tuple consists of min_val and max_val
    """
    x = x.astype(dtype)
    _min = x.min()
    _max = x.max()
    x = np_div_no_nan(x-_min,_max-_min)
    domain_min = min(domain)
    domain_max = max(domain)
    x = x*(domain_max-domain_min)+domain_min
    return np_zero_close(x)

def combine2patches(ka:str,a:tf.Tensor,ar:tf.Tensor,am:tf.Tensor,kb:str,b:tf.Tensor,br:tf.Tensor,bm:tf.Tensor):
    tf.debugging.assert_equal(ka,kb)
    tf.debugging.assert_equal(a.shape,am.shape)
    tf.debugging.assert_equal(b.shape,bm.shape)
    tf.debugging.assert_equal(tf.abs(ar[:,0]-ar[:,1])+1,a.shape)
    tf.debugging.assert_equal(tf.abs(br[:,0]-br[:,1])+1,b.shape)
    ar_l = tf.reduce_min(ar,axis=-1)
    ar_r = tf.reduce_max(ar,axis=-1)
    br_l = tf.reduce_min(br,axis=-1)
    br_r = tf.reduce_max(br,axis=-1)
    out_range_l = tf.reduce_min(tf.stack([ar_l,br_l],axis=-1),axis=-1)
    out_range_r = tf.reduce_max(tf.stack([ar_r,br_r],axis=-1),axis=-1)
    ar_padding = tf.stack([ar_l-out_range_l,out_range_r-ar_r],axis=-1)
    br_padding = tf.stack([br_l-out_range_l,out_range_r-br_r],axis=-1)
    new_a = tf.pad(a,ar_padding, mode='CONSTANT', constant_values=0, name=None)
    new_b = tf.pad(b,br_padding, mode='CONSTANT', constant_values=0, name=None)
    new_am = tf.pad(am,ar_padding, mode='CONSTANT', constant_values=0, name=None)
    new_bm = tf.pad(bm,br_padding, mode='CONSTANT', constant_values=0, name=None)
    return ka,new_a+new_b,tf.stack([out_range_l,out_range_r],axis=-1),new_am+new_bm
def extend_to(x,patch_ranges,total_ranges):
    padding_vectors = [[min(inner_range)-min(outer_range),max(outer_range)-max(inner_range)] for outer_range,inner_range in zip(total_ranges,patch_ranges)]
    return tf.pad(x,padding_vectors,"CONSTANT")
    
def mse(y_true:tf.Tensor,y_pred:tf.Tensor):
    axis = list(range(1,tf.rank(y_true)))
    return tf.reduce_mean(tf.math.square(y_true-y_pred),axis=axis)


