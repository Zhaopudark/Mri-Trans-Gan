import numpy as np

from .compute_op import np_div_no_nan,np_zero_close,np_nan_to_zero
from .reduce_op import np_individual_min,np_individual_max

def norm_min_max(x:np.ndarray,
                global_min_max:tuple[np.ndarray,np.ndarray]|None=None,
                mask:np.ndarray|None=None,
                foreground_offset:float=0.0,
                dtype:type=np.float32,**kwargs):
    """
    将输入以min_max归一化到0-1domain
    x:input
    global_min_max: 记录全局的 min max
        若非 None 做基于全局的 min max norm 
        若为 None 仅做独立的min max norm
    mask:区分前景背景的 0 1mask
    foreground_offset: 前景偏移量
        x = Vaild_Value*(1-foreground_offset)+foreground_offset
    dtype: 全局的计算精度, x 和 mask 会被强制转换到该精度 为了数值稳定性 不允许为 [np.float32,np.float64] 之外的精度 
    """
    assert dtype in [np.float32,np.float64]
    x = x.astype(dtype)
    if global_min_max is not None:
        global_min_max = global_min_max[0].astype(dtype),global_min_max[1].astype(dtype)
        _min = np_individual_min(global_min_max[0],mask),
        _max = np_individual_max(global_min_max[1],mask)
    else:
        _min = np_individual_min(x.astype(dtype),mask)
        _max = np_individual_max(x.astype(dtype),mask)
    x = np_div_no_nan(x-_min,_max-_min)
    # the above steps support mask is None or not None
    # the following steps is special procedures when mask is not None
    if mask is not None:
        assert 0.<=foreground_offset<=1.
        assert np.equal(mask,np.where(mask>0.5,1,0)).all()
        x = x*(1.0-foreground_offset)+foreground_offset # 无mask时 foreground_offset是没有意义的
        x = x*(mask.astype(dtype))
        x = np_zero_close(x)
        x = np_nan_to_zero(x)  #if x*mask has inf*0 = Nan, make Nan to 0
    return x

def norm_z_score(x:np.ndarray,mask:np.ndarray|None=None,foreground_offset:float=0.0,dtype:type=np.float32,**kwargs):
    """
    将输入归一化到 均值为0 方差为1
    img:input
    mask:区分前景背景的 0 1mask
        当mask存在时,计算的是有效区域的均值标准差,将img的有效区域归一化到0均值 将背景归0
        img = (img-mean)/std
    foreground_offset: 前景偏移量 在z_score中,是有效区域整体(等价于有效区域均值)的偏移量,因此不同于`norm_min_max`的计算可以用于探究在z_score将背景也归到0或者其他值是否合理
        img = Vaild_Value+foreground_offset
        foreground_offset 在mask为None时无效
    dtype: 全局的计算精度, x 和 mask 会被强制转换到该精度 为了数值稳定性 不允许为 [np.float32,np.float64] 之外的精度 
    """
    assert dtype in [np.float32,np.float64] # np.float16 will lead to overflow
    x = x.astype(dtype)
    if mask is not None:
        assert np.equal(mask,np.where(mask>0.5,1,0)).all()
        _where = np.where(mask>0.5,True,False)
        mean = np.mean(x,where=_where)
        std = np.std(x,ddof=0.0,where=_where) #将一个图像的所有体素视为总体 求的是这个总体的标准差
        x = np_div_no_nan(x-mean,std)
        x = x+foreground_offset
        x = x*(mask.astype(dtype))
        x = np_zero_close(x) 
        return np_nan_to_zero(x) #if x*mask has inf*0 = Nan, make Nan to 0
    else:
        mean = np.mean(x)
        std = np.std(x,ddof=0.0) #将一个图像的所有体素视为总体 求的是这个总体的标准差
        return np_div_no_nan(x-mean,std)