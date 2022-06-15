
import tensorflow as tf 
from functools import wraps
from typeguard import typechecked
from typing import Iterable

@typechecked
def get_conv_paddings(input_length:int,filter_size:int,stride:int,dilation_rate:int,padding:str):
    """
    Give out equivalent conv paddings from current padding to VALID padding.
    For example, there is a conv(X,padding='same'), find the equivalent conv paddings and
    make conv(X,padding='same')===conv(pad(X,equivalent_conv_paddings),padding='VALID')
   
    see the:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    If padding == 'SAME': output_shape = ceil(input_length/stride)
    If padding == 'VALID': output_shape = ceil((input_length-(filter_size-1)*dilation_rate)/stride)

    NOTE In conv op, FULL padding, CUSUAL padding and VALID padding have explicit padding prodedure. 
    We can easliy give out the equivalent conv paddings without know the input_length.
    However, it 'SAME' is very special and ambiguous, beacuse its padding prodedure is influenced by 
    input_length and stride to ensure "output_shape===ceil(input_length/stride)". So we should give out a equivalent
    padding prodedure to find the equivalent conv paddings from 'SAME' to 'VALID' finally.
    Consider the equation, where pad_length is should known first:
        ceil(input_length/stride) == ceil((input_length+pad_length-(filter_size-1)*dilation_rate)/stride)
        ceil function makes the pad_length not unique.
        but by testing, 
        we find tensorflow (C++ API) 'SAME' padding's feature:
        1. always choose the minimum pad_length.
        2. divide pad_length equally to pad_left and pad_right and make pad_left<=pad_right
    using function conv_output_length()'s 'another2' artificial_padding_behavior
    return  paddings(padding vectors) for conv's padding behaviour
    """
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation_rate - 1)
    padding = padding.lower()
    if padding in 'valid':
        pad_left =  0
        pad_right = 0
    elif padding == 'causal': 
        pad_left =  dilated_filter_size - 1
        pad_right = 0
    elif padding == 'same': 
        flag = input_length%stride
        if flag==0:
            pad_all= max(dilated_filter_size-stride,0)
        else:
            pad_all= max(dilated_filter_size-flag,0)
        pad_left =  pad_all // 2
        pad_right = pad_all - pad_left
    elif padding == 'full': # full padding has been deprecated in many conv or deconv layers
        pad_left =  dilated_filter_size - 1
        pad_right = dilated_filter_size - 1
    else:
        raise ValueError("Padding should in 'valid', 'causal', 'same' or 'full', not {}.",format)
    return [pad_left,pad_right]

@typechecked
def conv_output_length(input_length:int,filter_size:int,padding:str,stride:int,dilation:int=1):
    """Determines output length of a convolution given input length.
    ANCHOR 问题2 已经表明
    在4点基本假设的前提下
    存在至少M种artificial_padding_behavior 
    使得output_length(artificial_padding_conv)===output_length(padding_conv)
    因此
    可以规定一个 artificial_padding_behavior 继而得到一个 artificial_padding_conv 计算output_length
    因此 本函数mimic了keras中不对外开放的output_length计算方法 并给予了解释 
    Args:
        input_length: integer.
        filter_size: integer.
        padding: one of 'same', 'valid', 'full', 'causal'
        stride: integer.
        dilation: dilation rate, integer.
    Returns:
        The output length (integer).
    basic
        由 问题2 和 问题1 可知
        output_length = (ALL_L + stride - 1) // stride
        ALL_L == input_length+left_pad+right_pad-filter_size+1
        规定一个artificial_padding_behavior 
        为了尽可能统一
            当padding == valid或full或causal时 
                即使 存在其他合理的artificial_padding_behavior 也可使得output_length(artificial_padding_conv)===output_length(padding_conv) 始终成立
                规定 artificial_padding_behavior 与 padding_behavior相同 即同问题1
                即由valid或full或causal自身的定义规定artificial_padding_behavior
            当padding == same时  
                不强制要求artificial_padding_behavior 与 padding_behavior相同
                而是尽可能让最终的计算形式便于统一
        padding == valid 时
            pad_left = 0 
            pad_right = 0
            那么 ALL_L == input_length+0+0-filter_size+1==input_length-filter_size+1
            output_length = (ALL_L + stride - 1) // stride 
            必定与output_length(valid_conv)一致 即结果正确
        padding == full 时
            pad_left = filter_size-1 
            pad_right = filter_size-1 
            那么 ALL_L == input_length+filter_size-1+filter_size-1-filter_size+1 == input_length+filter_size-1
            output_length = (ALL_L + stride - 1) // stride
            必定与output_length(full_conv)一致 即结果正确
        padding == causal时
            pad_left = filter_size-1 
            pad_right = 0
            那么 ALL_L == input_length+filter_size-1-filter_size+1 == input_length
            output_length = (ALL_L + stride - 1) // stride
            必定与output_length(causal_conv)一致 即结果正确
        padding == same时
            pad_left = filter_size//2
            pad_right = filter_size-filter_size//2-1
            那么 ALL_L == input_length+filter_size//2+filter_size-filter_size//2-1-filter_size+1 == input_length
            output_length = (ALL_L + stride - 1) // stride = (input_length+stride-1)//stride
            考察 (input_length+stride-1)//stride === ceil(input_length/stride) 是否成立
            即考察整数等式 (X+S-1)//S===ceil(X/S)是否成立 
            记 X = K*S+T K>=0 0<=T<=S-1
            当 T==0时
                左边= (K*S+S-1)//S = K
                右边= ceil(K*S/S) = K
                左边==右边
            当 1<=T<=S-1 必然有S>=2
                左边= (K*S+T+S-1)//S = K+1+(T-1)//S=K+1
                右边= ceil((K*S+T)/S) = ceil(K+T/S)=K+ceil(T/S)=K+1
                左边==右边
            因此 整数等式 (X+S-1)//S===ceil(X/S)成立
            因此 (input_length+stride-1)//stride === ceil(input_length/stride)成立
            从而与output_length(same_conv)一致 即结果正确
    another1:
        若取一个新的 artificial_padding_behavior 满足 output_length(artificial_padding_conv)===output_length(padding_conv)
        相比于 basic 中的 artificial_padding_behavior 其余不变 
        只修改 padding == same 时策略为
            pad_left = filter_size//2
            pad_right = filter_size//2
            那么 ALL_L == input_length+filter_size//2+filter_size//2-filter_size+1
            由于 filter_size >= 1 
            所以 ALL_L == input_length+1 (filter_size为偶数)
                或 ALL_L == input_length (filter_size为奇数)
            当 filter_size为奇数时  basic中已经证明 output_length=(ALL_L + stride - 1)//stride===ceil(input_length/stride)成立
            当 filter_size为偶数时 
                output_length=(input_length+1 + stride - 1)//stride=(input_length+stride)//stride
                考察 (input_length+stride)//stride === ceil(input_length/stride) 是否成立
                即考察整数等式 (X+S)//S===ceil(X/S)是否成立 
                记 X = K*S+T K>=0 0<=T<=S-1
                当 T==0时
                    左边= (K*S+S)//S = K+1
                    右边= ceil(K*S/S) = K
                    左边!=右边
                当 1<=T<=S-1 必然有S>=2
                    左边= (K*S+T+S)//S = K+1+T//S=K+1
                    右边= ceil((K*S+T)/S) = ceil(K+T/S)=K+ceil(T/S)=K+1
                    左边==右边
                因此 整数等式 (X+S-1)//S!==ceil(X/S) 
                因此 (input_length+stride-1)//stride !==ceil(input_length/stride) 
            从而与output_length(same_conv)不一致
            该artificial_padding_behavior 是不正确的
    another2:
        若取一个新的 artificial_padding_behavior 满足 output_length(artificial_padding_conv)===output_length(padding_conv)
        相比于 basic 中的 artificial_padding_behavior 其余不变 
        只修改 padding == same 时策略为
            pad_left = pad_all//2
            pad_right = pad_all - pad_left
            pad_all 为满足 output_length(artificial_padding_conv)===output_length(padding_conv) 非负最小值
            自然也与 output_length(same_conv)一致 即结果正确 这可以尽可能减少pad计算 平等的看待数据的左右
            
            以下过程是具体的pad_all计算推导 并证明满足 output_length(artificial_padding_conv)===output_length(padding_conv) 非负最小值pad_all是可以取道的
                有
                output_length=(input_length+left_pad+right_pad-filter_size+1+stride-1)//stride=(input_length+left_pad+right_pad-filter_size+stride)//stride
                (input_length+pad_all-filter_size+stride)//stride=== ceil(input_length/stride)
                考察 Y//S==ceil(Z/S)
                    当 Z被S整除时 Z-0<=Y<=Z+S-1
                    当 Z不被S整除时 记 S*ceil(Z/S) - Z = M
                        Z+M-0<=Y<=Z+M+S-1
                设 input_length = K*stride+T K>=0 0<=T<=stride-1             
                T == 0 时 
                    input_length<=input_length+pad_all-filter_size+stride<=input_length+stride-1
                    pad_all>=filter_size-stride
                    pad_all<=filter_size-1
                    当 filter_size-stride>=0时
                        取 pad_all== filter_size-stride 为非负最小值
                    当 filter_size-stride<0 时
                        filter_size-1>=0
                        所以0包含于 pad_all范围中
                        取 pad_all== 0
                    所以 
                    pad_all = max(filter_size-stride,0)
                1<=T<=stride-1 必然有stride>=2
                    input_length+(stride-T)<=input_length+pad_all-filter_size+stride<=input_length+(stride-T)+stride-1
                    pad_all>=filter_size-T 
                    pad_all<=filter_size+stride-T-1
                    当 filter_size-T >=0时
                        取 pad_all == filter_size-T 为非负最小值
                    当 filter_size-T <0 时
                        又因为 stride-1>= T 
                        所以 stride-T-1>= 0
                        filter_size+stride-T-1 >= filter_size>=1>0
                        所以0包含于 pad_all范围中
                        取 pad_all== 0
                    所以 
                    pad_all = max(filter_size-T,0)
            存在 pad_all 为满足 output_length(artificial_padding_conv)===output_length(padding_conv) 的非负最小值
            继而存在 artificial_same_behavior
                pad_left = pad_all//2
                pad_right = pad_all - pad_left
                pad_all 为满足 output_length(artificial_padding_conv)===output_length(padding_conv) 的非负最小值
            因此计算过程如下
            0---compute T = input_length%strides
            1---if T==0 then compute pad_all= max(filter_size-stride,0)
                if T>0  then compute pad_all= max(filter_size-T,0)
            2---compute pad_letf = pad_all//2
                compute pad_right = pad_all-pad_left
    NOTE 问题2指出 
    在4点基本假设的前提下
    存在至少M种artificial_padding_behavior 
    使得output_length(artificial_padding_conv)===output_length(padding_conv)
    本函数采用 basic artificial_padding_behavior 并非一定与实践时框架底层的 padding_behavior完全一致 
        给出的 another1 artificial_padding_behavior 是不正确的
        给出的 another2 artificial_padding_behavior 是更贴近 padding_behavior的行为 但底层的设计变动是无法被用户控制的 无法断言artificial_padding_behavior就是padding_behavior
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding in ['same', 'causal']:
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride

@typechecked  
def conv_input_length(output_length:int,filter_size:int,padding:str,stride:int):
    """Determines input length of a convolution given output length.
    由问题3可知
    在4点基本假设的前提下
    存在至少M种 artificial_padding_behavior 使得 input_length_range(artificial_padding_conv)===input_length_range(padding_conv) 
        计算input_length_range 就是将artificial_padding_behavior下的 
        output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
        计算逆向进行
    由问题4可知
    在某个 artificial_padding_behavior 下
        如果规定 p 为 "使返回的input_length为input_length_range的最小(最大)"
        必然有 input_length(artificial_padding_conv,p)===input_length(padding_conv,p) 成立 
        要计算 input_length 即通过input_length(artificial_padding_conv,p)计算input_length
        就是将 artificial_padding_behavior下的 
        output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
        计算逆向进行
        找到符合 'p' 要求的input_length
    NOTE tf.keras源码中的 conv_input_length采用了conv_output_length()中规定的 another1 artificial_padding_behavior 是不正确的
    这里进行correction 改为 basic artificial_padding_behavior
    即
    padding == valid 时
        pad_left = 0 
        pad_right = 0
    padding == full 时
        pad_left = filter_size-1 
        pad_right = filter_size-1 
    padding == causal时
        pad_left = filter_size-1 
        pad_right = 0
    padding == same时
        pad_left = filter_size//2
        pad_right = filter_size-filter_size//2-1
    设 p 为 "使返回的input_length为input_length_range的最小
    依据 
    output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
    对计算过程进行逆向 求出 最小的input_length
    因此可以直接改写为 
    output_length = (input_length+left_pad+right_pad-filter_size+stride)/stride # input_length是变量 其余为已知量 因此input_length最小是在整除与除等价时达到
    input_length = output_length*stride-left_pad-right_pad-stride+filter_size
    即 
    input_length = (output_length-1)*stride-left_pad-right_pad+filter_size
    Args:
        output_length: integer.
        filter_size: integer. means dilated_filter_size
        padding: one of 'same', 'valid', 'full'.
        stride: integer.
    Returns:
        The input length (integer).
    """
    padding = padding.lower()
    if padding == 'same':
        pad_left = filter_size // 2
        pad_right = filter_size-filter_size//2-1
    elif padding == 'valid':
        pad_left = pad_right = 0
    elif padding == 'full':
        pad_left = pad_right = filter_size - 1
    elif padding == 'causal':
        pad_left = filter_size - 1
        pad_right = 0
    else:
        raise ValueError(f"padding must be one of `same`, `valid`, `full`, `causal`, not `{padding}`.")
    return (output_length - 1) * stride - pad_left - pad_right + filter_size

@typechecked
def deconv_output_length(input_length:int,
                         filter_size:int,
                         padding:str,
                         output_padding:int|None,
                         stride:int,
                         dilation:int=1):
    """Determines output length of a transposed convolution given input length.
    
    由问题4可知
    在4点基本假设的前提下
    存在至少M种 artificial_padding_behavior 使得 input_length_range(artificial_padding_conv)===input_length_range(padding_conv) 
    在某个 artificial_padding_behavior 下
        如果规定 p 为 "使返回的input_length为input_length_range的最小(最大)"
            必然有 input_length(artificial_padding_conv,p)===input_length(padding_conv,p) 成立 
        如果规定 p 为 "padding==valid或者same或者causal时返回最接近output_length*stride的input_length padding==full时返回最小input_length"
            必然有 input_length(artificial_padding_conv,p)===input_length(padding_conv,p) 成立 
        要计算input_length 即通过input_length(artificial_padding_conv,p)计算input_length
        就是将artificial_padding_behavior下的 
        output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
        计算逆向进行
        找到符合 'p' 要求的input_length

    问题5指出 存在若干合理的 deconv(·)计算output_length的过程
    本函数mimic了和tf.keras源码一致的output_length计算过程
        用户有output_padding需求时
            先规定artificial_padding_behavior为 conv_output_length() 中的 basic artificial_padding_behavior
            规定 p 为 "使返回的input_length为input_length_range的最小"
            input_length(artificial_padding_conv,p)可以给出最小的input_length 同conv_input_length()函数 不再赘述
            即 
            _conv_input_length = conv_input_length(...)
            _conv_input_length += output_padding
        用户没有output_padding需求时 
            先规定artificial_padding_behavior为 conv_output_length() 中的 basic artificial_padding_behavior
            规定 p 为 "padding==valid或者same或者causal时返回最接近output_length*stride的input_length padding==full时返回最小input_length"   
            padding == 'full'时 
                pad_left = pad_right = filter_size - 1
                output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
                              = (input_length+filter_size+stride-2)//stride
                output_length*stride<=input_length+filter_size+stride-2<=output_length*stride+(stride-1)
                output_length*stride-filter_size-stride+2<=input_length<=output_length*stride-filter_size+1
                取最小 input_length = output_length*stride-filter_size-stride+2
                即
                _conv_input_length = _conv_output_length*stride-filter_size-stride+2
            padding == 'valid'时
                pad_left = pad_right = 0
                output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
                                    = (input_length-filter_size+stride)//stride
                output_length*stride<=input_length-filter_size+stride<=output_length*stride+(stride-1)
                output_length*stride+(filter_size-stride)<=input_length<=output_length*stride+(stride-1)+(filter_size-stride)
                当 filter_size>=stride时 最接近output_length*stride的值为
                    即 input_length = output_length*stride+(filter_size-stride)
                当 filter_size<stride时 
                    由于此时
                    output_length*stride+(filter_size-stride) < output_length*stride
                    output_length*stride+(stride-1)+(filter_size-stride) = output_length*stride+filter_size-1>=output_length*stride
                    output_length*stride 符合区间范围
                    input_length = output_length*stride
                所以
                input_length = output_length*stride+max(filter_size-stride,0) 是最终形式
                即
                _conv_input_length = _conv_output_length*stride+max(filter_size-stride,0)
            padding == 'causal'时  NOTE tf.keras 源码中无该情况 这里添加该情况
                pad_left = filter_size - 1
                pad_right = 0 
                output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
                                    = (input_length+stride-1)//stride
                output_length*stride<=input_length+stride-1<=output_length*stride+(stride-1) 
                output_length*stride-stride+1<=input_length<=output_length*stride
                取最接近output_length*stride的值为
                input_length = output_length*stride
                即
                _conv_input_length = _conv_output_length*stride
            padding == 'same'时
                pad_left = filter_size//2
                pad_right = filter_size-filter_size//2-1 
                output_length = (input_length+left_pad+right_pad-filter_size+1+stride-1)//stride
                                    = (input_length+stride-1)//stride
                output_length*stride<=input_length+stride-1<=output_length*stride+(stride-1) 
                output_length*stride-stride+1<=input_length<=output_length*stride
                取最接近output_length*stride的值为
                input_length = output_length*stride
                即
                _conv_input_length = _conv_output_length*stride  
    Args:
        input_length: Integer.
        filter_size: Integer.
        padding: one of `same`, `valid`, `full`, `causal`.
        output_padding: Integer, amount of padding along the output dimension. Can
            be set to `None` in which case the output length is inferred.
        stride: Integer.
        dilation: Integer.
    Returns:
        The output length (integer).
    """
    _conv_output_length = input_length
    # Get the dilated kernel size
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    # Infer length if output padding is None, else compute the exact length
    if output_padding is None:
        if padding == 'valid':
            _conv_input_length = _conv_output_length * stride + max(dilated_filter_size - stride, 0)
        elif padding == 'full':
            _conv_input_length = _conv_output_length * stride - (stride + dilated_filter_size - 2)
        elif padding == 'causal':
            _conv_input_length = _conv_output_length * stride
        elif padding == 'same':
            _conv_input_length = _conv_output_length * stride
        else:
            raise ValueError(f"padding must be one of `same`, `valid`, `full`, `causal`, not `{padding}`.")
    else:
        _conv_input_length = conv_input_length(output_length=_conv_output_length,filter_size=dilated_filter_size,padding=padding,stride=stride)
        _conv_input_length += output_padding
    return _conv_input_length

@typechecked
def get_padded_length_from_paddings(length:int|None,paddings:tuple[int,int]):
    if length is not None:
        pad_left,pad_right = paddings
        length = length + pad_left + pad_right
    return length
@typechecked
def normalize_paddings_by_data_format(data_format:str,paddings:Iterable):
    out_buf = []
    for data_format_per_dim in data_format:
        if data_format_per_dim.upper() in ['N','C']:
            out_buf.append(tuple([0,0]))
        elif data_format_per_dim.upper() in ['D','H','W']:
            out_buf.append(tuple(next(paddings)))
        else:
            raise ValueError(f"data_format should consist with `N`, `C`, `D`, `H` or `W` but not `{out_buf}`.")
    return out_buf
@typechecked
def grab_length_by_data_format(data_format:str,length:tuple|list):
    out_buf = []
    for data_format_per_dim,length_per_dim in zip(data_format,length):
        if data_format_per_dim.upper() in ['N','C']:
            pass
        elif data_format_per_dim.upper() in ['D','H','W']:
            out_buf.append(int(length_per_dim))
        else:
            raise ValueError(f"data_format should consist with `N`, `C`, `D`, `H` or `W` but not `{out_buf}`.")
    return out_buf
def normalize_padding(value):
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if padding not in {'valid', 'same', 'causal','full'}:
        raise ValueError(f"The `padding` argument must be a list/tuple or one of `valid`, `same`, `full` (or `causal`, only for `Conv1D`). Received: {padding}")
    return padding
def normalize_specific_padding_mode(value):
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if padding not in {'constant', 'reflect', 'symmetric'}:
        raise ValueError(f"The `padding` argument must be a list/tuple or one of `constant`, `reflect` or `symmetric`. Received: {padding}")
    return padding.upper()
def normalize_tuple(value, n, name, allow_zero=False):
    """Transforms non-negative/positive integer/integers into an integer tuple.
    Args:
        value: The value to validate and convert. Could an int, or any iterable of
        ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. 'strides' or
        'kernel_size'. This is only used to format error messages.
        allow_zero: Default to False. A ValueError will raised if zero is received
        and this param is False.
    Returns:
        A tuple of n integers.
    Raises:
        ValueError: If something else than an int/long or iterable thereof or a
        negative value is
        passed.
    """
    error_msg = (f"The `{name}` argument must be a tuple of {n} "
                f"integers. Received: {value}")

    if isinstance(value, int):
        value_tuple = (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(error_msg)
        if len(value_tuple) != n:
            raise ValueError(error_msg)
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                error_msg += (f"including element {single_value} of "
                            f"type {type(single_value)}")
                raise ValueError(error_msg)
    if allow_zero:
        unqualified_values = {v for v in value_tuple if v < 0}
        req_msg = '>= 0'
    else:
        unqualified_values = {v for v in value_tuple if v <= 0}
        req_msg = '> 0'
    if unqualified_values:
        error_msg += (f" including {unqualified_values}"
                    f" that does not satisfy the requirement `{req_msg}`.")
        raise ValueError(error_msg)

    return value_tuple
