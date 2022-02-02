"""
helper build conv layers
"""
import tensorflow as tf 
from functools import wraps
def typeguard_for_conv_helper(func):
    def _arg_trans(input_arg):
        """
        specify for conv helper
        trans input_shape,filter,kernel_size to int 
        trans padding to str in lowercase
        """
        if isinstance(input_arg,list) or isinstance(input_arg,tuple) or isinstance(input_arg,tf.TensorShape):
            if len(input_arg)>1 or len(input_arg)<=0:
                raise ValueError("input arg must be 1-dim, not {}".format(input_arg))
            else:
                if hasattr(input_arg,"as_list"):
                    out_put = input_arg.as_list()[0]
                else:
                    out_put =  input_arg[0]
            if out_put is None: # To avoid None in [None,shape1,shape2]
                raise ValueError("input_arg[0] must be a element but not a None")
            else:
                return int(out_put)
        elif isinstance(input_arg,int) or isinstance(input_arg,float) or isinstance(input_arg,tf.Tensor):
            return  int(input_arg)
        elif isinstance(input_arg,str):
            return  input_arg.lower()
        else:
            raise ValueError("input_arg must be a list, tuple, TensorShape, int, float or Tensor")
    @wraps(func)
    def wrappered(*args,**kwargs):
        args = list(map(_arg_trans,args))
        for key,value in kwargs.items():
            kwargs[key] = _arg_trans(value)
        _output = func(*args,**kwargs)
        return _output
    return wrappered

@typeguard_for_conv_helper
def get_patch_conv_padding_needs(input_shape,kernel_size,stride,padding,dilation_rate):
    """
    see the:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    If padding == "SAME": output_shape = ceil(input_shape/stride)
    If padding == "VALID": output_shape = ceil((input_shape-(kernel_size-1)*dilation_rate)/stride)
    """
   

    

def GetConvShape(input_shape,filters,kernel_size,strides,padding,*args,**kwargs):
    """
    卷积过程的参数计算
    3D--BDHWC
    2D--BHWC
    input_shape [H,W]or[D,H,W] 不带Batch和Deepth
    filters 单值int
    kernel_size [Hy,Wx]or[Dz,Hy,Wx]
    strides [Hs2,Ws1]or[Ds3,Hs2,Ws1]
    padding "SAME","REFLECT","CONSTANT","SYMMETRIC",
    深度(卷积核个数)、单个卷积核大小、步长和pad方式后 计算相关的参数
    返回两个重要的内容(卷积输出)
    If padding == "SAME": output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    If padding == "VALID": output_spatial_shape[i] = ceil((input_spatial_shape[i] - (spatial_filter_shape[i]-1) * dilation_rate[i]) / strides[i]). 向上取整 因为就算不够整除 第一个位置也是有的
    对于VALID的公式的解释 和中文教材另一个公式有出入 但是是一样的 即将第一个位置摆上卷积核后 如卷积核宽4 输入宽64 那么64-3后 有61个位置 可以给卷积核的最右端那一列进行移动 除以步长 计算剩下的位置即可 卷积核非最后一列的前列位置都是不考虑的 就是一个摆格子的算法
    而(input_spatial_shape[i]-spatial_filter_shape[i])//strides[i] +1 则是将第一个位置摆上卷积核后 计算剩余位置可以摆放的卷积核的个数 然后加上已经考虑的第一个 他们本质一致
    
    用pad+valid模拟same 
    """
    out_shape = []
    out_shape.append(input_shape[0])
    input_shape = input_shape[1:-1]

    if (len(input_shape)!=len(kernel_size))or(len(input_shape)!=len(strides)):
        raise ValueError("Convolution calculations needs the same dims on input_shape,kernel_size and strides")
    l = len(input_shape)
    padding_vect = []
    padding_list = ["SAME","REFLECT","CONSTANT","SYMMETRIC"]
    if padding == "VALID":
        for i in range(l):
            padding_vect.append([0,0])
        padding = "CONSTANT"
    elif padding in padding_list:
        if padding=="SAME":
            padding = "CONSTANT"#默认为一般的0 padding
        for i in range(l):
            # padding目的就是保持一致 所以直接用结果推结论 不考虑中间的奇偶关系
            pad_begin = kernel_size[i]//2#开头的padding是一定存在的 右边的padding
            pad_end = kernel_size[i]//2
            for j in range(0,pad_begin+pad_end+1,1):#kernel_size很大时 padding的大小有很多 找寻最小的padding方式
                if tf.math.ceil((input_shape[i]+j-kernel_size[i]+1)/strides[i]) == tf.math.ceil(input_shape[i]/strides[i]):
                    break
            """
            tf内部的实现逻辑是  计算最小的padding数
            (右端)后端先padding 然后 (左端)前端padding
            实现时 begin = j//2 end=j-begin 即可
            """
            pad_begin = j//2
            pad_end = j - pad_begin
            padding_vect.append([pad_begin,pad_end])
    else:
        raise ValueError("Unsupported padding ways")
    buf = []
    for i in range(l):
        buf.append((input_shape[i]+padding_vect[i][0]+padding_vect[i][1]-kernel_size[i])//strides[i]+1)
    out_shape = out_shape+buf+[filters]
    return out_shape,padding,[[0,0]]+padding_vect+[[0,0]]
    
# @classmethod
# def ConvTransCal(cls,input_shape,filters,kernel_size,strides,padding,*args,**kwargs):
#     """
#     在没有给出输出shape的时候 手动计算输出shape
#     这里的output_shape 已经将Batch算在其中了
#     """
#     out_shape = []
#     out_shape.append(input_shape[0])
#     input_shape = input_shape[1:-1]
#     if (len(input_shape)!=len(kernel_size))or(len(input_shape)!=len(strides)):
#         raise ValueError("DeConvolution calculations needs the same dims on input_shape,kernel_size and strides")
#     l = len(input_shape)
#     if padding == "VALID":
#         for i in range(l):
#             out_shape.append(strides[i]*(input_shape[i]-1)+kernel_size[i])
#             # 根据不等式计算出最小的output 取值就是上述计算公式
#             # if tf.math.ceil((out_shape[i]-kernel_size[i]+1)/strides[i]) != input_shape[i]:
#                 # raise ValueError("Mismatched input shapes and output shapes.")
#         out_shape.append(filters)
#         return out_shape
#     elif padding == "SAME":
#         for i in range(l):
#             out_shape.append(strides[i]*input_shape[i])
#         out_shape.append(filters)
#         return out_shape
#     else:
#         raise ValueError("Unsupported padding ways")
# @classmethod
# def ConvTransCheck(cls,input_shape,output_shape,filters,kernel_size,strides,padding,*args,**kwargs):
#     """
#     反卷积不再是使用pad+valid 模拟same padding 的方式 而是辅助计算卷积给定参数是否合理 是否满足卷积公式
#     卷积公式是依旧成立的
#     input_shape 是转置卷积的输入  等于正常卷积过程的输出
#     """
#     if (len(input_shape)!=len(output_shape))or(len(input_shape)!=len(kernel_size))or(len(input_shape)!=len(strides)):
#         raise ValueError("DeConvolution calculations needs the same dims on input_shape,output_shape,kernel_size and strides")
#     conv_output_shape = input_shape
#     conv_input_shape = output_shape
#     l = len(conv_input_shape)
#     if padding == "VALID":
#         for i in range(l):
#             if tf.math.ceil((conv_input_shape[i]-kernel_size[i]+1)/strides[i]) != conv_output_shape[i]:
#                 raise ValueError("Mismatched input shapes and output shapes.")
#     elif padding == "SAME":
#         for i in range(l):
#             if tf.math.ceil(conv_input_shape[i]/strides[i]) != conv_output_shape[i]:
#                 raise ValueError("Mismatched input shapes and output shapes.")
#     else:
#         raise ValueError("Unsupported padding ways")
    
# @classmethod
# def Trans2UpsampleCal(cls,input_shape,output_shape,filters,kernel_size,strides,padding,*args,**kwargs):
#     """
#     反卷积过程转化为卷积的相关计算。
#     满足转置卷积的参数条件后，开始卷积。

#     希望up_op可以达到真正的大小 然后进行SAME卷积 需要深入分析

#     x-1<ceil(x)<x+1
#     当x为整数时  x=ceil(x)<x+1
#     当x为非整数时 x<ceil(x)<x+1
#     所以x<=ceil(x)<x+1
#     原本指定SAME时 input_shape = ceil(output_shape/strides)
#     output_shape/strides<=input_shape<output_shape/strides +1
#     output_shape <=input_shape*strides<output_shape+strides
#     input_shape*strides是上采样后得到的直接shape
#     和output_shape目标之间存在差异 寻找办法使得卷积后得到output_shape 
#     如果就对上采样后得到的直接shape进行一步长的卷积公式即如下
#     (input_shape*strides-kernelsize)/1 + 1 = input_shape*strides-kernelsize+1 <output_shape+strides-kernelsize+1

#     原本指定VALID时 input_shape = ceil(output_shape[i]-kernel_size[i]+1)/strides[i])
#     (output_shape-kernel_size+1)/strides<=input_shape< (output_shape-kernel_size+1)/strides +1
#     output_shape-kernel_size+1<=input_shape*strides<output_shape-kernel_size+1+strides
    
#     一般的 如 kernel_size>=3 strides<=2 的情况下 delt=strides-kernelsize+1<=0
#         本着对输入信息只增不减的原则 
#         原本指定SAME时 input_shape*strides-kernelsize+1<output_shape+delt<=output_shape
#         那么就存在正的padding方式 对input_shape*strides补齐 然后满足pad=valid,kernelsize不变,strides=1的卷积 实现指定的输出维度

#         原本指定VALID时 input_shape*strides<output_shape+delt<=output_shape
#         那么就存在正的padding方式 对input_shape*strides补齐到output_shape维度 然后进行pad=SAME,kernelsize不变,strides=1的卷积 实现指定的输出维度

#     delt>0时 
#         input_shape*strides如果小于output_shape 则做pad补齐 
#         input_shape*strides如果大于output_shape 则做cut裁剪 
#     """
#     cls.ConvTransCheck(input_shape,output_shape,filters,kernel_size,strides,padding)
#     l = len(input_shape)#dim
#     padding_vect=[]
#     padding_list = []
#     for i in range(l):
#         delt = strides[i]-kernel_size[i]+1
#         if delt <= 0:
#             if padding == "VALID":
#                 differ = output_shape[i]-input_shape[i]*strides[i]
#                 pad_begin = differ//2
#                 pad_end = differ - pad_begin
#                 padding_vect.append([pad_begin,pad_end])
#                 padding_list.append("SAME")
#             elif padding == "SAME":
#                 differ = output_shape[i]-(input_shape[i]*strides[i]-kernel_size[i]+1)
#                 pad_begin = differ//2
#                 pad_end = differ - pad_begin
#                 padding_vect.append([pad_begin,pad_end])
#                 padding_list.append("VALID")
#             else:
#                 pass
#         else:
#             if input_shape[i]*strides[i] <= output_shape[i]:
#                 differ = output_shape[i]-input_shape[i]*strides[i]
#                 pad_begin = differ//2
#                 pad_end = differ - pad_begin
#                 padding_vect.append([pad_begin,pad_end])
#                 padding_list.append("SAME")
#             else:
#                 differ = -(output_shape[i]-input_shape[i]*strides[i])
#                 pad_begin = differ//2
#                 pad_end = differ - pad_begin
#                 padding_vect.append([-pad_begin,-pad_end])
#                 padding_list.append("SAME")
                

#     tmp = padding_list[0]
#     for item in padding_list:
#         if item != tmp:
#             raise ValueError("Padding ways not euqual "+str(item)+" and "+str(tmp))
#     positive_pad = 0
#     negative_pad = 0
#     for item in padding_vect:
#         if (item[0]<0)or(item[1]<0):
#             negative_pad += 1
#         elif (item[0]>0)or(item[1]>0):
#             positive_pad += 1
#         else:
#             pass 
#     if (positive_pad>0)and(negative_pad>0):
#         raise ValueError("Can not handle both positive_pad and negative_pad")
#     if negative_pad>0:
#         cut_flag = True
#         padding = padding_list[0]
#         return  padding,[[0,0]]+padding_vect+[[0,0]],cut_flag
#     else:
#         cut_flag = False
#         padding = padding_list[0]
#         return  padding,[[0,0]]+padding_vect+[[0,0]],cut_flag