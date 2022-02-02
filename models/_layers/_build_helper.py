import tensorflow as tf
__all__ = [
    "Reconstruction",
]
class Reconstruction():
    def __init__(self):
        "集中处理不规范的shape,给前面添加必要的None以规范化"
        "集中处理不规范的kernel_size"
        "做一个激活函数的集合,以调用类方法的形式返回一个类方法"
        "做一个变量初始化函数的集合"
        "做一个卷积的辅助计算函数，用于依据用户输入，给出真正的输出shape"
        "做一个转置卷积的辅助计算函数，计算过程中的输出shape"
    @classmethod
    def remake_shape(cls,shape,dims):
        if type(shape) == int:
            buf = [shape]
            for _ in range(dims-1):
                buf = [None]+buf
            return buf
        elif type(shape) == tuple:
            shape = list(shape)
            return cls.remake_shape(shape,dims)
        elif type(shape) == list:
            if len(shape) == dims:
                return shape
            elif len(shape) < dims:
                buf = shape
                for _ in range(dims-len(shape)):
                    buf = [None]+buf
                return buf
            else:
                raise ValueError("Unsupported shape "+str(shape))
        else:
            raise ValueError("Unsupported shape type "+str(shape))
    @classmethod
    def remake_kernel_size(cls,kernel_size,dims):
        if type(kernel_size) == tuple:
            kernel_size = list(kernel_size)
            return cls.remake_kernel_size(kernel_size,dims)
        elif type(kernel_size) == list:
            if len(kernel_size) == dims:
                return [1]+kernel_size+[1]
            elif len(kernel_size) == dims+2:
                return kernel_size
            else:
                raise  ValueError("Unsupported kernel size "+str(kernel_size))
        else:
            raise ValueError("Unsupported kernel size type"+str(kernel_size))
    @classmethod
    def remake_strides(cls,strides,dims):
        if type(strides) == tuple:
            strides = list(strides)
            return cls.remake_strides(strides,dims)
        elif type(strides) == list:
            if len(strides) == dims:
                return [1]+strides+[1]
            elif len(strides) == dims+2:
                return strides
            else:
                raise ValueError("Unsupported strides "+str(strides))
        else:
            raise ValueError("Unsupported strides type "+str(strides))  
    @classmethod
    def activation(cls,activation):
        if activation == "relu":
            return tf.nn.relu
        elif activation == "leaky_relu":
            return tf.nn.leaky_relu
        elif activation == "sigmoid":
            return tf.nn.sigmoid
        elif activation == "tanh":
            return tf.nn.tanh
        elif activation == None:
            return lambda x:x
        else:
            raise  ValueError("Unsupported strides activation: "+activation)
    @classmethod
    def initializer(cls,initializer,*args,**kwargs):
        if initializer == "glorot_normal":
            return tf.keras.initializers.GlorotNormal(*args,**kwargs)
        elif initializer == "glorot_uniform":
            return tf.keras.initializers.GlorotUniform(*args,**kwargs)
        elif initializer == "random_normal":
            return tf.keras.initializers.RandomNormal(*args,**kwargs)
        elif initializer == "random_uniform":
            return tf.keras.initializers.RandomUniform(*args,**kwargs)
        else:
            raise ValueError("Unsupported strides initializer: "+initializer)
    @classmethod
    
if __name__ == "__main__":

    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    filters_zp=1
    inshape=[16,16]
    k_size = [4,4]
    strides = [8,8]
    # for i in range(16,128+1,1):
    #     for j in range(2,9+1,1):
    #         for k in range(1,4+1):
    #             inshape=[i,i]
    #             k_size = [j,j]
    #             strides = [k,k]
    #             padding = "SAME"
    #             # x = tf.random.normal([1]+inshape+[1])
    #             x = tf.ones([1]+inshape+[1])
    #             w = tf.ones(k_size+[1]+[filters_zp])
    #             y=tf.nn.conv2d(x,w,strides,padding)
    #             out_shape,padding,padding_vect=Reconstruction.ConvCalculation(input_shape=inshape,
    #                                     filters=filters_zp,
    #                                     kernel_size=k_size,
    #                                     strides=strides,
    #                                     padding=padding)
    #             x_ = tf.pad(x,padding_vect,"CONSTANT")
    #             y_ = tf.nn.conv2d(x_,w,strides,"VALID")
    #             temp = tf.reduce_mean((y-y_)[0,:,:,0])
    #             if temp !=0.0 :
    #                 print("**********")
    #                 print(padding_vect)
    #                 print(temp.numpy())
    #                 print(inshape,k_size,strides)
                            


    padding = "SAME"
    # x = tf.ones([1]+inshape+[1])
    x = tf.random.normal([1]+inshape+[1])
    w = tf.ones(k_size+[1]+[filters_zp])
    print(w[:,:,0,0])
    y=tf.nn.conv2d(x,w,strides,padding)
    
    # print(Reconstruction.ConvCalculation(input_shape=inshape,
                                        # filters=filters_zp,
                                        # kernel_size=k_size,
                                        # strides=strides,
                                        # padding=padding))
    out_shape,padding,padding_vect=Reconstruction.ConvCalculation(input_shape=inshape,
                                        filters=filters_zp,
                                        kernel_size=k_size,
                                        strides=strides,
                                        padding=padding)
    print(padding_vect)
    x_ = tf.pad(x,padding_vect,"CONSTANT")
    print(x[0,:,:,0])
    print(x_[0,:,:,0])
    print(x.shape)
    print(x_.shape)
    y_ = tf.nn.conv2d(x_,w,strides,"VALID")
    print(y[0,:,:,0])
    print(y_[0,:,:,0])
    print(y.shape)
    print(y_.shape)
    print((y-y_)[0,:,:,0])
    # x conv2d(same) y                       
    # x pad(same)+conv2d(valid) y_ = y
    """
    x->y conv2d(same) === pad(same)[x_]+conv2d(valid)
    x->y conv2d(valid) === pad(0)+conv2d(valid)
    抽象出一个卷积操作 和tf的卷积一致 同时可以做reflect padding

    现在需要抽象出一个转置卷积操作 和tf的转置卷积一致 同时关注padding问题
    conv_transpose 特殊性在于 指定了same valid后 输入输出的维度必须满足一个逻辑约束才是正确可行的
    y->x conv2d_tanspose(valid)
    """

    x1 = tf.nn.conv2d_transpose(y_,w,x.shape,[1]+strides+[1],"SAME")
    print("padding_vect",padding_vect)
    
    x1_ = tf.pad(x1,padding_vect,"CONSTANT")
    y_2 = tf.pad(y_,[[0,0],[1,1],[1,1],[0,0]],"CONSTANT")
    print(y_2[0,:,:,0])
    x2 = tf.nn.conv2d_transpose(y_,w,x_.shape,[1]+strides+[1],"VALID")
    print(x1.shape)
    print(x1_.shape)
    print(x2.shape)
    print(x1[0,:,:,0])
    print(x1_[0,:,:,0])
    print(x2[0,:,:,0])

    print((x1_-x2)[0,:,:,0])
    filters_zp=[1]
    inshape=[16,16,16]
    k_size = [4,4,4]
    strides = [4,4,4]
    padding = "SAME"
    x = tf.random.normal([1]+inshape+[2])
    
    w = tf.random.normal(k_size+[2]+filters_zp)
    y=tf.nn.conv3d(x,w,[1]+strides+[1],padding,data_format='NDHWC')
    
    print(Reconstruction.ConvCalculation(input_shape=inshape,
                                        filters=filters_zp,
                                        kernel_size=k_size,
                                        strides=strides,
                                        padding=padding))
    out_shape,padding,padding_vect=Reconstruction.ConvCalculation(input_shape=inshape,
                                        filters=filters_zp,
                                        kernel_size=k_size,
                                        strides=strides,
                                        padding=padding)
    x_ = tf.pad(x,padding_vect,"CONSTANT")
    print(x.shape)
    print(x_.shape)
    y_ = tf.nn.conv3d(x_,w,[1]+strides+[1],"VALID")
    print(y.shape)
    print(y_.shape)
    print((y-y_)[0,:,:,:,0])

    x1 = tf.nn.conv3d_transpose(y_,w,x.shape,[1]+strides+[1],"SAME")
    print(x1[0,:,:,:,0])
    print(x1.shape)
    print("padding_vect",padding_vect)
    x1_ = tf.pad(x1,padding_vect,"CONSTANT")
    print(x1_[0,:,:,:,0])
    print(x1_.shape)
    x2 = tf.nn.conv3d_transpose(y_,w,x_.shape,[1]+strides+[1],"VALID")
    print(x2[0,:,:,:,0])
    print(x2.shape)
    print((x1_-x2)[0,:,:,:,0])
    padding,padding_vect,cut_flag = Reconstruction.Trans2UpsampleCal(
                                input_shape=[16,16],
                                output_shape=[32,32],
                                filters=8000,
                                kernel_size=[2,2],
                                strides=[2,2],
                                padding="SAME")
    print(padding,padding_vect,cut_flag)
    padding,padding_vect,cut_flag = Reconstruction.Trans2UpsampleCal(
                                input_shape=[16,16],
                                output_shape=[31,31],
                                filters=8000,
                                kernel_size=[2,2],
                                strides=[2,2],
                                padding="SAME")
    print(padding,padding_vect,cut_flag)


   