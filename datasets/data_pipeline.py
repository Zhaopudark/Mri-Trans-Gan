"""
构建训练集
测试集
验证集
seed
"""
import os 
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)
import tensorflow as tf
from brats import BraTS
# from ixi.ixi_pipeline import DataPipeLine as IXIDataPipeLine
__all__ = [
    "PipeLine",
]
class PipeLine():
    def __init__(self,args) -> None:
        if args.dataset.lower() == "brats":
            _data_format = args.data_format
            _norm = args.norm
            _cut_ranges = args.cut_ranges
            _patch_size = args.patch_size
            _patch_nums = args.patch_nums
            _random_seed = args.data_random_seed
            _batch_size = args.batch_size
            self.data = BraTS(path="D:\\Datasets\\BraTS\\BraTS2021_new",data_format=_data_format,norm=_norm,cut_ranges=_cut_ranges,patch_size=_patch_size,patch_nums=_patch_nums,random_seed=_random_seed)
        elif args.dataset.lower() == "ixi":
            # DataPipeLine = IXIDataPipeLine
            # train_path = "G:\\Datasets\\IXI\\Registration_train"
            # test_path = "G:\\Datasets\\IXI\\Registration_test"
            # seed_path = "G:\\Datasets\\IXI\\Registration_seed"
            # validation_path = "G:\\Datasets\\IXI\\Registration_validate"
            pass
        else:
            raise ValueError("Unsupported dataset {}".format(args.dataset))
        _output_shapes = self.data.pipline_output_shape
        _output_types = self.data.pipline_output_dtype
        __map_func = self.data.add_channel_before_batch
        def __set_wrapper(pipeline): # tf.data.Dataset.from_generator 传递的一定是tensor
            dataset = tf.data.Dataset.from_generator(pipeline,output_types=_output_types,output_shapes=_output_shapes)\
                                    .map(__map_func,num_parallel_calls=4)\
                                    .batch(_batch_size)\
                                    .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)             
            return dataset
        self.data_format = "B"+_data_format+"C"
        buf = []
        for shape in _output_shapes:
            buf.append([_batch_size]+list(shape)[:]+[1])
        self.input_shape_for_model = buf
        self.patch_combiner = self.data.patch_combiner
        self.train_set,self.test_set,self.validation_set,self.seed_set= list(map(__set_wrapper,self.data.piplines))
    def __call__(self):
        return  self
if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    class aaacc():
        def __init__(self) -> None:
            pass
    args = aaacc()
    args.dataset= "braTS"
    args.norm="z_score"
    args.data_format= "DHW"
    args.cut_ranges=[[0,239],[0,239],[155//2-8,155//2+7]]
    args.patch_size=[128,128,3]
    args.patch_nums=[6,6,6]
    args.batch_size =1
    args.data_random_seed = None
    pipe_line = PipeLine(args)
    pipe_line2 = PipeLine(args)
    for (imgs,img_masks,padding_vectors),(imgs2,img_masks2,padding_vectors2) in zip(pipe_line.train_set,pipe_line2.train_set):
        # print(imgs.shape,img_masks.shape,padding_vectors.shape)
        print(imgs.shape,tf.reduce_mean(imgs-imgs2))
        # print(imgs.dtype,img_masks.dtype,padding_vectors.dtype)
        # print(imgs.numpy().min(),img_masks.numpy().min(),padding_vectors.numpy().min())
        # print(imgs.numpy().max(),img_masks.numpy().max(),padding_vectors.numpy().max())
        # print(pipe_line.input_shape_for_model)
    # import itertools
    # import random
    
    # LIST = [1,2,3,4,5,6,7,8]
    # def gen():
    #     random.seed(100)
    #     random.shuffle(LIST)
    #     for x in LIST:
    #         yield x
    # train_set = itertools.cycle(gen())
    # for i in range(333):
    #     x = next(train_set)
    #     if i%10==0:
    #         print("****",i)
    #     else:
    #         print(x,end="")

    
    # from matplotlib import pyplot as plt
    # for x,x_m,x_v,y,y_m,y_v,m,m_m,m_v in pipe_line.seed_set:
    #     print(x.shape,y_m.shape,m_v.shape)
    #     print(x.dtype,y_m.dtype,m_v.dtype)
    #     print(x.numpy().min(),y_m.numpy().min(),m_v.numpy().min())
    #     print(x.numpy().max(),y_m.numpy().max(),m_v.numpy().max())
    #     fig = plt.figure()
    #     plt.imshow(x[0,0,:,:,0],cmap="gray")
    #     plt.show()
    #     plt.close()
    #     fig = plt.figure()
    #     plt.imshow(x_m[0,0,:,:,0],cmap="gray")
    #     plt.show()
    #     plt.close()
    # X_V = tf.random.normal(shape=[1,3,2],dtype=tf.float32)
    # tmp = tf.constant([[0,0]],dtype=tf.float32)
    # X_V= tf.concat([tmp,X_V[0,:],tmp], axis=0)
    # print(X_V)