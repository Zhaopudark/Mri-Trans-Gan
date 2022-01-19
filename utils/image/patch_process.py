import os 
import sys
import math
import tensorflow as tf
import random
import tensorflow.experimental.numpy as tnp
import numpy as np

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from utils.types_check import type_check
__all__ = [
    "PacthesProcesser",
]
class PacthesProcesser():
    def __init__(self) -> None:
        pass
    def _index_cal(self,valid_range:list,sub_seq_len:int,sub_seq_num=None,method=""):
        """计算
        如果给定一个数组某个维度的范围视为有效范围
        在这个有效范围内,分割出指定个数的子范围,确保这些子范围的并集与有效范围全等
        策略:
        先选取中间范围 且使得有效范围的中心位置元素(或者偶数个元素时,中心右侧元素)为中间范围的中心元素(或者偶数个元素时,中心右侧元素) [0,1,2,3] 2为中心  [0,1,2,3,4] 时 2为中心
        然后遵循先左后右的原则从中心扩散出若干区间
        如此 便可以保证 返回的序列的中心序列包含中心点,
        遇到边界时,考虑重叠范围,故不需要每个范围都考虑重叠,而是在边界处考虑(最多考虑两次边界,未证明是否只要考虑左边界一次) 

        valid_range:有效范围的左右下标
        sub_seq_len:子范围长度
        sub_seq_num:子范围个数
        返回list[list] [[a,b],[...]]  a,b 表示第一个区间的左下标和右下标  已经自动排序 所以中心序列的中心值就是原本的中心点                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        """
        start_index = valid_range[0]
        end_index = valid_range[1]
        valid_len = end_index-start_index+1
        # if valid_len < sub_seq_len:
        #     bigger_len = sub_seq_len-valid_len
        #     end_index += bigger_len//2
        #     start_index -= bigger_len-(bigger_len//2)
        if sub_seq_num is None:
            raise ValueError("Must give specific sub_seq_num!")
        else:
            if sub_seq_num==1:#中心剪裁
                if valid_len < sub_seq_len:
                    bigger_len = sub_seq_len-valid_len
                    end_index += bigger_len//2
                    start_index -= bigger_len-(bigger_len//2)
                else:
                    pass
            else:#非中心剪裁 要求可以覆盖所有有效区域
                if sub_seq_num*sub_seq_len < valid_len:
                    old_sub_seq_num = sub_seq_num
                    sub_seq_num = math.ceil(valid_len/sub_seq_len)
                    print("Warning: sub_seq_num has been changed from {} to {}.".format(old_sub_seq_num,sub_seq_num))
                else:
                    pass
        index_range_buf = []
        mid_index = math.ceil((start_index+end_index)/2)
        left_index = mid_index-sub_seq_len//2 # 中心区间的左下标
        right_index = left_index+sub_seq_len-1 # 中心区间的右下标
        for i in range(sub_seq_num):
            if i%2==0:
                index_range_buf.append([left_index,right_index])
                left_index = index_range_buf[0][0]
                left_index -= sub_seq_len
                if left_index < start_index:
                    left_index = start_index
                right_index = left_index+sub_seq_len-1
            else:
                index_range_buf.insert(0,[left_index,right_index])
                right_index = index_range_buf[-1][-1]
                right_index += sub_seq_len
                if right_index > end_index:
                    right_index = end_index
                left_index = right_index-sub_seq_len+1
        return index_range_buf
    class GetPatches():
        def __init__(self,cut_ranges=None,patch_size=None,patch_nums=None,random_seed=None) -> None:
            if type_check(cut_ranges,mode=[list,list],meta_instance_or_type=int):
                self._cut_ranges = cut_ranges
            else:
                raise ValueError("")
            if type_check(patch_size,mode=[list],meta_instance_or_type=int):
                self._patch_size = patch_size
            else:
                raise ValueError("")
            if type_check(patch_nums,mode=[list],meta_instance_or_type=int):
                self._patch_nums = patch_nums
            else:
                raise ValueError("")
            self._random_seed = random_seed
            self._input_shape_checked = False
        @property
        def output_shape(self):
            shape = []
            cut_shape = [b-a+1 for (a,b) in self._cut_ranges]
            shape.append(tuple(self._patch_size))
            shape.append(tuple(cut_shape))
            shape.append(tuple([len(self._cut_ranges),len(self._cut_ranges[0])]))
            return tuple(shape)
        @property
        def output_dtype(self):
            dtype = []
            dtype.append(tf.float32)
            dtype.append(tf.float32) #mask 将参与div_no_nan计算因此不能是int32 一般化取float32
            dtype.append(tf.int32)
            return tuple(dtype)
            
        def _input_shape_check(self,img,cut_ranges,patch_size,patch_nums,):
            dims = len(img.shape)
            if (dims!=len(patch_size))or(dims!=len(patch_nums))or(dims!=len(cut_ranges)):
                raise ValueError("Dims of img and patch sizes|nums|ranges must be the same!")
            else:
                self._input_shape_checked = True
        def _div_lists(self,l):
            """
            NOTE 设计一个深度优先搜索算法的应用(卧槽 写出来变成了广度优先搜索)
            针对仅有两个子列表的列表[[a,b],[c,d,e]],将列表拆为[[a,c],[a,d],[a,e],[b,c],[b,d],[b,e]] 其中a,b,c,d,e为抽象的最小单元,可以是列表
            如果列表多余两个子列表[[a,b],[c,d],[e,f]],则先将前两个列表合并[[[a,c],[a,d],[b,c],[b,d]],[e,f]] 前一个列表为最小元的列表，可以使用+合并后一个列表组成新的最小元列表
            """
            def _ismeta(l):# 在本方法中用于判断l是否是拆分列表的最小元 即[int1,int2]为一个最小元
                if isinstance(l,list):
                    if  not isinstance(l[0],list):
                        return True
                    else:
                        return False
                else:
                    raise ValueError("The initial list was over disassembled!")
            out_buf = []
            if isinstance(l,list):
                if (len(l)==1):
                    for a in l[0]:
                        if _ismeta(a):
                            a = [a]
                            out_buf.append(a)
                elif (len(l)==2):
                    for a in l[0]:
                        for b in l[1]:
                            if _ismeta(a):
                                a = [a]
                            if _ismeta(b):
                                b = [b]
                            out_buf.append(a+b)
                elif (len(l)>2):
                    for a in self._div_lists(l[0:-1]):
                        for b in l[-1]:
                            if _ismeta(a):
                                a = [a]
                            if _ismeta(b):
                                b = [b]
                            out_buf.append(a+b)
                else:
                    raise ValueError()
            else:
                raise ValueError()
            return out_buf
        def _index_list_to_slice_range(self,index_range_list):
            """
            为了通过tf.slice 获得 index_range 描述的区域
            计算tf.slice需要的对应的begin,size
            """
            begin = []
            size = []
            for index_range in index_range_list:
                begin.append(int(index_range[0]))
                size.append(int(index_range[1]+1-index_range[0]))
            return begin,size
        def _index_list_to_padding_vector_by_cut_ranges(self,cut_range_list,index_range_list):
            """
            为了通过 zero padding 获得 index_range 描述的区域之于整体的0-1mask
            计算tf.pad(ones,padding=?)中的padding vector
            """
            padding_vector = []
            for i,(cut_range,index_range) in enumerate(zip(cut_range_list,index_range_list)):
                padding_vector.append([index_range[0]-cut_range[0],cut_range[1]-index_range[1]]) # index_range=[a,b] cut_range = [A,B] 一般地 有A<=a<b<=B 所以左边pad a-A个0 右边pad B-b个0
            return tf.constant(padding_vector,dtype=tf.int32) # 变为tf.pad可接受的int32形式
        def get_center_patches(self,img):
            if not self._input_shape_checked:
                self._input_shape_check(img,self._cut_ranges,self._patch_size,self._patch_nums)
            else:
                pass
            patch_index_buf = []
            for i in range(len(img.shape)):
                patch_index_buf.append(PacthesProcesser._index_cal(PacthesProcesser,valid_range=self._cut_ranges[i],sub_seq_len=self._patch_size[i],sub_seq_num=self._patch_nums[i]))
            patch_index_lists = self._div_lists(patch_index_buf)
            if self._random_seed is not None:
                random.seed(int(self._random_seed))
                random.shuffle(patch_index_lists)
            else:
                pass
            for patch_index_list in patch_index_lists:
                begin,size = self._index_list_to_slice_range(patch_index_list)
                patch_in_list = tf.slice(img,begin=begin,size=size)
                padding_vector = self._index_list_to_padding_vector_by_cut_ranges(self._cut_ranges,patch_index_list)
                mask_in_list = tf.pad(tf.ones(shape=patch_in_list.shape),paddings=padding_vector) 
                padding_vector_in_list = padding_vector
                yield patch_in_list,mask_in_list,padding_vector_in_list

    def combine_patches(self,generator_p_m_v):
        """
        输入:generator_p_m_v 可迭代对象
        输出: generator_p_m_v迭代截止后的结果
        """
        def __mask_and(mask_1,mask_2): # mask必须为0-1 mask
            return mask_1*mask_2 
        def __mask_or(mask_1,mask_2): # mask必须为0-1 mask
            add_buf = mask_1+mask_2
            out = tf.where(add_buf>=1.0,x=1.0,y=0.0)
            return out
        def __padding_vector_reshape(padding_vector):
            if len(padding_vector.shape)>2:
                padding_vector = tf.reshape(padding_vector,shape=[-1]+list(padding_vector.shape[-2::]))
                return padding_vector[0,:,:]
            elif len(padding_vector.shape)==2:
                return padding_vector[:,:]
            else:
                raise ValueError("The shape of padding_vector must be [N,2]")
        def __add_patch_add_mask(patch,mask,new_patch,new_mask,padding_vector=None,new_padding_vector=None):
            padding_vector = __padding_vector_reshape(padding_vector)
            new_padding_vector = __padding_vector_reshape(new_padding_vector)
            board = tf.pad(patch,paddings=padding_vector)
            patch = tf.pad(new_patch,paddings=new_padding_vector)
            out_board = board+patch
            out_mask = mask+new_mask # mask的代数和代表了权重 若最终div该权重即使得输出值回归正常
            out_padding_vector = tf.zeros_like(padding_vector)
            return out_board,out_mask,out_padding_vector
        for i,(current_patch,current_mask,current_padding_vector) in enumerate(generator_p_m_v):
            if i == 0:
                patch_0,mask_0,padding_vector_0 = current_patch,current_mask,current_padding_vector
            else:
                patch_1,mask_1,padding_vector_1 = current_patch,current_mask,current_padding_vector
                patch_0,mask_0,padding_vector_0 = __add_patch_add_mask(patch_0,mask_0,patch_1,mask_1,padding_vector_0,padding_vector_1)
        patch_0 = tf.math.divide_no_nan(patch_0,mask_0)
        mask_0 = tf.where(mask_0>=1.0,x=1.0,y=0.0)
        return patch_0,mask_0,padding_vector_0
    def combine_n_patches(self,generator_p_m_v,total_patch_nums):
        """
        输入: generator_p_m_v 可迭代对象
              total_patch_nums 每迭代多少个进行一次输出
        输出: 可迭代对象
        """
        def __mask_and(mask_1,mask_2): # mask必须为0-1 mask
            return mask_1*mask_2 
        def __mask_or(mask_1,mask_2): # mask必须为0-1 mask
            add_buf = mask_1+mask_2
            out = tf.where(add_buf>=1.0,x=1.0,y=0.0)
            return out
        def __padding_vector_reshape(padding_vector):
            if len(padding_vector.shape)>2:
                padding_vector = tf.reshape(padding_vector,shape=[-1]+list(padding_vector.shape[-2::]))
                return padding_vector[0,:,:]
            elif len(padding_vector.shape)==2:
                return padding_vector[:,:]
            else:
                raise ValueError("The shape of padding_vector must be [N,2]")
        def __add_patch_add_mask(patch,mask,new_patch,new_mask,padding_vector,new_padding_vector):
            padding_vector = __padding_vector_reshape(padding_vector)
            new_padding_vector = __padding_vector_reshape(new_padding_vector)
            board = tf.pad(patch,paddings=padding_vector)
            patch = tf.pad(new_patch,paddings=new_padding_vector)
            out_board = board+patch
            out_mask = mask+new_mask # mask的代数和代表了权重 若最终div该权重即使得输出值回归正常
            out_padding_vector = tf.zeros_like(padding_vector)
            return out_board,out_mask,out_padding_vector
        for i,(current_patch,current_mask,current_padding_vector) in enumerate(generator_p_m_v):
            if i == 0:
                patch_0,mask_0,padding_vector_0 = current_patch,current_mask,current_padding_vector
            else:
                patch_1,mask_1,padding_vector_1 = current_patch,current_mask,current_padding_vector
                patch_0,mask_0,padding_vector_0 = __add_patch_add_mask(patch_0,mask_0,patch_1,mask_1,padding_vector_0,padding_vector_1)
            if i%int(total_patch_nums) >= int(total_patch_nums)-1:
                patch_0 = tf.math.divide_no_nan(patch_0,mask_0)
                mask_0 = tf.where(mask_0>=1.0,x=1.0,y=0.0)
                yield patch_0,mask_0,padding_vector_0
    def combine_n_patches_x(self,generator_p_m_v,total_patch_nums):
        """
        输入: generator_p_m_v 可迭代对象 不同于combine_n_patches方法中的combine_n_patches 本方法支持多组输出
              total_patch_nums 每迭代多少个进行一次输出
        输出: 可迭代对象
        """
        def __mask_and(mask_1,mask_2): # mask必须为0-1 mask
            return mask_1*mask_2 
        def __mask_or(mask_1,mask_2): # mask必须为0-1 mask
            add_buf = mask_1+mask_2
            out = tf.where(add_buf>=1.0,x=1.0,y=0.0)
            return out
        def __padding_vector_reshape(padding_vector):
            if len(padding_vector.shape)>2:
                padding_vector = tf.reshape(padding_vector,shape=[-1]+list(padding_vector.shape[-2::]))
                return padding_vector[0,:,:]
            elif len(padding_vector.shape)==2:
                return padding_vector[:,:]
            else:
                raise ValueError("The shape of padding_vector must be [N,2]")
        def __add_patch_add_mask(patch,mask,new_patch,new_mask,padding_vector=None,new_padding_vector=None):
            padding_vector = __padding_vector_reshape(padding_vector)
            new_padding_vector = __padding_vector_reshape(new_padding_vector)
            board = tf.pad(patch,paddings=padding_vector)
            patch = tf.pad(new_patch,paddings=new_padding_vector)
            out_board = board+patch
            out_mask = mask+new_mask # mask的代数和代表了权重 若最终div该权重即使得输出值回归正常
            out_padding_vector = tf.zeros_like(padding_vector)
            return out_board,out_mask,out_padding_vector
        buf_0 = []
        for i,current_list in enumerate(generator_p_m_v):
            if i == 0:
                for patch,mask,padding_vector in current_list:
                    buf_0.append([patch,mask,padding_vector])
            else:
                buf_1 = []
                for (patch_1,mask_1,padding_vector_1),(patch_0,mask_0,padding_vector_0) in zip(current_list,buf_0):
                    patch_0,mask_0,padding_vector_0 = __add_patch_add_mask(patch_0,mask_0,patch_1,mask_1,padding_vector_0,padding_vector_1)
                    buf_1.append([patch_0,mask_0,padding_vector_0])
                buf_0 = buf_1
            if i%int(total_patch_nums) >= int(total_patch_nums)-1:
                buf_1 = []
                for patch,mask,padding_vector in buf_0:
                    patch = tf.math.divide_no_nan(patch,mask)
                    mask = tf.where(mask>=1.0,x=1.0,y=0.0)
                    buf_1.append([patch,mask,padding_vector])
                buf_0 = buf_1
                yield buf_0
    
        
#------------------------------------------------------------------------------------------------------------------------------#
def _my_slice(img,patch_index_lists):
    """
    保留一个自行设计的类似于tf.slice的方法 
    patch_index_lists 为
        [
            [
                [轴0下标下届,轴0下标上届(可取得)],
                [轴1下标下届,轴1下标上届(可取得)],
                ...(Y个维度,与img维度数相同且一一对应),
            ],
            ...(X组),
        ]
    返回X组img的切片
    """
    img_list = []
    for patch_index_list in patch_index_lists:
        current_img = img
        board = tf.zeros(shape=img.shape,dtype=tf.int16)
        for i,index_range in enumerate(patch_index_list):
            perm = [x for x in range(len(patch_index_list))]
            perm[0],perm[i] = perm[i],perm[0]
            tmp_img = tf.transpose(current_img,perm=perm)
            current_img = tf.reshape(tmp_img,shape=[tmp_img.shape[0],-1])
            current_img = current_img[index_range[0]:index_range[1]+1,:]
            current_img = tf.reshape(current_img,shape=[index_range[1]+1-index_range[0]]+tmp_img.shape[1::])
            current_img = tf.transpose(current_img,perm=perm)
        img_list.append(current_img)
    return img_list


if __name__=="__main__":
    p = PacthesProcesser()
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    img = tf.random.uniform(shape=[155,240,240],minval=-10.0,maxval=10.0)
    a = p.GetPatches(cut_ranges=[[155//2-8,155//2+7],[0,239],[0,239]],patch_size=[3,128,128],patch_nums=[6,6,6],random_seed=0)
    a = a.get_center_patches(img=img)
    for i,(item1,item2,item3) in enumerate(a):
        print(i,item1.shape,item2.shape,item3.shape)
        # print(tf.math.count_nonzero(item2))
        # print(3*128*128)
        # print(tnp.nonzero(item2.numpy())[0].numpy().min(),tnp.nonzero(item2.numpy())[0].numpy().max())
        valid_range = tnp.nonzero(item2.numpy())
        indices_buf = tf.transpose(tf.stack([item.numpy() for item in valid_range]),perm=[1,0])
        values = tf.ones(shape=[indices_buf.shape[0]])
        item2_ = tf.scatter_nd(indices=indices_buf,updates=values,shape=item2.shape)
        item2__ = tf.SparseTensor(indices=tf.cast(indices_buf,tf.int64),values=values,dense_shape=tf.cast(item2.shape,tf.int64))
        print(tf.reduce_mean(item2_-item2))
        print(tf.reduce_mean(tf.sparse.to_dense(item2__)-item2))
    a = p.GetPatches(cut_ranges=[[155//2-8,155//2+7],[0,239],[0,239]],patch_size=[3,128,128],patch_nums=[6,6,6],random_seed=0)
    a = a.get_center_patches(img=img)
    out_patch,out_mask,out_padding_vector = p.combine_patches(a)
    print(out_patch.shape,out_patch.dtype)
    print(out_mask.shape,out_mask.dtype)
    print(out_padding_vector)
    print(tnp.nonzero(out_mask))
    def _np_zero_close(x):
        x = np.array(x)
        _where = np.isclose(x,0.0,rtol=1e-05,atol=1e-08,equal_nan=False)# 与0比较 其余取默认值(默认nan与nan不相 等返回false,nan与非nan不相等,返回false)
        x[_where] = 0
        return np.array(x)#返回新数组避免出问题
    print("out-error",tf.reduce_mean(out_patch-img[155//2-8:155//2+8,0:240,0:240]*out_mask))
    out = out_patch-img[155//2-8:155//2+8,0:240,0:240]*out_mask
    print("out-error",tf.reduce_mean(_np_zero_close(out.numpy())))
    print("out-error",np.mean(_np_zero_close(out.numpy())))
    print(tf.reduce_mean(img-(img+img+img)/3.0)) # 证明计算误差确实存在 但是该方法已经是最均衡的方法了
    out = img-(img+img+img)/3.0
    print(tf.reduce_mean(_np_zero_close(out.numpy())))
    print(np.mean(_np_zero_close(out.numpy())))

    # def iter_wrapper(iterable,iter_nums=216):
    #     return p.combine_n_patches_x(iterable,total_patch_nums=iter_nums)
    # def gen_img(nums):
    #     for i in range(nums):
    #         yield tf.random.uniform(shape=[155,240,240],minval=-10.0,maxval=10.0)
    # def gen_patches(nums):
    #     for item in gen_img(nums):
    #         a = p.GetPatches(cut_ranges=[[155//2-8,155//2+7],[0,239],[0,239]],patch_size=[3,128,128],patch_nums=[6,6,6],random_seed=1)
    #         a = a.get_center_patches(img=img)
    #         for i,(item1,item2,item3) in enumerate(a):
    #             print(i)
    #             yield [(item1,item2,item3),(item1,item2,item3),(item1,item2,item3),(item1,item2,item3)]

    # for i,*out in enumerate(gen_patches(2)):
    #     print("out",i)
    # for i,*out in enumerate(iter_wrapper(gen_patches(10))):
    #     print("out",i)

    