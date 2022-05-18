import os 
import sys
from PIL import Image
import numpy as np 
import nibabel as nib
from scipy import ndimage
import random
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from utils.cut_padding import index_cal,range_cal,index_list_and,index_list_or,index_list_rm_bias
class DataPipeLine():
    def __init__(self,path,patch_size=[128,128,3],max_thick=16,sub_seq_num_list=[1,1,1],random_flag=True):
        """
        path 文件夹路径 确保同一个患者的所有模态在同一个子文件夹下, 患者文件夹之间不需要确保同级目录
        patch_size 输出的patch 大小
        max_thick 最大的第三维度 即厚度  最好是人为给定 避免考察太多小脑部分
        sub_seq_num_list 每个对应维度在有效范围内的个数
        目前没有根据有效像素获得有效范围 故
            第一 第二维度有效范围都是0--239 即最大长度 240
            第三维度有效范围手动指定为中心的100长度
        """
        self.path = path
        self.datalist = self.gen_file_list(path)
        self.patch_size = patch_size
        self.max_thick = max_thick
        self.sub_seq_num_list = sub_seq_num_list
        self.random_flag = random_flag
    def gen_file_list(self,path):
        buf_t1_norm = []
        buf_t2_norm = []
        buf_mask = []
        for (dirName, _, fileList) in os.walk(path):
            for filename in fileList:
                if "t1_norm.nii.gz" in filename.lower():  
                    tmp = filename[:]
                    print(tmp)
                    print(dirName)
                    buf_t1_norm.append(os.path.join(dirName,filename))
                    buf_t2_norm.append(os.path.join(dirName[0:-6]+"IXI-T2",tmp[0:-14]+"T2_norm.nii.gz"))
                    buf_mask.append(os.path.join(dirName[0:-6]+'Mask',tmp[0:-14]+"brain_mask.nii.gz"))
        file_list = list(zip(buf_t1_norm,buf_t2_norm,buf_mask))
        return file_list
    def read_nii_file(self,path):
        img = nib.load(path)
        img = np.array(img.dataobj[:,:,:])
        return img
    def read_single_group(self,single_group_path=None):#读取一个患者文件夹下是三个模态文件并返回
        img_t1 = self.read_nii_file(single_group_path[0])
        img_t2 = self.read_nii_file(single_group_path[1])
        img_mask = self.read_nii_file(single_group_path[2])
        return img_t1,img_t2,img_mask  
    def gen_patches(self,img_t1,img_t2,img_mask): # 按照指定的patch_size和策略 先计算在原3D矩阵中抽取的patch下标范围,然后应用这些下标,返回抽取的部分
        out_buf = []
        valid_range_list = range_cal(img_mask)
        
        # print(valid_range_list)
        # print()

        shape = img_t1.shape
        left_index =  0
        right_index = shape[0]-1
        patch_dim_0_index_list = index_cal(valid_range=valid_range_list[0],sub_seq_len=self.patch_size[0],sub_seq_num=self.sub_seq_num_list[0])

        # print(patch_dim_0_index_list)
        # print()

        left_index =  0
        right_index = shape[1]-1
        patch_dim_1_index_list = index_cal(valid_range=valid_range_list[1],sub_seq_len=self.patch_size[1],sub_seq_num=self.sub_seq_num_list[1])

        # left_index =  shape[2]//2-self.patch_size[2]//2
        # right_index = left_index+self.patch_size[2]-1
        left_index =  shape[2]//2-self.max_thick//2 # 前面两个维度可以中心中心剪裁,但是第三个维度还是手动指定比较好
        right_index = left_index+self.max_thick-1
        patch_dim_2_index_list = index_cal(valid_range=[left_index,right_index],sub_seq_len=self.patch_size[2],sub_seq_num=self.sub_seq_num_list[2])

        index_buf = []
        #得到在三个维度上的可取区间的下标 以这些下标开始攫取数组
        for dim_0 in patch_dim_0_index_list:
            for dim_1 in patch_dim_1_index_list:
                for dim_2 in patch_dim_2_index_list:
                    
                    out_buf.append((img_t1[dim_0[0]:dim_0[1]+1,
                                           dim_1[0]:dim_1[1]+1,
                                           dim_2[0]:dim_2[1]+1
                                           ],
                                    img_t2[dim_0[0]:dim_0[1]+1,
                                           dim_1[0]:dim_1[1]+1,
                                           dim_2[0]:dim_2[1]+1
                                           ],
                                    img_mask[dim_0[0]:dim_0[1]+1,
                                             dim_1[0]:dim_1[1]+1,
                                             dim_2[0]:dim_2[1]+1
                                             ]
                                    ))
                    index_buf.append([[dim_0[0],dim_0[1]],[dim_1[0],dim_1[1]],[dim_2[0],dim_2[1]]])
        return out_buf,index_buf
    def generator(self):
        if self.random_flag:
            random.shuffle(self.datalist)
        for _,single_group in enumerate(self.datalist):
            t1,t2,mask= self.read_single_group(single_group)
            buf,index_buf = self.gen_patches(t1,t2,mask)
            for item,index_buf in zip(buf,index_buf):
                yield item[0],item[1],item[2],np.array(index_buf)
        return None

if __name__ == '__main__':
    import time 
    from matplotlib import pyplot as plt
    c = DataPipeLine(path="G:\\Datasets\\IXI\\Registration_validate",
                     patch_size=[128,128,3],
                     sub_seq_num_list=[3,3,6],
                     random_flag=False)

    # for i,(t1,t2,mask,index_buf) in enumerate(c.generator()):
    #     pass
    #     print(i,t1.min(),t1.max(),t2.min(),t2.max(),mask.min(),mask.max()) 
    #     print(i,t1.shape,t1.dtype,t2.shape,t2.dtype,mask.shape,mask.dtype)
    #     print(i,index_buf)
    patch_nums = 3*3*6
    patch_i = 0
    patch_buf = []
    basic_buf = [0,0,0]
    for _ in range(1):
        start = time.perf_counter() 
        for i,(t1,t2,mask,index_buf) in enumerate(c.generator()):
            if patch_i==0:
                black_board = np.array(np.zeros(shape=[240,240,16]))
                basic_buf = [0,0,index_buf[2,0]]
            patch_buf = index_list_rm_bias(index_buf,basic_buf)
            black_board[patch_buf[0][0]:patch_buf[0][1]+1,patch_buf[1][0]:patch_buf[1][1]+1,patch_buf[2][0]:patch_buf[2][1]+1] = mask[:,:,:]
            plt.imshow(black_board[:,:,7])
            plt.show()
            print(i,t1.min(),t1.max(),t2.min(),t2.max(),mask.min(),mask.max()) 
            print(i,t1.shape,t1.dtype,t2.shape,t2.dtype,mask.shape,mask.dtype)
            patch_i += 1
            patch_i %= patch_nums
        print(time.perf_counter() -start)
    print(np.mean(t1),np.mean(t2),np.mean(mask))
        


    