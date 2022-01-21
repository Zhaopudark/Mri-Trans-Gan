"""
工具类
其实里面都只是数值计算函数
"""
import numpy as np 
import math
import logging
def cut_img_3D(img):
    """
    将3D的数据进行剪裁，去除3D黑边
    img:numpy array
    return a cut img with same axis number but not a fixed one
    """
    # print(img.shape,img.dtype)
    buf=[]
    for i in range(img.shape[0]):
        temp = img[i,:,:]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[0]-1,-1,-1):
        temp = img[i,:,:]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[1]):
        temp = img[:,i,:]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[1]-1,-1,-1):
        temp = img[:,i,:]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[2]):
        temp = img[:,:,i]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[2]-1,-1,-1):
        temp = img[:,:,i]
        if(temp.sum()!=0):
            buf.append(i)
            break
    pw=1 # plus_width 前后增加的额外像素 防止3D图像缺失一小部分
    for i in range(3):
        if buf[2*i]-pw>=0:
            buf[2*i] -= pw
    for i in range(3):
        if buf[2*i+1]+pw<=(img.shape[i]-1):
            buf[2*i+1] += pw
    cut_img = img[buf[0]:buf[1]+1,buf[2]:buf[3]+1,buf[4]:buf[5]+1]
    # print(cut_img.shape) # buf 记录的是坐标下标 自身不涉及index+1 -1 认为考虑+-1
    max_length = max(cut_img.shape)
    zeros = np.zeros(shape=[1,cut_img.shape[1],cut_img.shape[2]],dtype=np.int16)
    letf_layers = max_length - cut_img.shape[0]
    for i in range(letf_layers//2):
        cut_img = np.concatenate((zeros,cut_img),axis=0)
    for i in range(letf_layers-letf_layers//2):
        cut_img = np.concatenate((cut_img,zeros),axis=0)
    # print(cut_img.shape)
    zeros = np.zeros(shape=[cut_img.shape[0],1,cut_img.shape[2]],dtype=np.int16)
    letf_layers = max_length - cut_img.shape[1]
    for i in range(letf_layers//2):
        cut_img = np.concatenate((zeros,cut_img),axis=1)
    for i in range(letf_layers-letf_layers//2):
        cut_img = np.concatenate((cut_img,zeros),axis=1)
    # print(cut_img.shape)
    zeros = np.zeros(shape=[cut_img.shape[0],cut_img.shape[1],1],dtype=np.int16)
    letf_layers = max_length - cut_img.shape[2]
    for i in range(letf_layers//2):
        cut_img = np.concatenate((zeros,cut_img),axis=2)
    for i in range(letf_layers-letf_layers//2):
        cut_img = np.concatenate((cut_img,zeros),axis=2)
    # print(cut_img.shape)
    # print(cut_img.min())
    # print(cut_img.max())
    return cut_img
def gen_hole_mask(img,pix_val=1):
    shape = img.shape
    dtype = img.dtype
    #部分样本不是int16 类型 强制mask为int16
    new_mask_img = img[:,:,:]
    new_mask_img[new_mask_img!=pix_val]=0
    new_mask_img[new_mask_img==pix_val]=1
    return new_mask_img
def gen_mask(img):
    shape = img.shape
    dtype = img.dtype
    #部分样本不是int16 类型 强制mask为int16
    new_mask_img = np.ones(shape=shape,dtype=np.int16)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i,j,k]<=0:
                    new_mask_img[i,j,k]=0
                else:
                    break
            for k in range(img.shape[2]-1,-1,-1):
                if img[i,j,k]<=0:
                    new_mask_img[i,j,k]=0
                else:
                    break
    return new_mask_img
def center_crop_3D(img,target_size):
    #img numpy array 3D
    """
    极右原则的好处是 有限剪裁末端 这样即便值剪裁单个值点 也可以使用[0:-1]实现剪裁而不必考虑末端点不需要剪裁时语法上的特殊性
    crop 等价于逆向padding
    """
    shape = img.shape
    for i in range(len(target_size)):
        if shape[i]<target_size[i]:
            raise ValueError("Unsupported target size")
        elif shape[i]==target_size[i]:
            pass
        else:
            diff = shape[i]-target_size[i]
            begin = diff//2
            end = diff-begin
            if i == 0:
                img = img[begin:-end,:,:]
            elif i == 1:
                img = img[:,begin:-end,:]
            elif i == 2:
                img = img[:,:,begin:-end]
            else:
                raise ValueError("Dim crash")
    return img
def option1_gen_whole_mask():
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pylab as plt
    import nibabel as nib
    from nibabel import nifti1
    from nibabel.viewers import OrthoSlicer3D
    import os
    buf_seg = []
    for (dirName, subdirList, fileList) in os.walk("G:\\Datasets\\BraTS\\ToCrop\\MICCAI_BraTS2020_TrainingData"):
        for filename in fileList:
            if "seg.nii" in filename.lower():  
                buf_seg.append(os.path.join(dirName,filename))
    for i,item in enumerate(buf_seg):
        img0 = nib.load(item)
        img = np.array(img0.dataobj[:,:,:],dtype=np.int16)
        save_path =item[:-7]+"mask_v1.nii"
        mask = gen_hole_mask(img,pix_val=1)
        print(i+1,mask.shape,mask.dtype)
        data = mask
        affine = img0.affine
        new_image = nib.Nifti1Image(data, affine)
        nib.save(new_image,save_path)

        img = np.array(img0.dataobj[:,:,:],dtype=np.int16)
        save_path =item[:-7]+"mask_v2.nii"
        mask = gen_hole_mask(img,pix_val=2)
        print(i+1,mask.shape,mask.dtype)
        data = mask
        affine = img0.affine
        new_image = nib.Nifti1Image(data, affine)
        nib.save(new_image,save_path)

        img = np.array(img0.dataobj[:,:,:],dtype=np.int16)
        save_path =item[:-7]+"mask_v4.nii"
        mask = gen_hole_mask(img,pix_val=4)
        print(i+1,mask.shape,mask.dtype)
        data = mask
        affine = img0.affine
        new_image = nib.Nifti1Image(data, affine)
        nib.save(new_image,save_path)
def option2_gen_mask():
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pylab as plt
    import nibabel as nib
    from nibabel import nifti1
    from nibabel.viewers import OrthoSlicer3D
    import os
    buf_A = []
    buf_B = []
    for (dirName, subdirList, fileList) in os.walk("G:\\Datasets\\BraTS\\ToCrop"):
        for filename in fileList:
            if "t1.nii" in filename.lower():  
                buf_A.append(os.path.join(dirName,filename))
            if "t2.nii" in filename.lower(): 
                buf_B.append(os.path.join(dirName,filename))
    for i,item in enumerate(buf_A):
        save_path =item[:-6]+"mask_t1_v0.nii"
        img0 = nib.load(item)
        img = np.array(img0.dataobj[:,:,:],dtype=np.int16)
        mask = gen_mask(img)
        print(i+1,mask.shape,mask.dtype)
        data = mask
        affine = img0.affine
        new_image = nib.Nifti1Image(data, affine)
        nib.save(new_image,save_path)
    for i,item in enumerate(buf_B):
        save_path =item[:-6]+"mask_t2_v0.nii"
        img0 = nib.load(item)
        img = np.array(img0.dataobj[:,:,:],dtype=np.int16)
        mask = gen_mask(img)
        print(i+1,mask.shape,mask.dtype)
        data = mask
        affine = img0.affine
        new_image = nib.Nifti1Image(data, affine)
        nib.save(new_image,save_path)  
def range_cal(mask):
    """计算
    mask 的有效范围                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    """
    if len(mask.shape)!=3:
        raise ValueError("Mask must be 3D array!")
    
    out_buf = []
    buf=[]
    for i in range(mask.shape[0]):
        if np.sum(mask[i,:,:])!=0:
            buf.append(i)
            break
    for i in range(mask.shape[0]-1,-1,-1):
        if np.sum(mask[i,:,:])!=0:
            buf.append(i)
            break
    out_buf.append(buf)  

    buf=[]
    for i in range(mask.shape[1]):
        if np.sum(mask[:,i,:])!=0:
            buf.append(i)
            break
    for i in range(mask.shape[1]-1,-1,-1):
        if np.sum(mask[:,i,:])!=0:
            buf.append(i)
            break
    out_buf.append(buf)
    
    buf=[]
    for i in range(mask.shape[2]):
        if np.sum(mask[:,:,i])!=0:
            buf.append(i)
            break
    for i in range(mask.shape[2]-1,-1,-1):
        if np.sum(mask[:,:,i])!=0:
            buf.append(i)
            break
    out_buf.append(buf)
    return out_buf

def index_cal(valid_range:list,sub_seq_len:int,sub_seq_num=None):
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
                logging.warning("sub_seq_num has been changed from {} to {}.".format(old_sub_seq_num,sub_seq_num))
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

def index_list_and(index_buf1,index_buf2):
    buf = []
    for list1,list2 in zip(index_buf1,index_buf2):
        if list1[0]>list1[1]:
            raise ValueError("index order not supported!")
        if list2[0]>list2[1]:
            raise ValueError("index order not supported!")    
        if list1[1]<list2[0]:
            return None # 无交集 
        elif list1[1]<=list2[1]:
            if list1[0]<=list2[0]:
                buf.append([list2[0],list1[1]])
            else:
                buf.append([list1[0],list1[1]])
        elif list1[0]<=list2[1]:
            if list1[0]<=list2[0]:
                buf.append([list2[0],list2[1]])
            else:
                buf.append([list1[0],list2[1]])
        else:
            return None # 无交集
    return buf 
            
def index_list_or(index_buf1,index_buf2):
    buf = []
    for list1,list2 in zip(index_buf1,index_buf2):
        if list1[0]>list1[1]:
            raise ValueError("index order not supported!")
        if list2[0]>list2[1]:
            raise ValueError("index order not supported!")   
        if list1[1]==(list2[0]-1):
            buf.append([list1[0],list2[1]])
            continue
        if list2[1]==(list1[0]-1):
            buf.append([list2[0],list1[1]])
            continue
        if list1[1]<list2[0]:
            return None # 无法合并 中间空余
        elif list1[1]<=list2[1]:
            if list1[0]<=list2[0]:
                buf.append([list1[0],list2[1]])
            else:
                buf.append([list2[0],list2[1]])
        elif list1[0]<=list2[1]:
            if list1[0]<=list2[0]:
                buf.append([list1[0],list1[1]])
            else:
                buf.append([list2[0],list1[1]])
        else:
            return None # 无法合并 中间空余
    return buf 

def index_list_rm_bias(index_buf,basic_buf):
    buf = []
    for index_list,basic in zip(index_buf,basic_buf):
        buf.append([index_list[0]-basic,index_list[1]-basic])
    return buf 
         
if __name__ == "__main__":
    # import matplotlib
    # matplotlib.use('TkAgg')
    # from matplotlib import pylab as plt
    # import nibabel as nib
    # from nibabel import nifti1
    # from nibabel.viewers import OrthoSlicer3D

    # example_filename = 'G:\\Datasets\\BraTS\\Collections\\HGG\\Brats18_2013_17_1\\Brats18_2013_17_1_t2.nii'

    # img = nib.load(example_filename)
    # img = np.array(img.dataobj[:,:,:])
    # cut_img = cut_img_3D(img)
    # print(cut_img.shape)
    # from skimage import measure
    # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # def plot_3d(image, threshold=0):
    #     # Position the scan upright,
    #     # so the head of the patient would be at the top facing the camera
    #     p = image#.transpose(2,1,0)
    #     verts, faces, norm, val = measure.marching_cubes_lewiner(p,threshold,step_size=1, allow_degenerate=True)
    #     #verts, faces = measure.marching_cubes_classic(p,threshold)
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111, projection='3d')
    #     # Fancy indexing: `verts[faces]` to generate a collection of triangles
    #     mesh = Poly3DCollection(verts[faces], alpha=0.7)
    #     face_color = [0.45, 0.45, 0.75]
    #     mesh.set_facecolor(face_color)
    #     ax.add_collection3d(mesh)
    #     ax.set_xlim(0, p.shape[0])
    #     ax.set_ylim(0, p.shape[1])
    #     ax.set_zlim(0, p.shape[2])
    #     plt.show()
    # plot_3d(cut_img)
    # option1_gen_whole_mask()
    # option2_gen_mask()

    result = index_cal(valid_range=[2,2+128-1],sub_seq_len=128,sub_seq_num=1)
    print(result)
    result = index_cal(valid_range=[2,2+128-1],sub_seq_len=128,sub_seq_num=None)
    print(result)
    result = index_cal(valid_range=[1,17],sub_seq_len=8,sub_seq_num=5)
    print(result)
    

