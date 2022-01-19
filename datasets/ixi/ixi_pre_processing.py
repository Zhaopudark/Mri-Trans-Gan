import numpy as np 
import os 
"""
IXI数据集本身不是维度均衡的 需要re sample 到1mmX1mmX1mm

"""
import nibabel as nib 
from scipy import ndimage
import math
# def gen_mask(img,transhole=0):
#     shape = img.shape
#     dtype = img.dtype
#     #部分样本不是int16 类型 强制mask为int16
#     #鉴于IXI的特殊性 我们只从垂直轴切片层面进行mask筛选
#     new_mask_img_1 = np.ones(shape=shape,dtype=np.int16)
#     new_mask_img_2 = np.ones(shape=shape,dtype=np.int16)
#     new_mask_img_3 = np.ones(shape=shape,dtype=np.int16)
#     for k in range(img.shape[2]):
#         for i in range(img.shape[0]):
#             for j in range(img.shape[1]): 
#                 if img[i,j,k]<=transhole:
#                     new_mask_img_1[i,j,k]=0
#                 else:
#                     break
#             for j in range(img.shape[1]-1,-1,-1):
#                 if img[i,j,k]<=transhole:
#                     new_mask_img_1[i,j,k]=0
#                 else:
#                     break
#     for k in range(img.shape[2]):
#         for j in range(img.shape[1]):
#             for i in range(img.shape[0]): 
#                 if img[i,j,k]<=transhole:
#                     new_mask_img_2[i,j,k]=0
#                 else:
#                     break
#             for i in range(img.shape[0]-1,-1,-1):
#                 if img[i,j,k]<=transhole:
#                     new_mask_img_2[i,j,k]=0
#                 else:
#                     break
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             for k in range(img.shape[2]):
#                 if img[i,j,k]<=transhole:
#                     new_mask_img_3[i,j,k]=0
#                 else:
#                     break
#             for k in range(img.shape[2]-1,-1,-1):
#                 if img[i,j,k]<=transhole:
#                     new_mask_img_3[i,j,k]=0
#                 else:
#                     break
#     new_mask_img = new_mask_img_1+new_mask_img_2+new_mask_img_3
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             for k in range(img.shape[2]):
#                 if new_mask_img[i,j,k]>0:
#                     new_mask_img[i,j,k]=1
#     return new_mask_img
def get_paried_list(path):
    buf_t1 = []
    buf_t2 = []
    for (dirName, subdirList, fileList) in os.walk(path):
        for filename in fileList:
            if "t1.nii.gz" in filename.lower(): 
                buf_t1.append(os.path.join(dirName,filename))
            if "t2.nii.gz" in filename.lower(): 
                buf_t2.append(os.path.join(dirName,filename))
    if len(buf_t1)!=len(buf_t2):
        raise ValueError("Unpaied samples!")
    else:
        return zip(buf_t1,buf_t2)

def get_del_list(path):
    del_buf = []
    for (dirName, subdirList, fileList) in os.walk(path):
        for filename in fileList:
            if "mask" in filename.lower(): 
                del_buf.append(os.path.join(dirName,filename))
            if "norm" in filename.lower(): 
                del_buf.append(os.path.join(dirName,filename))
    return del_buf
def get_mask_list(path):
    del_buf = []
    for (dirName, subdirList, fileList) in os.walk(path):
        for filename in fileList:
            if "mask" in filename.lower(): 
                del_buf.append(os.path.join(dirName,filename))
    return del_buf

# import os
# path = "G:\\Datasets\\BraTS\\Renew\\"
# file_list = get_del_list(path)
# for i,item in enumerate(file_list):
#     os.remove(item)

def get_list(path):
    del_buf = []
    for (dirName, subdirList, fileList) in os.walk(path):
        for filename in fileList:
            if "t1.nii.gz" in filename.lower(): 
                del_buf.append(os.path.join(dirName,filename))
            if "t2.nii.gz" in filename.lower(): 
                del_buf.append(os.path.join(dirName,filename))
    return del_buf
def get_norm_list(path):
    del_buf = []
    for (dirName, subdirList, fileList) in os.walk(path):
        for filename in fileList:
            if "t1_norm.nii.gz" in filename.lower(): 
                del_buf.append(os.path.join(dirName,filename))
            if "t2_norm.nii.gz" in filename.lower(): 
                del_buf.append(os.path.join(dirName,filename))
    return del_buf    
def resample(img,current_spacing,direct_spacing=[1,1,1]):
    #代表像素的1 1 1 mm
    spacing = np.array(current_spacing)
    resize_factor = spacing / direct_spacing
    direct_shape = img.shape * resize_factor #作形状变换
    direct_shape = np.round(direct_shape) #去掉小数点之后的
    resize_factor = direct_shape/img.shape#去尾数后重新计算
    direct_spacing = spacing/resize_factor
    #调整后的值 一波操作猛如虎但是啥也没变
    img = ndimage.interpolation.zoom(img, resize_factor, mode='constant')
    return img

#-------------------------配准后重采样----------------------------------#
# path ="G:\\Datasets\\IXI\\IXI-T12T2\\"
# buf = get_list(path)
# for i,item in enumerate(buf):
#     tmp = list(item)
#     for index in range(len(tmp)-1,-1,-1):
#         if tmp[index]=="\\":
#             print(i,index)
#             break
#     file_name = item[index+1::]
#     print(file_name)
    
#     T1_path = "G:\\Datasets\\IXI\\IXI-T12T2\\"+file_name[0:-9]+"T1.nii.gz"
#     if os.path.exists(T1_path):
#         print("yes")
#     else:
#         raise ValueError("Not exist.")
#     T2_path = "G:\\Datasets\\IXI\\IXI-T2\\"+file_name[0:-9]+"T2.nii.gz"
#     if os.path.exists(T2_path):
#         print("yes")
#     else:
#         raise ValueError("Not exist.")
#     T1_save_path = "G:\\Datasets\\IXI\\Registration\\IXI-T1\\"+file_name[0:-9]+"resampled_T1.nii.gz"
#     T2_save_path = "G:\\Datasets\\IXI\\Registration\\IXI-T2\\"+file_name[0:-9]+"resampled_T2.nii.gz"
#     print(T1_path)
#     print(T2_path)
#     print(T1_save_path)
#     print(T2_save_path)

#     img_1 = nib.load(T1_path)
#     current_spacing_1 = img_1.header['pixdim'][1:4]
#     img_1 = np.array(img_1.dataobj[:,:,:])

#     img_2 = nib.load(T2_path)
#     current_spacing_2 = img_2.header['pixdim'][1:4]
#     img_2 = np.array(img_2.dataobj[:,:,:])
#     print(current_spacing_1,current_spacing_2)
#     if (current_spacing_1!=current_spacing_2).any():
#         raise(ValueError)
#     print(img_1.shape,img_2.shape)
#     if img_1.shape!=img_2.shape:
#         raise(ValueError) 


    # img = nib.load(T1_path)
    # current_spacing = img.header['pixdim'][1:4]
    # print(current_spacing)
    # img = np.array(img.dataobj[:,:,:])
    # img = resample(img,current_spacing=current_spacing)
    # img_ii = nib.Nifti1Image(img,np.eye(4))
    # nib.save(img_ii,T1_save_path) 

    # img = nib.load(T2_path)
    # current_spacing = img.header['pixdim'][1:4]
    # print(current_spacing)
    # img = np.array(img.dataobj[:,:,:])
    # img = resample(img,current_spacing=current_spacing)
    # img_ii = nib.Nifti1Image(img,np.eye(4))
    # nib.save(img_ii,T2_save_path) 
# print(len(buf))

#-------------------------合成mask 乘法----------------------------------#
# path ="G:\\Datasets\\IXI\\Registration\\IXI-T1-Mask\\"
# buf = get_mask_list(path)
# for i,item in enumerate(buf):
#     print(i)
#     for index in range(len(item)-1,-1,-1):
#         if item[index]=="\\":
#             print(i,index)
#             break
#     file_name = item[index+1::]
#     print(file_name)
#     T1_mask_path = "G:\\Datasets\\IXI\\Registration\\IXI-T1-Mask\\"+file_name[0:-20]+"T1_brain_mask.nii.gz"
#     T2_mask_path = "G:\\Datasets\\IXI\\Registration\\IXI-T2-Mask\\"+file_name[0:-20]+"T2_brain_mask.nii.gz"
#     if os.path.exists(T1_mask_path):
#         print("yes")
#     else:
#         raise ValueError("Not exist.")
#     if os.path.exists(T2_mask_path):
#         print("yes")
#     else:
#         raise ValueError("Not exist.")
#     mask_save_path = "G:\\Datasets\\IXI\\Registration\\Mask\\"+file_name[0:-20]+"brain_mask.nii.gz"
#     print(mask_save_path)

#     img_1 = nib.load(T1_mask_path)
#     current_spacing_1 = img_1.header['pixdim'][1:4]
#     img_1 = np.array(img_1.dataobj[:,:,:])

#     img_2 = nib.load(T2_mask_path)
#     current_spacing_2 = img_2.header['pixdim'][1:4]
#     img_2 = np.array(img_2.dataobj[:,:,:])

#     print(current_spacing_1,current_spacing_2)
#     if (current_spacing_1!=current_spacing_2).all():
#         raise ValueError()
#     print(img_1.shape,img_2.shape)
#     if img_1.shape!=img_2.shape:
#         raise ValueError()
#     print(img_1.dtype,img_2.dtype)
#     if img_1.dtype!=img_2.dtype:
#         raise ValueError()
#     if np.isnan(img_1).all():
#         raise ValueError()
#     if np.isnan(img_2).all():
#         raise ValueError()
#     img = img_1*img_2
#     if np.isnan(img).all():
#         raise ValueError()
#     img_ii = nib.Nifti1Image(img,np.eye(4))
#     nib.save(img_ii,mask_save_path) 
# print(len(buf))
#-------------------------mask抽取后 再norm----------------------------------#
# path ="G:\\Datasets\\IXI\\Registration\\"
# buf = get_list(path)
# for i,item in enumerate(buf):
#     for index in range(len(item)-1,-1,-1):
#         if item[index]=="\\":
#             print(i,index)
#             break
#     file_name = item[index+1::]
#     print(file_name)
#     mask_path = "G:\\Datasets\\IXI\\Registration\\Mask\\"+file_name[0:-9]+"brain_mask.nii.gz"
#     if os.path.exists(mask_path):
#         print("yes")
#     else:
#         raise ValueError("Not exist.")

#     mask = nib.load(mask_path)
#     mask = np.array(mask.dataobj[:,:,:],dtype=np.int16)

#     img = nib.load(item)
#     current_spacing = img.header['pixdim'][1:4]
#     print(current_spacing,i)
#     img = np.array(img.dataobj[:,:,:],dtype=np.float32)
#     img = img*mask
#     _min = img.min()
#     _max = img.max()
#     img = ((img-_min)/(_max-_min))*0.999+0.001
#     img = img*mask
#     if np.isnan(img).all():
#         raise ValueError()
#     img_ii = nib.Nifti1Image(img,np.eye(4))
#     save_path = item[:-7]+"_norm.nii.gz"
#     print(save_path)
#     nib.save(img_ii,save_path) 
# print(len(buf))
#-------------------------检查----------------------------------#
# path ="G:\\Datasets\\IXI\\Registration\\"
# buf = get_norm_list(path)
# for i,item in enumerate(buf):
#     img = nib.load(item)
#     current_spacing = img.header['pixdim'][1:4]
#     print(current_spacing,i)
#     img = np.array(img.dataobj[:,:,:],dtype=np.float32)
#     print(img.min(),img.max(),img.dtype)
# print(len(buf))
