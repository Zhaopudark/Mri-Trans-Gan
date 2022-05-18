import numpy as np 
import os 
"""
IXI数据集本身不是维度均衡的 需要re sample 到1mmX1mmX1mm

"""
import nibabel as nib 
from scipy import ndimage
import math
import random
from shutil import copyfile

# def get_del_list(path):
#     buf = []
#     for (dirName, _, fileList) in os.walk(path):
#         for filename in fileList:
#             if "t1.nii.gz" in filename.lower():  
#                 buf.append(os.path.join(dirName,filename))
#             if "t2.nii.gz" in filename.lower():  
#                 buf.append(os.path.join(dirName,filename))
#     return buf


def get_list(path):
    buf_t1_norm = []
    buf_t2_norm = []
    buf_mask = []
    for (dirName, _, fileList) in os.walk(path):
        for filename in fileList:
            if "t1_norm.nii.gz" in filename.lower():  
                tmp = filename[:]
                # print(tmp)
                # print(dirName)
                buf_t1_norm.append(os.path.join(dirName,filename))
                buf_t2_norm.append(os.path.join(dirName[0:-6]+"IXI-T2",tmp[0:-14]+"T2_norm.nii.gz"))
                buf_mask.append(os.path.join(dirName[0:-6]+'Mask',tmp[0:-14]+"brain_mask.nii.gz"))
    file_list = list(zip(buf_t1_norm,buf_t2_norm,buf_mask))
    return file_list

# file_list = get_list("G:\\Datasets\\IXI\\Registration_train")
# for i,(t1,t2,mask) in enumerate(file_list):
#     print(i)
#     print(t1)
#     print(t2)
#     print(mask)
# resultList=random.sample(range(0,577),177)
# print(resultList)

# for index in resultList:
#     t1,t2,mask = file_list[index]
#     cp_t1 = t1[:]
#     cp_t1 = cp_t1.replace('Registration_train','Registration_test')
#     cp_t2 = t2[:]
#     cp_t2 = cp_t2.replace('Registration_train','Registration_test')
#     cp_mask = mask[:]
#     cp_mask = cp_mask.replace('Registration_train','Registration_test')

#     copyfile(t1, cp_t1)
#     os.remove(t1)
#     print(cp_t1)

#     copyfile(t2, cp_t2)
#     os.remove(t2)
#     print(cp_t2)

#     copyfile(mask, cp_mask)
#     os.remove(mask)
#     print(cp_mask)
    

# file_list = get_list("G:\\Datasets\\IXI\\Registration_test")
# resultList=random.sample(range(0,177),25)
# for index in resultList:
#     t1,t2,mask = file_list[index]
#     cp_t1 = t1[:]
#     cp_t1 = cp_t1.replace('Registration_test','Registration_validate')
#     cp_t2 = t2[:]
#     cp_t2 = cp_t2.replace('Registration_test','Registration_validate')
#     cp_mask = mask[:]
#     cp_mask = cp_mask.replace('Registration_test','Registration_validate')
#     copyfile(t1, cp_t1)
#     print(cp_t1)
#     copyfile(t2, cp_t2)
#     print(cp_t2)
#     copyfile(mask, cp_mask)
#     print(cp_mask)

# file_list = get_list("G:\\Datasets\\IXI\\Registration_validate")
# resultList=random.sample(range(0,25),1)
# for index in resultList:
#     t1,t2,mask = file_list[index]
#     cp_t1 = t1[:]
#     cp_t1 = cp_t1.replace('Registration_validate','Registration_seed')
#     cp_t2 = t2[:]
#     cp_t2 = cp_t2.replace('Registration_validate','Registration_seed')
#     cp_mask = mask[:]
#     cp_mask = cp_mask.replace('Registration_validate','Registration_seed')
#     copyfile(t1, cp_t1)
#     print(cp_t1)
#     copyfile(t2, cp_t2)
#     print(cp_t2)
#     copyfile(mask, cp_mask)
#     print(cp_mask)

# file_list = get_del_list("G:\\Datasets\\IXI\\Registration_train")
# for i,item in enumerate(file_list):
#     os.remove(item)

#-----------------------------------失误导致的需要重新复制-----------------------------#
# file_list = get_list("G:\\Datasets\\IXI\\Registration_validate")
# for t1,t2,mask in file_list:
#     cp_t1 = t1[:]
#     cp_t1 = cp_t1.replace('Registration_validate','Registration_test')
#     cp_t2 = t2[:]
#     cp_t2 = cp_t2.replace('Registration_validate','Registration_test')
#     cp_mask = mask[:]
#     cp_mask = cp_mask.replace('Registration_validate','Registration_test')
#     copyfile(cp_t1,t1)
#     print(cp_t1)
#     copyfile(cp_t2,t2)
#     print(cp_t2)
#     copyfile(cp_mask,mask)
#     print(cp_mask)

#-----------------------------------失误导致的需要重新复制-----------------------------#
file_list = get_list("G:\\Datasets\\IXI\\Registration_seed")
for t1,t2,mask in file_list:
    cp_t1 = t1[:]
    cp_t1 = cp_t1.replace('Registration_seed','Registration_validate')
    cp_t2 = t2[:]
    cp_t2 = cp_t2.replace('Registration_seed','Registration_validate')
    cp_mask = mask[:]
    cp_mask = cp_mask.replace('Registration_seed','Registration_validate')
    copyfile(cp_t1,t1)
    print(cp_t1)
    copyfile(cp_t2,t2)
    print(cp_t2)
    copyfile(cp_mask,mask)
    print(cp_mask)

