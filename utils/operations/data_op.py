import os
import random

import numpy as np 
import nibabel as nib

def read_nii_file(path:str,dtype=np.int32): #np.int32确保足以承载原始数据 对于norm后的数据采用np.float32
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img:nib.Nifti2Image = nib.load(path)
    affine = img.affine
    header = img.header
    img = np.array(img.dataobj[:,:,:],dtype=dtype)
    return img,affine,header

def _sync_nii_header_dtype(img:np.ndarray,header=None):
    if img.dtype == np.int16:
        header['bitpix'] = np.array(16,dtype=header['bitpix'].dtype)
        header['datatype'] = np.array(4,dtype=header['datatype'].dtype)
    elif img.dtype == np.int32:
        header['bitpix'] = np.array(32,dtype=header['bitpix'].dtype)
        header['datatype'] = np.array(8,dtype=header['datatype'].dtype)
    elif img.dtype == np.float32:
        header['bitpix'] = np.array(32,dtype=header['bitpix'].dtype)
        header['datatype'] = np.array(16,dtype=header['datatype'].dtype)
    else:
        raise ValueError(
            f"Unsupported nii data type {img.dtype}. Only support np.int16 np.int32 np.float32. More dtypes will be supported in the future."
        )
    return header

def save_nii_file(img:np.ndarray,path:str,affine=None,header=None):
    header = _sync_nii_header_dtype(img,header)
    img_ii = nib.Nifti1Image(img,affine=affine,header=header)
    nib.save(img_ii,path)

def data_dividing(datas:list,dividing_rates:tuple[float,...],random:random.Random|None=None,selected_all=True)->list[list]:
    assert all(x>=0 for x in dividing_rates)
    assert sum(dividing_rates)<=1.0
    data_range = list(range(len(datas)))
    if random is not None:
        random.shuffle(data_range)
    length = len(data_range)
    end_indexes = []
    for rate in dividing_rates:
        increase_indexes = int(length*rate)
        assert increase_indexes>1
        if not end_indexes:
            end_indexes.append(increase_indexes)
        else:
            end_indexes.append(end_indexes[-1]+increase_indexes)
    if selected_all: # if ture, the last dividing rate will be ignored and the last dividing part will contain all remaining elements
        end_indexes[-1]=length
    else:
        assert end_indexes[-1]<=length
    range_buf = []
    previous_index = 0
    for end_index in end_indexes:
        range_buf.append(sorted(data_range[previous_index:end_index]))
        previous_index = end_index
    return  [[datas[index] for index in ranges] for ranges in range_buf]