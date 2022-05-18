import os 
import sys
import platform
import logging
from typing import Callable
from typeguard import typechecked
from subprocess import check_output
import functools
import numpy as np
import nibabel as nib
from alive_progress import alive_bar
from datasets.brats.brats_data import BraTSDataPathCollection,is_affine_euqal,is_header_euqal
from utils.dataset_helper import np_zero_close,read_nii_file,save_nii_file,norm_min_max,norm_z_score,np_min_max_on_sequence

class PreProcess():
    @typechecked
    def __init__(self,path:str) -> None:
        self.data_path_collection = BraTSDataPathCollection(path)
        if platform.system() == 'Linux':
            self.unpreserved_keys = ['brain','brain_mask']
        else:
            self.unpreserved_keys = ['z_score_norm','min_max_norm','z_score_and_min_max_norm','individual_min_max_norm','mask']
    def resample(self):
        raise ValueError("BraTS do not need Resample.")
    def registration(self):
        raise ValueError("BraTS do not need Registration.")
    def get_reshape_perm(self,target,source):
        return (target.index(item) for item in source)   
    def bet(self):
        if platform.system() != 'Linux':
            raise ValueError(
                f"Bet in FSL must run on Linux instead of {platform.system()}."
            )
        input_datas = self.data_path_collection.get_individual_datas('main')
        output_datas = self.data_path_collection.get_individual_datas('brain')
        for inputs,outputs in zip(input_datas,output_datas):
            assert inputs.name==outputs.name
            for (k1,input_path),(k2,output_path) in zip(inputs.datas.items(),outputs.datas.items()):
                assert k1==k2 
                reg_cmd = f"bet {input_path} {output_path} -m -R -f 0.05"
                reg_info = check_output(reg_cmd,shell=True).decode()
                print(reg_cmd)
    @typechecked
    def norm_with_masks(self,foreground_offset:int|float,norm_method:str):
        input_datas = self.data_path_collection.get_individual_datas('main',['mask'])
        if norm_method == 'min_max_norm':
            norm_func =  norm_min_max
            use_global_min_max=True
        elif norm_method == 'individual_min_max_norm':
            norm_func = norm_min_max
            use_global_min_max = False 
        elif norm_method == 'z_score_norm':
            logging.warning(f"z_score 不仅将有效区域进行标准化~N({foreground_offset},1.0) 也将背景归0 有可能会影响模型性能")
            norm_func = norm_z_score
            use_global_min_max = False 
        elif norm_method == 'z_score_and_min_max_norm':
            norm_func = norm_min_max
            input_datas=self.data_path_collection.get_individual_datas('z_score_norm',['mask'])
            use_global_min_max=True
        else: # 其他归一化方法
            raise  ValueError("") #TODO 
        output_datas = self.data_path_collection.get_individual_datas(norm_method,['mask'])

        if use_global_min_max:
            with alive_bar(ctrl_c=False, title='Computing global min_max:') as bar:  
                def _wrapper(func): #bar: draw progress bar 
                    @functools.wraps(func)
                    def wrappered(x1,x2):
                        if isinstance(x1,str):
                            x1,_,_ = read_nii_file(x1)
                        x2,_,_ = read_nii_file(x2)
                        bar()
                        return func(x1,x2)
                    return wrappered
                reduce_min_max = functools.partial(np_min_max_on_sequence,ignore_nan=True,wrapper=_wrapper)
                global_min_maxs = BraTSDataPathCollection.reduce_datas(input_datas,reduce_func=reduce_min_max)
        else:
            def reduce_min_max(_): #bar: draw progress bar 
                return None
            global_min_maxs = BraTSDataPathCollection.reduce_datas(input_datas,reduce_func=reduce_min_max)
            global_min_maxs.pop('mask')
        
        for input_data,output_data in zip(input_datas,output_datas):
            assert input_data.name == output_data.name
            input_data = input_data.datas
            output_data = output_data.datas
            mask1 = input_data.pop('mask')
            mask2 = output_data.pop('mask')
            
            assert  mask1==mask2
            mask_path = mask1
            for (k1,input_path),(k2,output_path),(k3,global_min_max) in zip(input_data.items(),output_data.items(),global_min_maxs.items()):
                assert k1==k2==k3
                img,affine,header = read_nii_file(input_path)
                mask,_,_ = read_nii_file(mask_path,dtype=np.int16)
                norm_out = norm_func(x=img,mask=mask,global_min_max=global_min_max,foreground_offset=foreground_offset)
                save_nii_file(norm_out,output_path,affine=affine,header=header)
                print(output_path)


    def combine_masks(self):
        datas = self.data_path_collection.get_individual_datas('brain_mask',['mask'])
        for data in datas:
            input_paths = list(data.datas.values())
            output_path = data.datas['mask']
            def combine_mask_infos(x,y):
                if isinstance(x,str):
                    x_img,x_affine,x_header = read_nii_file(x,dtype=np.int16)
                    y_img,y_affine,y_header = read_nii_file(y,dtype=np.int16)
                    assert x_img.shape==y_img.shape
                    assert is_affine_euqal(x_affine,y_affine)
                    assert is_header_euqal(x_header,y_header)
                    return np_zero_close(x_img*y_img),x_affine,x_header
                x_img,x_affine,x_header = x[0],x[1],x[2]
                y_img,y_affine,y_header = read_nii_file(y,dtype=np.int16)
                assert x_img.shape==y_img.shape
                assert is_affine_euqal(x_affine,y_affine)
                assert is_header_euqal(x_header,y_header)
                return np_zero_close(x_img*y_img),x_affine,x_header
            mask,affine,header = functools.reduce(combine_mask_infos,input_paths)
            save_nii_file(mask,output_path,affine=affine,header=header)
            print(output_path)
                
    @typechecked
    def del_files(self,keys:str):
        assert keys in self.unpreserved_keys
        if keys.lower()=='mask':
            #shared
            datas = self.data_path_collection.get_individual_datas('main',['mask'])
        else:
            datas = self.data_path_collection.get_individual_datas(keys)
        for data in datas:
            for path in data.datas.values():
                try:
                    # os.remove(path)
                    print(f"remove {path}")
                except FileNotFoundError:
                    print(f"{path} does not exist! Ignored")   
    
  
if __name__=='__main__':
    # pass
    preprocess = PreProcess(path="D:\\Datasets\\BraTS\\BraTS2021_new")
    
    # preprocess.del_files('brain')
    # preprocess.del_files('brain_mask')
    # preprocess.bet()


    # preprocess.del_files('mask')
    # preprocess.combine_masks()
    # preprocess.del_files('z_score_norm')
    # preprocess.del_files('min_max_norm')
    # preprocess.del_files('z_score_and_min_max_norm')
    # preprocess.del_files('individual_min_max_norm')
    preprocess.norm_with_masks(foreground_offset=0.001,norm_method='individual_min_max_norm')
    preprocess.norm_with_masks(foreground_offset=0.001,norm_method='min_max_norm')
    preprocess.norm_with_masks(foreground_offset=0.0,norm_method='z_score_norm')
    preprocess.norm_with_masks(foreground_offset=0.001,norm_method='z_score_and_min_max_norm')
    