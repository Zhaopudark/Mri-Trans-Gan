"""
面向brats数据集
我们认为，一个数据集可以抽象为记录-处理-输出三部分。
其中，处理包括了预处理和后续处理两部分

版本（2021,2020,）
图片
"""

import os
import sys
import nibabel as nib
import numpy as np
import platform
import logging
from subprocess import check_output
import tensorflow as tf
import json
import csv
import time
import datetime
import random
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from utils.image.patch_process import PacthesProcesser
__all__ = [
    "BraTS",
]
class BraTS():
    class _Data():
        """
        # 我们定义的3D卷积输入 应当是BDHWC  
        # 我们定义2D绘图时  纵轴为Y(H)(高度) 横轴为X(W)(宽度) 
        # 既定 BraTS数据集读取为numpy矩阵 [Z,Y,X](基于向量-二维向量-三维向量的stack顺序,最里层为最先定义的轴,因此取最小序号 即X,外层取最大序号 即Z) 
        ## 三轴分别为Z(coronal axis)(冠状轴) Y(sagittal axis)(矢状轴) X(vertical axis)(垂直轴)
        ### Z_min—>Z_max 冠状轴由右R(Right)至左L(Left)
        ### Y_min—>Y_max 矢状轴由前A(Anterior)至后P(Posterior)
        ### X_min—>X_max 垂直轴由下I(Inferior)至上S(Superior)
        ## Z*Y 固定X值 得到 横截面(transverse_plane)(水平面)(horizontal plane)  绘图时得到纵轴(R->L)横轴(A->P) 
        ## Z*X 固定Y值 得到 冠状面(coronal_plane)                              绘图时得到纵轴(R->L)横轴(I->S)
        ## Y*X 固定Z值 得到 矢状面(sagittal_plane)                             绘图时得到纵轴(A->P)横轴(I->S)

        # ITK-SNAP等软件绘图时 遵循如下模式
        ## 横截面 绘图时 纵轴(A->P)横轴(R->L)
        ## 冠状面 绘图时 纵轴(S->I)横轴(R->L)
        ## 矢状面 绘图时 纵轴(S->I)横轴(A->P)
        ### 若numpy矩阵的组织形式为[S:I,A:P,R:L] 则可绘制出与ITK-SNAP一致的3个截面
        ### 因此需要将[Z,Y,X]转置为[-X,Y,Z] 即可实现 "-X"表示反序
        ### 固定X值 [-X,:,:]  得到 横截面 [A:P,R:L] 
        ### 固定Y值 [:,Y,:]   得到 冠状面 [S:I,R:L] 
        ### 固定Z值 [,:,Z]    得到 矢状面 [S:I,A:P] 
        # ITK-SNAP等软件绘图不改变原始数据的组织形式 而这里为了方便用numpy矩阵做出3轴截面的切片 改变了原始数据的组织形式 因此，原始数据"X"的序号对应的横截面不再适用是必然的

            # 我之前的做法1 以BHWDC输入 HWD分别对应ZYX H*W(Z*Y)得到横截面
            # 后来我发现 应当以BDHWC形式输入 符合Tensoflow对3D卷积的定义 深度-高度-宽度
            ## 因此有之前的做法2 将输入转置为 DHW 分别对应XZY H*W(Z*Y)得到横截面
            ## 做法2与做法1在结果上完全一致 甚至 
            ## 将训练好的做法1模型M1内矩阵转置为做法2的形式 即模型M2 同时输入转置I1为做法2的输入I2 
            ## 可以得到evaluate(M1(I1)) = eavluate(M2(I2)) 评估的计算结果(评估方式本身不依赖于数据组织形式 DHW或者HWD)完全不变
        # 现在 为了和ITK-SNAP软件以及诸多论文的结果保持一致 我们选择新的方案
        ## 将BraTS数据集读为numpy矩阵[Z,Y,X]转置为[-X,Y,Z] 三轴分别为X(vertical axis)(垂直轴) Y(sagittal axis)(矢状轴) Z(coronal axis)(冠状轴) 
        ## 固定X值 Y*Z [-X,:,:]  得到 横截面 [A:P,R:L] 
        ## 固定Y值 X*Z[:,Y,:]    得到 冠状面 [S:I,R:L] 
        ## 固定Z值 X*Y [,:,Z]    得到 矢状面 [S:I,A:P] 
        ### 若定义BraTS的原始数据格式为[W,H,-D]即[Z,Y,X]
        ### 则转置后的BraTS数据格式为[D,H,W]即[-X,Y,Z]

        # 但是 如上的定义改变了BraTS原始数据的组织形式 当我研究特定切片时 容易产生混淆
        # 因此 鉴于我考察横截面比较多 对垂直轴的正反序不敏感 所以 
        ## 定义BraTS的原始数据格式为[W,H,D]即[Z,Y,X] 只进行三轴轴序调整 不对垂直轴反序
        ## 转置后的BraTS数据格式为[D,H,W]即[X,Y,Z]
        """
        def __init__(self,path,norm="") -> None:
            self.__modality_names = ["flair.nii.gz","t1.nii.gz","t1ce.nii.gz","t2.nii.gz"] #对内具象
            self.path = self._path_norm(path) # 患者文件夹 对外抽象
            self.name = self._get_dir_name(path)
            self._norm = norm
            self.data_format = "WHD" # coronal_plane sagittal_plane transverse_plane 
            self._modality_path_dict = {} # 记录患者所有模态的据对路径
            for (dir_name,_,file_list) in os.walk(path):
                for file_name in file_list:
                    for modality_name in self.__modality_names:
                        if modality_name in file_name.lower():
                            self._modality_path_dict[modality_name]=os.path.join(dir_name,file_name)
                            break # 一个文件只属于一个模态
            self._get_plans_info()
            self._is_enable()
            self.data_dict = self._modality_path_dict # 对外抽象
        def _get_plans_info(self):
            self._modality_plans_dict  = {}
            for key,value in self._modality_path_dict.items():
                csv_file_path = value[0:-7]+".csv"
                try:
                    with open(csv_file_path) as f:
                        f_csv = csv.reader(f)
                        headers = next(f_csv)
                        row = next(f_csv)
                    self._modality_plans_dict[key] = [int(row[headers.index("transverse_plane")]),
                                                      int(row[headers.index("sagittal_plane")]),
                                                      int(row[headers.index("coronal_plane")])]  # 对外抽象
                except FileNotFoundError:
                    self._modality_plans_dict[key] = [int(1),
                                                      int(1),
                                                      int(1)]     
        def _is_enable(self):
            ######################################### # TODO 可删改的逻辑 存在更加合适的判断 
            counter = 0 
            sum_buf = 0
            for modality,plans in self._modality_plans_dict.items():
                for plan in plans:
                    counter += 1 
                    sum_buf += plan
            if counter==sum_buf:
                self.enabled = True
            else:
                self.enabled = False
            #########################################
        @property
        def target_path_dict(self):#真正用于训练测试的数据路径 可以依据需要手动更改
            norm_suffix = self._norm+"_norm"
            tmp_dict = {}
            for key in self.data_dict.keys():
                tmp_dict[key]= self._get_suffix_file_name(self.data_dict[key],suffix=norm_suffix)
            tmp_dict["mask"] = self.mask_path
            return tmp_dict
        @property
        def z_score_path_dict(self):
            norm_suffix = "z_score"+"_norm"
            tmp_dict = {}
            for key in self.data_dict.keys():
                tmp_dict[key]= self._get_suffix_file_name(self.data_dict[key],suffix=norm_suffix)
            return tmp_dict
        @property
        def z_score_and_min_max_path_dict(self):
            norm_suffix = "z_score_and_min_max"+"_norm"
            tmp_dict = {}
            for key in self.data_dict.keys():
                tmp_dict[key]= self._get_suffix_file_name(self.data_dict[key],suffix=norm_suffix)
            return tmp_dict
        @property
        def norm_path_dict(self):
            norm_suffix = self._norm+"_norm"
            tmp_dict = {}
            for key in self.data_dict.keys():
                tmp_dict[key]= self._get_suffix_file_name(self.data_dict[key],suffix=norm_suffix)
            return tmp_dict
        @property
        def mask_path(self):
            return self.path+"\\"+self.name+"_brain_mask.nii.gz"
        @property
        def mask_path_list(self):
            tmp_list = []
            for key in self.data_dict.keys():
                mask_path = self.data_dict[key]
                tmp_list.append(self._get_suffix_file_name(mask_path,suffix="brain_mask"))
            return tmp_list
        @property
        def brain_path_list(self):
            tmp_list = []
            for key in self.data_dict.keys():
                brain_path = self.data_dict[key]
                tmp_list.append(self._get_suffix_file_name(brain_path,suffix="brain"))
            return tmp_list
        def _get_suffix_file_name(self,file_name,suffix):
            _file_name = str(file_name).strip(".nii.gz")
            _file_name = _file_name+"_"+suffix+".nii.gz"
            return _file_name
        def _path_norm(self,path):
            _path = str(path).replace("/","\\")
            if _path[-1]=="\\":
                _path = _path[0:-1]
            return _path
        def _get_dir_name(self,path):
            """
            从路径中抽取最后一级目录的名字
            """
            _path = str(path).replace("/","\\")
            if _path[-1]=="\\":
                _path = _path[0:-1]
            for index in range(len(_path)-1,-1,-1):
                if _path[index]=="\\":
                    break
            return _path[index+1::]
        def is_target_path(self,path):
            # 暂时不使用该方法 因为即便判断不是brats路径,也欠缺可以自动操作的办法 非目标路径自然会在其余的处理中显现
            return False
    class _PreProcess():
        def __init__(self,norm="") -> None:
            self._norm = norm
        def _np_zero_close(self,x):
             _where = np.isclose(x,0.0,rtol=1e-05,atol=1e-08,equal_nan=False)# 与0比较 其余取默认值(默认nan与nan不相 等返回false,nan与非nan不相等,返回false)
             x[_where]=0
             return np.array(x)#返回新数组避免出问题
        def _np_div_no_nan(self,a,b):
            """构建基于numpy的div_no_nan
            input:被除数a,除数b
            output:a/b, Nan值归0
            """
            _where = np.where(b!=0,True,False) # b!=0 直接写b!=0不符合程序的整体的写法
            img = np.divide(a,b,out=np.zeros_like(a), where=_where) 
            return img
        def _valid_min_max(self,x,mask):
            """
            获取 0 1 mask标记的有效区域的最小值
            要求mask必须为 0 1 mask               
            """
            x_max = x.max()
            out_min = self._np_zero_close((x-x_max)*mask).min()+x_max
            x_min = x.min()
            out_max = self._np_zero_close((x+x_min)*mask).max()-x_min
            return out_min,out_max
        def resample(self):
            raise ValueError("BraTS do not need Resample.")
        def registration(self):
            raise ValueError("BraTS do not need Registration.")
        def bet(self,datas):
            if platform.system() == "Linux":
                for data in datas:
                    for key,value in data.data_dict.items():
                            file_path = value
                            input_path = file_path
                            output_path = file_path[0:-7]+"_brain.nii.gz"
                            if os.path.exists(input_path):
                                reg_cmd = r"bet {} {} -m -R -f 0.05".format(input_path,output_path)
                                reg_info = check_output(reg_cmd,shell=True).decode()
                                print(reg_cmd)
            else:
                raise ValueError("Bet in FSL must run on Linux instead of {}.".format(platform.system()))
        def get_random(self,random_num,source,direct):
            """
            在一个目录中抽取随机的若干样本到新目录
            """
            logging.info("在一个目录中抽取(复制)随机的若干样本到新目录的方法占用空间大,弃用,将全面采用logs的方式", DeprecationWarning)
            t1_buf = []
            for (dir_name,_, file_list) in os.walk(source):
                for file_name in file_list:
                    if "t1.nii.gz" in file_name.lower(): 
                        t1_buf.append(os.path.join(dir_name,file_name))
            files_buf = []
            for i,z1 in enumerate(t1_buf):
                for index in range(len(z1)-1,-1,-1):
                    if z1[index]=="\\":
                        break
                files_buf.append(z1[0:index])   
            out_buf = random.sample(files_buf,random_num)
            for item in out_buf:
                # print(item)
                # str1 = "Copy-Item {} {}  -recurse -force".format(item,direct)
                # print(item,direct)
                for index in range(len(item)-1,-1,-1):
                    if item[index]=="\\":
                        break 
                os.system("xcopy {} {} /E".format(item,direct+item[index+1::]+"\\"))
        def _redomain(self,img,domain=[0.0,1.0]):
            """
            将任意值域的输入均匀得映射到目标值域
            """
            _min = img.min()
            _max = img.max()
            img = self._np_div_no_nan(img-_min,_max-_min)
            domain_min = min(domain)
            domain_max = max(domain)
            img = img*(domain_max-domain_min)+domain_min
            img = self._np_zero_close(img)
        def _norm_min_max(self,img,global_min,global_max,mask,foreground_offset=0.0):
            """
            将输入以min_max归一化到0-1domain
            img:input
            mask:区分前景背景的 0 1mask
            foreground_offset: 前景偏移量。
            img = Vaild_Value*(1-foreground_offset)+foreground_offset
            """
            img = img.astype(np.float32)
            if mask is not None:
                mask = mask.astype(np.float32)
                _min,_max = global_min,global_max
                img = self._np_div_no_nan(img-_min,_max-_min)
                img = img*(1.0-foreground_offset)+foreground_offset
                img = img*mask # 以上操作对mask标注的有效区域有效,无效区域的数值保证计算安全(无inf,无nan即可)进行一次*mask回归正常
                img = self._np_zero_close(img)
                return img 
            else:
                _min,_max = global_min,global_max
                img = self._np_div_no_nan(img-_min,_max-_min)
                return img 
        def _norm_z_score(self,img,mask=None,foreground_offset=0.0):#
            """
            将输入归一化到 均值为0 方差为1
            img:input
            mask:区分前景背景的 0 1mask
                当mask存在时,计算的是有效区域的均值标准差,将img的有效区域归一化到0均值 将背景归0
                img = (img-mean)/std
            foreground_offset: 前景偏移量 在z_score中,是有效区域整体(等价于有效区域均值)的偏移量 可以用于探究在z_score将背景也归到0或者其他值是否合理
            img = Vaild_Value+foreground_offset
            """
            img = img.astype(np.float32)
            if mask is not None:
                mask = mask.astype(np.float32)
                _where = np.where(mask>0.5,True,False)
                mean = np.mean(img,where=_where)
                std = np.std(img,ddof=0,where=_where) #将一个图像的所有体素视为总体 求的是这个总体的标准差
                img = self._np_div_no_nan(img-mean,std)
                img = img+foreground_offset
                img = img*mask # 以上操作对mask标注的有效区域有效,无效区域的数值保证计算安全(无inf,无nan即可)进行一次*mask回归正常
                img = self._np_zero_close(img)
                return img 
            else:
                mean = np.mean(img)
                std = np.std(img,ddof=0) #将一个图像的所有体素视为总体 求的是这个总体的标准差
                img = self._np_div_no_nan(img-mean,std)
                return img 
        def norm_with_masks(self,datas,foreground_offset):
            if self._norm == "min_max":
                out_min_max_dict = {}
                for i,data in enumerate(datas):
                    for key,value in data.data_dict.items():
                        img = nib.load(value)
                        img = np.array(img.dataobj[:,:,:])
                        _min = float(img.min())
                        _max = float(img.max())
                        if key not in out_min_max_dict:
                            out_min_max_dict[key] = [_min,_max]
                        else:
                            if _min < out_min_max_dict[key][0]:
                                out_min_max_dict[key][0] = _min
                            if _max > out_min_max_dict[key][1]:
                                out_min_max_dict[key][1] = _max
                    print(i,out_min_max_dict)
                for data in datas:
                    mask,_,_ = self._read_nii_file(data.mask_path,dtype=np.int16)
                    for key in data.data_dict.keys():
                        img,affine,header = self._read_nii_file(data.data_dict[key])
                        global_min,global_max = out_min_max_dict[key]
                        norm_out = self._norm_min_max(img=img,global_min=global_min,global_max=global_max,mask=mask,foreground_offset=foreground_offset)
                        norm_out_path = data.norm_path_dict[key]
                        self._save_nii_file(norm_out,norm_out_path,affine=affine,header=header)
                        print(norm_out_path)
            elif self._norm == "z_score":
                logging.warning("z_score 不仅将有效区域进行标准化~N({},1.0) 也将背景归0 有可能会影响模型性能".format(foreground_offset))
                for data in datas:
                    mask,_,_ = self._read_nii_file(data.mask_path,dtype=np.int16)
                    for key in data.data_dict.keys():
                        img,affine,header = self._read_nii_file(data.data_dict[key])
                        norm_out = self._norm_z_score(img=img,mask=mask,foreground_offset=foreground_offset)
                        norm_out_path = data.norm_path_dict[key]
                        self._save_nii_file(norm_out,norm_out_path,affine=affine,header=header)
                        print(norm_out_path)
            if self._norm == "z_score_and_min_max":
                out_min_max_dict = {}
                for i,data in enumerate(datas):
                    for key,value in data.z_score_path_dict.items():
                        img = nib.load(value)
                        img = np.array(img.dataobj[:,:,:])
                        _min = float(img.min())
                        _max = float(img.max())
                        if key not in out_min_max_dict:
                            out_min_max_dict[key] = [_min,_max]
                        else:
                            if _min < out_min_max_dict[key][0]:
                                out_min_max_dict[key][0] = _min
                            if _max > out_min_max_dict[key][1]:
                                out_min_max_dict[key][1] = _max
                    print(i,out_min_max_dict)
                for data in datas:
                    mask,_,_ = self._read_nii_file(data.mask_path,dtype=np.int16)
                    for key in data.z_score_path_dict.keys():
                        img,affine,header = self._read_nii_file(data.z_score_path_dict[key])
                        global_min,global_max = out_min_max_dict[key]
                        norm_out = self._norm_min_max(img=img,global_min=global_min,global_max=global_max,mask=mask,foreground_offset=foreground_offset)
                        norm_out_path = data.z_score_and_min_max_path_dict[key]
                        self._save_nii_file(norm_out,norm_out_path,affine=affine,header=header)
                        print(norm_out_path)
            else: # 其他归一化方法
                raise  ValueError("") #TODO 
        def _read_nii_file(self,path,dtype=np.int32): #np.int32确保足以承载原始数据 对于norm后的数据采用np.float32
            if os.path.exists(path):
                img = nib.load(path)
                affine = img.affine
                header = img.header
                img = np.array(img.dataobj[:,:,:],dtype=dtype)
                return img,affine,header
            else:
                raise FileNotFoundError
        def _sync_nii_header_dtype(self,img,header):
            if img.dtype == np.int16:
                header["bitpix"] = np.array(16,dtype=header["bitpix"].dtype)
                header["datatype"] = np.array(4,dtype=header["datatype"].dtype)
            elif img.dtype == np.int32:
                header["bitpix"] = np.array(32,dtype=header["bitpix"].dtype)
                header["datatype"] = np.array(8,dtype=header["datatype"].dtype)
            elif img.dtype == np.float32:
                header["bitpix"] = np.array(32,dtype=header["bitpix"].dtype)
                header["datatype"] = np.array(16,dtype=header["datatype"].dtype)
            else:
                raise ValueError("Unsupported nii data type {}. Only support np.int16 np.int32 np.float32. More dtypes will be supported in the future.".format(img.dtype))
            return header
        def _save_nii_file(self,img,path,affine,header):
            header = self._sync_nii_header_dtype(img,header)
            img_ii = nib.Nifti1Image(img,affine=affine,header=header)
            nib.save(img_ii,path)
        def _combine_affine_list(self,affine_list):
            # 联合所有的affines 要求affines的内容必须完全一致
            for i in range(1,len(affine_list),1):
                if (affine_list[i]==affine_list[i-1]).all():
                    continue
                else:
                    raise ValueError("Affines in affine_list are not the same.")
            return affine_list[0]
        def _combine_header_list(self,header_list):
            # 联合所有的headers 要求headers的内容必须完全一致
            for i in range(1,len(header_list),1):
                for (key0,value0),(key1,value1) in zip(header_list[i-1].items(),header_list[i].items()):
                    if key0==key1:
                        dtype_list = [np.uint8,np.uint16,np.uint32,np.int8,np.int16,np.int32,np.float32]
                        if (value0.dtype in dtype_list)and(value1.dtype in dtype_list):
                            if (np.isclose(value0,value1,equal_nan=True)).all():
                                continue
                        else:
                            if (value0==value1).all():
                                continue
                    else:
                        raise ValueError("headers in affine_list are not the same i {} {} but {}.".format(key0,value0,value1))
            return header_list[0]
        def combine_masks(self,datas):
            def __read_wrapper(fuc):
                def read(path):
                    return fuc(path,dtype=np.int16)
                return read
            read_fuc = __read_wrapper(self._read_nii_file)
            for data in datas:
                try:
                    mask_info_list = list(map(read_fuc,data.mask_path_list))
                    mask_list,mask_affine_list,mask_header_list = zip(*mask_info_list)
                    mask = self._combine_mask_list(mask_list)
                    affine = self._combine_affine_list(mask_affine_list)
                    header = self._combine_header_list(mask_header_list)
                    mask_save_path = data.mask_path
                    self._save_nii_file(mask,mask_save_path,affine=affine,header=header)
                    print(mask_save_path)
                except FileNotFoundError:
                    print("The masks in '{}' needn't be combined!".format(data.path))
                    continue
        def del_redundant_masks(self,datas):
            for data in datas:
                for path in data.mask_path_list:
                    try:
                        os.remove(path) 
                    except FileNotFoundError:
                        print("{} does not exist! Ignored".format(path))
                for path in data.brain_path_list:
                    try:
                        os.remove(path) 
                    except FileNotFoundError:
                        print("{} does not exist! Ignored".format(path))
                print(data.path)  
        def del_norm_files(self,datas):
            for data in datas:
                for key,path in data.norm_path_dict.items():
                    try:
                        os.remove(path) 
                    except FileNotFoundError:
                        print("{} does not exist! Ignored".format(path))
                print(data.path)      
        def _combine_mask_list(self,masks_list):
            """
            将若干mask进行合并(哈达玛积)
            """
            if isinstance(masks_list,list) or isinstance(masks_list,tuple):
                if len(masks_list)==1:
                    mask_shape = masks_list[0].shape
                    return masks_list[0]
                elif len(masks_list)>=2:
                    mask_buf = masks_list[0]
                    mask_shape = mask_buf.shape
                    for mask in masks_list[1::]:
                        if mask.shape == mask_shape:
                            mask_buf = mask_buf*mask
                            mask_buf = self._np_zero_close(mask_buf) #减少一些乘0造成的误差
                        else:    
                            raise ValueError("masks in masks_list should have the same shape!")
                    return mask_buf
                else:
                    raise ValueError("masks_list is []!")
            else:
                raise ValueError("masks_list should be a list or tuple!")
    class _Dividing():
        def __init__(self,datas,random_seed=None,redivide_dict={}) -> None: 
            """
            redivide_dict是最后的强制手段 基本用不到 
            数据集的划分规则为:
            原始数据-(原始数据内部处理enable)-净化后的数据-(默认按照BraTS官方划分训练测试集,其余按既定规则随机)-
                                                     -(或对净化后的数据全部随机抽取训练与测试集等,统一全局的随机种子)         
            """
            self.datas = datas
            self.random_seed = random_seed
            if redivide_dict=={}:
                self.active_dict = self.get_default_active_dict(datas)
            else:
                self.active_dict = redivide_dict

        def get_default_active_dict(self,datas):
            _active_datas = [] 
            for item in datas:
                if item.enabled:
                    _active_datas.append(item)
            active_dict = {}
            buf = []
            for item in _active_datas:
                if "TrainingData" in item.path:
                    buf.append(item)
            active_dict["train"] = buf 
            buf = []
            for item in _active_datas:
                if "ValidationData" in item.path:
                    buf.append(item)
            active_dict["test"] = buf 
            if self.random_seed is not None:
                random.seed(int(self.random_seed))
            else:
                random.seed(25)
            _buf = random.sample(buf, 25)
            active_dict["validate"] = _buf 
            if self.random_seed is not None:
                random.seed(int(self.random_seed))
            else:
                random.seed(0)
            _buf = random.sample(buf, 1)
            active_dict["draw"] = _buf 
            return active_dict
        @property
        def datas_dict(self):
            return self.active_dict
        # def __init__(self,path,datas,redivide_dict={}) -> None: # TODO 数据集应当以 random_seed划分 
        #     for i,item in enumerate(datas):
        #         item.static_index=i
        #     self.datas = datas
        #     self.logs_path = self.get_logs_path(path)
        #     self.current_logs = {}
        #     self._logs_state = self._read_logs()
        #     if self._logs_state:
        #         if redivide_dict=={}:
        #             active_dict = self.get_default_active_dict(datas)
        #             append_logs = self._make_single_logs(active_dict)
        #             self.current_logs = self._make_logs(logs=self.current_logs,single_logs=append_logs)
        #             self._save_logs(self.current_logs)
        #         else:
        #             append_logs = self._make_single_logs(redivide_dict) 
        #             """
        #             TODO 为原始数据结构标上等级 依据标的等级统一管理数据 再抽取得到数据集
        #             原始数据-> 可行的数据 -> 整体划分出训练-测试集等
        #             """
        #             self.current_logs = self._make_logs(logs=self.current_logs,single_logs=append_logs)
        #             self._save_logs(self.current_logs)
        #     else:
        #         active_dict = self.get_default_active_dict(datas)
        #         append_logs = self._make_single_logs(active_dict)
        #         self.current_logs = self._make_logs(logs={},single_logs=append_logs) # logs先是空 然后和新logs合并
        #         self._save_logs(self.current_logs)
        #         self._logs_state = True 
        #     self._read_nii_file = BraTS._PreProcess._read_nii_file
        #     self.patch_processer = PacthesProcesser()
        # def get_logs_path(self,path):
        #     if path[-1]=="/":
        #         _path = path[0:-1]
        #     elif path[-1]=="\\":
        #         _path = path[0:-1]
        #     else:
        #         _path = path[:]
        #     _path = _path+"\\"+"logs.json"
        #     return _path
        # def get_default_active_dict(self,datas):
        #     _active_datas = [] 
        #     for item in datas:
        #         if item.enabled:
        #             _active_datas.append(item)
        #     active_dict = {}
        #     buf = []
        #     for item in _active_datas:
        #         if "TrainingData" in item.path:
        #             buf.append(item.static_index)
        #     active_dict["train"] = buf 
        #     buf = []
        #     for item in _active_datas:
        #         if "ValidationData" in item.path:
        #             buf.append(item.static_index)
        #     active_dict["test"] = buf 
        #     random.seed(25)
        #     _buf = random.sample(buf, 25)
        #     _buf = sorted(_buf)
        #     active_dict["validate"] = _buf 
        #     random.seed(0)
        #     _buf = random.sample(buf, 1)
        #     _buf = sorted(_buf)
        #     active_dict["draw"] = _buf 
        #     return active_dict
        # def _make_logs(self,logs,single_logs):
        #     tmp_logs = logs
        #     for item in single_logs.keys():
        #         tmp_logs[item]=single_logs[item]
        #     tmp_list = list(tmp_logs.items())
        #     tmp_list = sorted(tmp_list,key=lambda x:x[0])
        #     flow_index = 0
        #     while flow_index < len(tmp_list)-1:
        #         key_0,value_0 = tmp_list[flow_index]
        #         key_1,value_1 = tmp_list[flow_index+1]
        #         if value_0==value_1:
        #             tmp_list.pop(flow_index+1)
        #     tmp_logs = {}
        #     for key,value in tmp_list:
        #         tmp_logs[key] = value   
        #     return tmp_logs
        # def _make_single_logs(self,active_dict):
        #     tmp_dict = {}
        #     current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #     tmp_dict[current_datetime] = active_dict
        #     return tmp_dict
        # def _save_logs(self,logs_dict):
        #     with open(self.logs_path,'w') as f:
        #         json.dump(logs_dict,f) 
        # def _read_logs(self):
        #     try:
        #         with open(self.logs_path,'r') as f:
        #             self.current_logs = json.load(f) 
        #         return True
        #     except (FileNotFoundError,json.decoder.JSONDecodeError):
        #         return False
        # @property
        # def datas_dict(self):
        #     key = list(self.current_logs.keys())[-1]
        #     active_dict = self.current_logs[key]
        #     tmp_dict = {}
        #     for key,value in active_dict.items():
        #         buf = [self.datas[index] for index in value]
        #         tmp_dict[key]=buf
        #     return tmp_dict
    class _DataPipeLine():
        def __init__(self,datas,data_format,cut_ranges,patch_size,patch_nums,random_seed=None,name=None):
            """
            datas:接受 _Data()的实例列表 data in datas中 data的target_path_dict 存放我们希望的数据路径(一般为真正处理好的数据)
            """
            self.datas = datas
            #-----------------------------#
            self.data_format = data_format
            self.cut_ranges =  self._data_format_apply(data_format,cut_ranges)
            self.patch_size = self._data_format_apply(data_format,patch_size)
            self.patch_nums = self._data_format_apply(data_format,patch_nums)
            #-----------------------------#
            self.random_seed = random_seed
            self.patch_processer = PacthesProcesser()
            self.patch_divider = self.patch_processer.GetPatches(cut_ranges=self.cut_ranges,patch_size=self.patch_size,patch_nums=self.patch_nums,random_seed=random_seed)
            self.name = name
            def __patch_combine_wrapper(fuc,x):
                if isinstance(x,list):
                    mul_buf = 1
                    for item in x:
                        mul_buf *= item
                else:
                    mul_buf = int(x)
                def patch_combiner(patch_inputs):
                    return fuc(generator_p_m_v=patch_inputs,total_patch_nums=mul_buf)
                return patch_combiner
            self.patch_combiner = __patch_combine_wrapper(self.patch_processer.combine_n_patches_x,self.patch_nums)
            def __read_wrapper(cls,fuc,data_format):
                def read(path):
                    img,_,_ =fuc(cls,path,dtype=np.float32)
                    if data_format == "DHW":
                        transpose_vertor = [2,1,0]
                        # print("The out put data_format is {}, which is different from original data_format WHD.".format(data_format))
                    elif data_format == "WHD":
                        transpose_vertor = [0,1,2]
                        # print("The out put data_format is {}, which is the same as original data_format WHD.".format(data_format))
                    else:
                        raise ValueError("Unsupported data_format:{}.".format(data_format))
                    img = tf.transpose(img,transpose_vertor)
                    return img
                return read
            self.read_nii_file = __read_wrapper(BraTS._PreProcess,BraTS._PreProcess._read_nii_file,data_format)
            self._output_shape = self.patch_divider.output_shape
            self._output_dtype = self.patch_divider.output_dtype
        def _data_format_apply(self,data_format,input_list):
            if data_format == "DHW":
                transpose_vertor = [2,1,0]
            elif data_format == "WHD":
                transpose_vertor = [0,1,2]
            else:
                raise ValueError("Unsupported data_format:{}.".format(data_format))
            out = [input_list[index] for index in transpose_vertor]
            return out
        @property
        def output_shape(self):#考虑到会依据data中的文件个数进行堆叠
            extend_dim = len(self.datas[0].target_path_dict.items())
            buf = []
            for item in self._output_shape:
                _item = list(item)
                _item = [extend_dim]+_item
                _item = tuple(_item)
                buf.append(_item)
            return tuple(buf)
        @property
        def output_dtype(self):#考虑到会依据data中的文件个数进行堆叠 不影响dtype
            return self._output_dtype
        def _read_single_group(self,data):#读取一个患者文件夹下是若干模态文件并返回
            # keys,values = zip(*sorted(list(data.target_path_dict.items()),key=lambda x:x[0])) #排序打乱了既定的顺序 不该排序
            keys,values = zip(*list(data.target_path_dict.items()))
            img_list = list(map(self.read_nii_file,values))
            return img_list
        def _gen_patches(self,img_list): # 按照指定的patch_size和策略抽取patches
            gens = list(map(self.patch_divider.get_center_patches,img_list))
            for zipped in zip(*gens):
                buf_list = []
                for items in zipped:
                    if len(buf_list)==0:
                        for index,item in enumerate(items):
                            buf_list.append([item])
                    else:
                        for index,item in enumerate(items):
                            buf_list[index].append(item)
                buf_list = list(map(np.stack,buf_list))
                yield buf_list
        def generator(self):
            """
            作为数据管道被tf.data.Dataset.from_generator 则输出的打包方式决定了tf.data.Dataset.from_generator的解包方式
            切不可出现
            for xxx in patches_across_img_list:
                yield xxx 打包
            而 tf.data.Dataset.from_generator则以
                for a,b,c in data:
                    ...
                解包 
            在大部分情况下 a,b,c解包的原因时他们属于不同类型 不同大小的数据 本身没法合并为一个Tensor或者ragged Tensor 因此只能在拆包的情况下传递和分析
            因此 需要保证全局的拆包解包一致 因此设计时 尽量减少pipeline的改动 争取一步到位 给出足够多的信息 面对不同研究时 在模型端改动
            """
            if self.random_seed is not None:
                random.seed(int(self.random_seed))
                random.shuffle(self.datas)
                print("{} pipeline rerandom complete!".format(self.name))
            for _,data in enumerate(self.datas):
                img_list = self._read_single_group(data)
                patches_across_img_list = self._gen_patches(img_list)
                for imgs,img_masks,padding_vectors in patches_across_img_list: #此处拆包逻辑与_gen_patches绑定且一致(不一致不影响python程序但是影响tf.data.Dataset.from_generator的拆解包)
                    yield imgs,img_masks,padding_vectors
        def add_channel_before_batch(self,imgs,img_masks,padding_vectors):
            # 作为类方法被 tf.data.Dataset.from_generator 调用 作为其支持的 map function 
            imgs = tf.reshape(imgs,shape = imgs.shape[::]+[1])
            img_masks = tf.reshape(img_masks,shape = img_masks.shape[::]+[1])
            tmp = tf.constant([0,0],dtype=padding_vectors.dtype)
            target_shape = padding_vectors.shape[0:-2]+[1]+padding_vectors.shape[-1]
            tmp = tf.broadcast_to(tmp,target_shape)
            padding_vectors = tf.concat([tmp,padding_vectors[:,:],tmp], axis=-2) #提前为batch 堆叠添加batch维度
            return imgs,img_masks,padding_vectors
    def __init__(self,path,data_format,norm,cut_ranges,patch_size,patch_nums,random_seed=None) -> None:
        self.files_dir_list = self._get_dir_list(path=path)
        def __init_wrapper(cls,**kwargs):
            def wappered_cls(arg):
                return cls(arg,**kwargs)
            return wappered_cls
        _Data = __init_wrapper(self._Data,norm=norm)
        self.datas = list(map(_Data,self.files_dir_list))
        self.datas = sorted(self.datas,key=lambda x:x.name)
        self.pre_process = self._PreProcess(norm=norm)
        self.dividing = self._Dividing(datas=self.datas,random_seed=None)
        self.diveded_dict = self.dividing.datas_dict # 依据划分的数据集生成若干生成器 [gen1,gen2,...,]
        if isinstance(random_seed,list) or isinstance(random_seed,tuple):
            pass
        else:
            random_seeds = [random_seed for _ in range(len(self.diveded_dict.items()))]
        buf = []
        for (diveded_name,diveded_data),_random_seed in zip(self.diveded_dict.items(),random_seeds):
            if diveded_name.lower() == "train":
                pass
            else:
                _random_seed = None
            _pipline = self._DataPipeLine(datas=diveded_data,data_format=data_format,cut_ranges=cut_ranges,patch_size=patch_size,patch_nums=patch_nums,random_seed=_random_seed,name=diveded_name)
            buf.append(_pipline.generator)
        self.piplines = buf
        self.pipline_output_shape = _pipline.output_shape
        self.pipline_output_dtype = _pipline.output_dtype
        self.add_channel_before_batch = _pipline.add_channel_before_batch
        self.patch_combiner = _pipline.patch_combiner
    def _get_dir_list(self,path):
        files_dirs = set()
        for (dir_name,_,file_list) in os.walk(path):
            for file_name in file_list:
                if ".nii.gz" in file_name.lower():  
                    files_dirs.add(dir_name)
        files_dir_list = list(sorted(files_dirs))
        return files_dir_list

if __name__=="__main__":
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0],True)
    # b = BraTS(path="G:\\Datasets\\BraTS\\BraTS2021_new",norm="min_max",
    #             cut_ranges=[[0,239],[0,239],[155//2-8,155//2+7]],
    #                  patch_size=[128,128,5],
    #                  patch_nums=[1,1,1],random_seed=1000)
    b = BraTS(path="G:\\Datasets\\BraTS\\BraTS2021_new",norm="z_score_and_min_max",
                     data_format="DHW",
                     cut_ranges=[[0,239],[0,239],[155//2-8,155//2+7]],
                     patch_size=[128,128,5],
                     patch_nums=[1,1,1],random_seed=1000)
    b2 = BraTS(path="G:\\Datasets\\BraTS\\BraTS2021_new",norm="z_score_and_min_max",
                     data_format="DHW",
                     cut_ranges=[[0,239],[0,239],[155//2-8,155//2+7]],
                     patch_size=[128,128,5],
                     patch_nums=[1,1,1],random_seed=1000)
    # out_min_max_dict = {}
    # for i,data in enumerate(b.datas):
    #     # for key,value in data.z_score_and_min_max_path_dict.items():
    #     for key,value in data.z_score_path_dict.items():
    #         img = nib.load(value)
    #         img = np.array(img.dataobj[:,:,:])
    #         _min = float(img.min())
    #         _max = float(img.max())
    #         if key not in out_min_max_dict:
    #             out_min_max_dict[key] = [_min,_max]
    #         else:
    #             if _min < out_min_max_dict[key][0]:
    #                 out_min_max_dict[key][0] = _min
    #             if _max > out_min_max_dict[key][1]:
    #                 out_min_max_dict[key][1] = _max
    #     print(i,out_min_max_dict)
    

    # b.pre_process.combine_masks(b.datas)
    # b.pre_process.del_redundant_masks(b.datas)
    # b.pre_process.norm_with_masks(b.datas,foreground_offset=0.001)
    # b.pre_process.norm_with_masks(b.datas,foreground_offset=0.0)
    ## b.pre_process.del_norm_files(b.datas)
    # print(b.dividing.datas_dict)
    # i_buf = []
    # for i,((a1,a2,a3),(a11,a12,a13)) in enumerate(zip(b.piplines[2](),b2.piplines[2]())):
        # print(i,tf.reduce_mean(a1-a11),a1.shape,tf.reduce_mean(a2-a12),tf.reduce_mean(a3-a13),tf.reduce_mean(a1))
        # break
        # print(i,a1.shape,a2.shape,a3.shape,a1.dtype,a2.dtype,a3.dtype,b.pipline_output_shape,b.pipline_output_dtype)
        # break
    # i_buf.append(i)

    # for i,((a1,a2,a3),(a11,a12,a13)) in enumerate(zip(b.piplines[2](),b2.piplines[2]())):
    #     print(i,tf.reduce_mean(a1-a11),a1.shape,tf.reduce_mean(a2-a12),tf.reduce_mean(a3-a13),tf.reduce_mean(a1))
        # break
        # print(i,a1.shape,a2.shape,a3.shape,a1.dtype,a2.dtype,a3.dtype,b.pipline_output_shape,b.pipline_output_dtype)
        # break
    # i_buf.append(i)

    # for i,(a1,a2,a3) in enumerate(b.piplines[1]()):
    #     print(i,a1.shape,a2.shape,a3.shape,a1.dtype,a2.dtype,a3.dtype,b.pipline_output_shape,b.pipline_output_dtype)
    # i_buf.append(i)
    # for i,(a1,a2,a3) in enumerate(b.piplines[2]()):
    #     print(i,a1.shape,a2.shape,a3.shape,a1.dtype,a2.dtype,a3.dtype,b.pipline_output_shape,b.pipline_output_dtype)
    # i_buf.append(i)
    # for i,(a1,a2,a3) in enumerate(b.piplines[3]()):
    #     print(i,a1.shape,a2.shape,a3.shape,a1.dtype,a2.dtype,a3.dtype,b.pipline_output_shape,b.pipline_output_dtype)
    # i_buf.append(i)
    # print(i_buf)
    # dataset = tf.data.Dataset.from_generator(b.piplines[2],output_types=b.pipline_output_dtype,output_shapes=b.pipline_output_shape)
    # from matplotlib import pyplot as plt
    # for imgs,img_masks,padding_vectors in dataset:
    #     print(imgs.shape,img_masks.shape,padding_vectors.shape)
    #     print(imgs.dtype,img_masks.dtype,padding_vectors.dtype)
    #     print(imgs.numpy().min(),img_masks.numpy().min(),padding_vectors.numpy().min())
    #     print(imgs.numpy().max(),img_masks.numpy().max(),padding_vectors.numpy().max())
    #     fig = plt.figure()
    #     plt.imshow(imgs[0,:,:,0],cmap="gray")
    #     plt.show()
    #     plt.imshow(imgs[0,0,:,:],cmap="gray")
    #     plt.show()
    #     fig = plt.figure()
    #     plt.imshow(imgs[1,:,:,0],cmap="gray")
    #     plt.show()
    #     fig = plt.figure()
    #     plt.imshow(imgs[2,:,:,0],cmap="gray")
    #     plt.show()
    #     fig = plt.figure()
    #     plt.imshow(imgs[3,:,:,0],cmap="gray")
    #     plt.show()
    #     fig = plt.figure()
    #     plt.imshow(imgs[4,:,:,0],cmap="gray")
    #     plt.show()
    #     plt.close()
    # path = "G:\Datasets\BraTS\BraTS2021_new\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\BraTS2021_00000\BraTS2021_00000_flair.nii.gz"
    # img = nib.load(path)
    # img = np.array(img.dataobj[:,:,:])
    # print(img.shape)
    # fig = plt.figure()
    # plt.subplot(3,2,1)
    # plt.imshow(img[:,:,77],cmap="gray")
    # plt.subplot(3,2,2)
    # plt.imshow(img[:,:,38],cmap="gray")
    # plt.subplot(3,2,3)
    # plt.imshow(img[:,120,:],cmap="gray")
    # plt.subplot(3,2,4)
    # plt.imshow(img[:,80,:],cmap="gray")
    # plt.subplot(3,2,5)
    # plt.imshow(img[120,:,:],cmap="gray")
    # plt.subplot(3,2,6)
    # plt.imshow(img[80,:,:],cmap="gray")
    # plt.show()
    # plt.close()

    # cp_buf=[]
    # for data in b.datas:
    #     for key,value in data.data_dict.items():
    #         path = value
    #     #     print(value)
    #     # path = "G:\Datasets\BraTS\BraTS2021_new\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\BraTS2021_00000\BraTS2021_00000_flair.nii.gz"
    #         img = nib.load(path)
    #         if len(cp_buf)==0:
    #             cp_buf.append(img.affine)
    #             cp_buf.append(img.header["dim_info"])
    #             cp_buf.append(img.header["dim"])
    #             cp_buf.append(img.header["pixdim"])
    #             cp_buf.append(img.header["quatern_b"])
    #             cp_buf.append(img.header["quatern_c"])
    #             cp_buf.append(img.header["quatern_d"])
    #             cp_buf.append(img.header["qoffset_y"])
    #             for key,items in img.header.items():
    #                 print(key)
    #                 print(type(items))
    #                 print(items.dtype)
    #             # print(type(img.header.keys()))
    #             # print(img.extra)
    #             # print(img.file_map)
    #             continue
    #         else:
    #             buf=[]
    #             buf.append(img.affine)
    #             buf.append(img.header["dim_info"])
    #             buf.append(img.header["dim"])
    #             buf.append(img.header["pixdim"])
    #             buf.append(img.header["quatern_b"])
    #             buf.append(img.header["quatern_c"])
    #             buf.append(img.header["quatern_d"])
    #             buf.append(img.header["qoffset_y"])
                
                
    #             # print(type(img.header["dim_info"]))
    #         out_buf = []
    #         for index in range(len(cp_buf)):
    #             if isinstance(cp_buf[index],np.ndarray):
    #                 out = (cp_buf[index]==buf[index]).all()
    #             else:
    #                 out = cp_buf[index]==buf[index]
    #             out_buf.append(out)
    #         if all(out_buf):
    #             # print(out_buf)
    #             pass
    #         else:
    #             print(out_buf,path)
    #             pass
            # break
        # break
    
    ggg = tf.random.Generator.from_seed(1314)
    gg = tf.random.Generator.from_seed(1314)
    gggg = tf.random.Generator.from_seed(1314)
    g = tf.random.Generator.from_seed(1000)
    e = ggg.uniform(shape=(1,),minval=0.0,maxval=1.0)
    e1 = gg.uniform(shape=(1,),minval=0.0,maxval=1.0)
    e2 = gggg.uniform(shape=(1,),minval=0.0,maxval=1.0)
    e3 = g.uniform(shape=(1,),minval=0.0,maxval=1.0)
    print(e.numpy(),e1.numpy(),e2.numpy(),e3.numpy())
    ggg.reset_from_seed(1314)
    gg.reset_from_seed(1000)
    e = ggg.uniform(shape=(1,),minval=0.0,maxval=1.0)
    e1 = gg.uniform(shape=(1,),minval=0.0,maxval=1.0)
    e2 = gggg.uniform(shape=(1,),minval=0.0,maxval=1.0)
    e3 = g.uniform(shape=(1,),minval=0.0,maxval=1.0)
    print(e.numpy(),e1.numpy(),e2.numpy(),e3.numpy())

    for _ in range(100):
        e = ggg.uniform(shape=(1,),minval=0.0,maxval=1.0)
    print(e)
    tf.random.set_seed(1)
    ggg.reset_from_seed(1314)
    tf.random.set_seed(1)
    ggg.skip(100)

    e = ggg.uniform(shape=(1,),minval=0.0,maxval=1.0)
    print(e)
    ggg.reset_from_seed(1314)
    seeds = ggg.make_seeds(count=3)
    print(seeds)
    ggg.reset_from_seed(1314)
    seeds = ggg.make_seeds(count=3)
    print(seeds)
    ggg.reset_from_seed(1314)
    seeds = ggg.make_seeds(count=1)
    print(seeds)
    seeds = ggg.make_seeds(count=1)
    print(seeds)
    seeds = ggg.make_seeds(count=1)
    print(seeds)
    ggg.reset_from_seed(1314)
    ggg.skip(2)
    seeds = ggg.make_seeds(count=1)
    print(seeds)
    
 


    
    



    


 