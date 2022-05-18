import sys 
import os 
class Data():
    """
    In BraTS dataset, if we directly 
    read a ".nii" file and transform it to numpy array,
    we will get a [Z,Y,X] tensor in shape [240,240,155],
        `Z` dimension: coronal axis, index min to max == R(Right) to L(Left)
        `Y` dimension: sagittal axis, index min to max == A(Anterior) to P(Posterior)
        `X` dimension: vertical axis, index min to max == I(Inferior) to S(Superior)
    This can be recorded as `R:L,A:P,I:S` format.

        [Z,:,:] == a slice of sagittal plane
        [:,Y,:] == a slice of coronal plane
        [:,:,X] == a slice of transverse plane or horizontal plane

    Generally, in image-to-image translations, we concern more about transverse plane slices,
    treating it as a 2-D image, with sagittal and coronal axes considered as H(Height) and W(Width) dimensions.
    What's more, we usually use [D(Depth),H(Height),W(Width)] shape to represent a 3D image. 
    So, if D(Depth), H(Height) and W(Width) dimensions are vertical, sagittal and  coronal axes respectively,
    it will be helpful for next procedures.

    So, the [Z,Y,X] tensor from BraTS should be transpose to [X,Y,Z]
    If directly tranpose, we will get `I:S,A:P,R:L` format.
    For more conveniently drawing, we transpose [Z,Y,X] tensor to [-X,Y,Z],
    then, we get `S:I,A:P,R:L` format. Regard  [-X,Y,Z] dimension as [D1,D2,D3]
        Then [D1,:,:] is  transverse plane or horizontal plane in `A:P,R:L` format
        Then [:,D2,:] is  coronal plane in `S:I,R:L` format
        Then [:,:,D3] is  sagittal plane in `S:I,A:P` format.


    This class describe a collection of each individual patient files on BraTS dataset.
    Args:
        path: a directory that contains all files (modalities) (`nii.gz` or `nii` file) of an individual patient.
    """
    def __init__(self,path,norm="") -> None:
        self.__modality_names = ["flair.nii.gz","t1.nii.gz","t1ce.nii.gz","t2.nii.gz"] #对内具象
        self._meta_data = {
            "modalities": ["flair", "t1", "t1ce", "t2"],
            "norm_method": ["z_score", "min_max"],
            "suffix": ".nii.gz",
            "key_word": None,
            "axes_format": ("coronal", "sagittal", "vertical"),
            "direction_format": ("R:L", "A:P", "I:S"),
        }

        self._additional_meta_data = {
            "traget_aixs_format": ("vertical", "sagittal", "coronal"),
            "traget_direction_format": ("S:I", "A:P", "R:L"),
        }
        self.data_dict = {}
    

        self._raw_file = {}
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
        
    @property
    def raw_data_paths_dict(self):
        if not hasattr(self,"_raw_data_paths_dict"):
            self._raw_data_paths_dict = {}
        for (dir_name,_,file_list) in os.walk(self.path):
            for file_name in file_list:
                for modality in self._meta_data["modalities"]:
                    corrected_name = modality.lower()+self._meta_data["suffix"]
                    if corrected_name in file_name.lower():
                        self._raw_data_paths_dict[modality]=os.path.join(dir_name,file_name)
                        break # one modality has one file

    def get_reshape_perm(self,target,source):
        return (target.index(item) for item in source)
         
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

if __name__ == "__main__":
    import nibabel as nib 
    import numpy as np
    import tensorflow as tf 
    _meta_data = {}
    _meta_data["axes_format"] = ("coronal","sagittal","vertical")
    _meta_data["direction_format"] = ("R:L","A:P","I:S")
    _additional_meta_data = {}
    _additional_meta_data["traget_aixs_format"] = ("vertical","sagittal","coronal")
    _additional_meta_data["traget_direction_format"] = ("S:I","A:P","R:L")
    def get_reshape_perm(target:list,source:list):
        if target and source:
            assert len(target)==len(source)
            return tuple(source.index(item) for item in target) # perm conducted on source
        else:
            return tuple()
    print(get_reshape_perm(_additional_meta_data["traget_aixs_format"],_meta_data["axes_format"]))
    print(get_reshape_perm((),()))
