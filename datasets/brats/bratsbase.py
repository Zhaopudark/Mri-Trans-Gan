"""
Consider here are original BraTS datas' paths:
    t1
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1.nii.gz
    t2
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t2.nii.gz
    flair
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_flair.nii.gz
    t1ce
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1ce.nii.gz
    seg (if exist)
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_seg.nii.gz
    ...

we define:
    `Training`,`Validation`  as  'train_or_validate'  
    `BraTS2021_00000` as 'patient_id' or 'pat_id'
    `t1`,`t2`,`flair`,`t1ce`,`shared` as 'modality'
    `seg`,`mask`,`main`,`norm` as 'remark'

e.g.:
    Training-BraTS2021_00000-t1-main 
        means a BraTS's defined training data, who belongs to `BraTS2021_00000`, in `t1` modality, original file
        .../BraTS2021_00000_t1.nii.gz

    Validation-BraTS2021_01798-t2-main 
        means a BraTS's defined validation data, who belongs to `BraTS2021_01798`, in `t2` modality, original file
        .../BraTS2021_01798_t2.nii.gz

    Training-BraTS2021_00000-t1ce-mask
        means a BraTS's defined training data, who belongs to `BraTS2021_00000`, in `t1ce` modality,  `mask` file

    Training-BraTS2021_00000-flair-brain
        means a BraTS's defined training data, who belongs to `BraTS2021_00000`, in `flair` modality, `brain` file, 
        i.e., brain extracted image
        .../BraTS2021_00000_flair_brain.nii.gz

    Training-BraTS2021_00000-flair-brain_mask
        means a BraTS's defined training data, who belongs to `BraTS2021_00000`, in `flair` modality, `brain_mask` file,
        i.e., mask for brain extracted image
        .../BraTS2021_00000_flair_brain_mask.nii.gz

    Training-BraTS2021_00000-shared-seg 
        means a BraTS's defined training data, who belongs to `BraTS2021_00000`, in `shared` modality, the `seg` file, 
        i.e., tumor segmentation map, for all `t1`,`t2`,`flair`,`t1ce` modality
        .../BraTS2021_00000_seg.nii.gz

    Training-BraTS2021_00000-shared-mask
        means a BraTS's defined training data, who belongs to `BraTS2021_00000`, in `shared` modality, the `mask` file,
        i.e., ROI(region of interest) mask  for all `t1`,`t2`,`flair`,`t1ce` modality, usually, it's the `brain_mask`s' aggregation across `t1`,`t2`,`flair`,`t1ce`
        file path is .../BraTS2021_00000_mask.nii.gz
        .../BraTS2021_00000_mask.nii.gz
------------------------------------------------------------------------------
本文件无法自动化,也不应该自动化
使用者必须提前知道brats数据集的若干必要的信息,包括基本的文件组织结构,格式,文件中的信息布局等
根据所知的brats原始文件信息
    指定正则规则, 
    指定分类(打标签)规则
    指定建库(建立文件路径和需要使用的meta信息的数据库)规则
    指定查询规则
"""

from dataclasses import dataclass,field
import logging
import re 
import collections
import json
from typing import Callable
import numpy as np
import pymysql
import pathlib
import os


#----------------------------------sql config-------------------------------------------#
def get_sql_connect():
    return pymysql.connect(host='192.168.3.3',port=3306,user='guest_remote',passwd='123456',database='brats',charset='utf8mb4')
def tb_add_delete_modify(sql:str,args:tuple=None,is_many=False):
    try:
        conn = get_sql_connect()
        with conn.cursor() as cursor:
            if is_many:
                assert args is not None
                cursor.executemany(sql,args)
            else:
                cursor.execute(sql,args)
        logging.getLogger(__name__).info('Insert Successfully!')
        conn.commit()
    except pymysql.MySQLError as error:
        logging.getLogger(__name__).error(error)
        conn.rollback()
    finally:
        conn.close()

def tb_select(sql:str,args:tuple=None):
    try:
        conn = get_sql_connect()
        with conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
            cursor.execute(sql,args)
            tmp_tb = cursor.fetchall()
        logging.getLogger(__name__).info('Select Successfully!')
    except pymysql.MySQLError as error:
        logging.getLogger(__name__).error(error)
    finally:
        conn.close()
    return tmp_tb

def tb_select_gen(sql:str,args:tuple=None,bar:Callable=None):
    try:
        conn = get_sql_connect()
        with conn.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
            cursor.execute(sql,args)
            if callable(bar):
                while row:=cursor.fetchone():
                    bar()
                    yield row
            else:
                while row:=cursor.fetchone():
                    yield row
                
        logging.getLogger(__name__).info('Select Successfully!')
    except pymysql.MySQLError as error:
        logging.getLogger(__name__).error(error)
    finally:
        conn.close()

#-----------------------------------config for BraTS------------------------------------------#
FILE_PATTERN = r'^(?P<prefix>[\w\\/:]*)RSNA_ASNR_MICCAI_BraTS(?P<year>\d*)_(?P<train_or_validate>Training|Validation)Data(?:\\{1}|[/]{1})(?P<patient_id>BraTS\d+_\d+)(?:\\{1}|[/]{1})'\
                r'(?P=patient_id)(?:[_]*)?(?P<modality>[a-zA-Z0-9]*)?(?:[_]*)?(?P<remark>\w*)?(?P<suffix>\.nii\.gz)$'
REPL_PATTERN = r'\g<prefix>RSNA_ASNR_MICCAI_BraTS\g<year>_\g<train_or_validate>Data/'\
               r'\g<patient_id>/\g<patient_id>{}\g<suffix>' # `{}` is for `format` to insert
AXES_FORMAT = ('coronal', 'sagittal', 'vertical')
AXES_DIRECTION_FORMAT = ("R:L", "A:P", "I:S")
META_DATA = {
    'AXES_FORMAT':AXES_FORMAT,
    'AXES_RECONSTRUCTION' :('TBD', 'TBD', 'TBD'),
    'AXES_DIRECTION_FORMAT' : AXES_DIRECTION_FORMAT,
    'SIZE' : (240,240,155),
}

_path_prog = re.compile(FILE_PATTERN)

class SingleRecord(collections.UserDict):
    """
    This class is a `info` collection of a single file. 
    The collected `info` is defined by user, not directly related to the SQL tables
    The post procedure can insert part or whole of the `info` here into SQL tables
    """
    # stamp_tags = ('patient_id','train_or_validate','modality','remark')
    base_record = {   
        'prefix':'[.]*',
        'year':'[.]*',
        'train_or_validate':'[.]*',
        'patient_id':'[.]*',
        'modality':'[.]*',
        'remark':'[.]*',
        'is_basic':'[.]*',
    }
    is_matched = False
    def __init__(self,path:str,meta_data:dict=None):
        super().__init__(self.base_record|{'path':path,'meta_data':json.dumps(META_DATA) if meta_data is None else json.dumps(meta_data)})
        self.init_update()
    def modify_update(self,tags_dict:dict):
        for key in tags_dict:
            assert re.match('^[\w]*$',tags_dict[key]) is not None # prohibit fuzzy tags
        match tags_dict: 
            case {'modality':'shared','remark':'main',**__}:
                tags_dict['modality'] = ''
                tags_dict['remark'] = ''
            case {'remark':'main',**__}:
                tags_dict['remark'] = ''
            case _:
                pass
        inserted = f"{tags_dict['modality']}_{tags_dict['remark']}".strip('_')
        str_path = _path_prog.sub(REPL_PATTERN,self.data['path']).format(f"_{inserted}" if inserted != '' else inserted)
        self.data['path'] = os.fspath(pathlib.Path(str_path).absolute())
        self.init_update()

    def init_update(self):
        if matched:=_path_prog.fullmatch(self.data['path']):
            match tags_dict:=matched.groupdict(): 
                case {'modality':'','remark':'',**__}:
                    tags_dict['modality'] = 'shared'
                    tags_dict['remark'] = 'main'
                case {'modality':modality,'remark':'',**__}:
                    tags_dict['modality'] = modality
                    tags_dict['remark'] = 'main'
                case {'modality':'','remark':remark,**__}: # do not exist since FILE_PATTERN definne `(?:[_]*)?(?P<modality>[a-zA-Z0-9]*)?(?:[_]*)?(?P<remark>\w*)?`
                    tags_dict['modality'] = 'shared'
                    tags_dict['remark'] = remark
                case {'modality':modality,'remark':remark,**__}:
                    pass
                case _:
                    raise ValueError(" ") # TODO
            self.data.update(tags_dict)
            # is_basic
            match self.data:
                case {'modality':'t1'|'t2'|'t1ce'|'flair'|'seg',"remark":'main',**__}:
                    self.data['is_basic'] = True
                case _:
                    self.data['is_basic'] = False
            # stamp
            # self.data['stamp'] = '_'.join(self.data[item] for item in self.stamp_tags)
            self.is_matched = True
        else:
            self.is_matched = False
#-------------------------------------------------------------#
def is_affine_euqal(affine1,affine2):
    # 依据brats文件的特性而定义, 比较两个 affine 是否相等
    return (affine1==affine2).all()

def is_header_euqal(header1,header2):
    # 依据brats文件的特性而定义, 比较两个 header 是否相等
    for (key0,value0),(key1,value1) in zip(header1.items(),header2.items()):
        if key0 != key1:
            return False
        dtype_list = [np.uint8,np.uint16,np.uint32,np.int8,np.int16,np.int32,np.float32]
        if (value0.dtype in dtype_list)and(value1.dtype in dtype_list):
            if (np.isclose(value0,value1,equal_nan=True)).all():
                continue
        elif (value0==value1).all():
            continue
    return True
#-------------------------------------------------------------#
if __name__ == '__main__': 
    import itertools
 

    # targets = tb_select_gen('SELECT tb1.patient_id,`patch_path` FROM tb_patches tb1 JOIN tb_patients '
    #         ' tb2 ON tb1.patient_id=tb2.patient_id where `modality`="t1" and `remark`="individual_min_max_norm" ORDER BY tb1.patient_id and `patch_index`')
 
    targets = tb_select_gen('SELECT tb1.patient_id,`patch_path` FROM tb_patches tb1 JOIN tb_patients '
        ' tb2 ON tb1.patient_id=tb2.patient_id where `modality`="t1" and `remark`="individual_min_max_norm" order by tb1.patient_id, `patch_index`')

    # sorted(data, key=keyfunc)
    for group_name,group in itertools.groupby(targets,key=lambda x:x['patient_id']):
        print(group_name)
        # for item1 in item:
        #     print(item1) 
    # tb_append_patch()
    # for item in select_from_tb({'modality':'^(t1ce|t1|t2|flair|seg)$','remark':'[.]*'}):
    #     print(item)
    # import itertools
    # for modality,remark in itertools.product(['t1','t2','t1ce','flair'],['braissn','brain_maaeaedsk']):
        # update_tb({'modality':modality,'remark':remark})
        # update_tb({'modality':modality,'remark':remark})
        # delete_from_tb({'modality':modality,'remark':remark})


    




    # def tf_py_function_wrapper(func=None):
    #     # since tf.py_function can not deal with dict directly, and its using form is not easy
    #     # here we make this wrapper, it can trans `func`'s output 
    #     # all tensors that a user want to use by calling their `numpy()` functions should become the inputs of the wrapped `func`, otherwise, `numpy()` will not work
    #     # since tf.nest.flatten  tf.nest.pack_sequence_as will sort dict structure's `keys` automaticlly, we do not use tf.nest here to avoid unexpected behavior
    #     if func is None:
    #         return functools.partial(tf_py_function_wrapper,)
    #     @functools.wraps(func)
    #     def wrappered(inputs:dict[str,tf.Tensor],output_structure:dict[str,tf.TensorSpec])->dict[str,tf.Tensor]:
    #         inp = tuple(inputs.values())
    #         Tout = tuple(output_structure.values())
    #         flattened_output = tf.py_function(func,inp=inp,Tout=Tout)
    #         return dict(zip(output_structure.keys(),flattened_output))
    #     return wrappered    

    # class Tags():
    #     __match_args__ = TAGS
    #     def __init__(self,match_result:Match|tuple):
    #         if match_result is None:
    #             for attr_name in TAGS:
    #                 self.__setattr__(attr_name,None)
    #         elif isinstance(match_result,Match):
    #             for attr_name in TAGS:
    #                 self.__setattr__(attr_name,match_result.group(attr_name))
    #         elif isinstance(match_result,tuple):
    #             for attr_name,attr_value in zip(TAGS,match_result):
    #                 self.__setattr__(attr_name,attr_value)
    #         else:
    #             raise ValueError(" ") # TODO

    # # _training_type_pattern = r'(?P<training_type>Training|Validation)'
    # # _patient_id_pattern = r'(?P<patient_id>BraTS\d+_\d+)'
    # # _modality_pattern = r'(?P<modality>flair|t1ce|t1|t2)'
    # # _info_pattern = r'(?P<info>\w*)'
    # # _suffix_pattern = r'(?P<suffix>\.nii\.gz|\.yaml|\.*)'
    # class StringWithPattern():
    #     def __init__(self,data) -> None:
    #         self.prog = re.compile(data)
    #     def __eq__(self,string):
    #         if isinstance(string,StringWithPattern):
    #             return self.prog == string.prog
    #         _result = self.prog.match(string)
    #         return False if _result is None else string == _result.group(0)
    # from enum import Enum
    # class DataTemplate(Enum):
    #     TRAINING_TYPE = StringWithPattern(r'(?P<training_type>Training|Validation)')
    #     PATIENT_ID = StringWithPattern(r'(?P<patient_id>BraTS\d+_\d+)')
    #     MODALITY = StringWithPattern(r'(?P<modality>flair|t1ce|t1|t2)')
    #     INFO = StringWithPattern(r'(?P<info>\w*)')
    # # suffix = StringWithPattern(r'(?P<suffix>\.nii\.gz|\.yaml|\.*)')

    # # swp = StringWithPattern(_training_type_pattern)
    # # swp2 = StringWithPattern(_patient_id_pattern)
    # # swp3 = StringWithPattern(_training_type_pattern)
    # # print(swp=="Training")
    # # print(swp=="training")
    # # print(swp==swp2)
    # # print(swp==swp3)

    # class OrigianlDatas():
    #     __match_args__ = ('training_type','patient_id','modality','info')
    #     def __init__(self,training_type,patient_id,modality,info):
    #         self.training_type = DataTemplate.TRAINING_TYPE
    #         self.patient_id = DataTemplate.PATIENT_ID
    #         self.modality = DataTemplate.MODALITY
    #         self.info = DataTemplate.INFO

    # # tests = OrigianlDatas('Training','00000000','t1','main')
    # # target = OrigianlDatas(training_type=r'(?P<training_type>Training|Validation)',patient_id=r'(?P<patient_id>BraTS\d+_\d+)',modality=r'(?P<modality>flair|t1ce|t1|t2)',info=r'(?P<info>\w*)')
    # tests = StringWithPattern(r'(?P<training_type>Training|Validation)')
    # match tests:
    #     case DataTemplate.TRAINING_TYPE:
    #         print('matched')