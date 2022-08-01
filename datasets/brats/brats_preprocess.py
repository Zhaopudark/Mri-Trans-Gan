"""
本文件 brats_preprocess.py 的架构十分杂乱, 通俗的说, 写的很烂, 
随着下一层和上一层架构的不断变动, 本层的设计很难形成章法
所以 以不变应万变, 定义若干显式的函数, 完成显式的预处理操作即可, 不再过多考虑设计模式
"""
import os
import pathlib
import itertools
import json
import operator
import platform
import functools
from re import L
from subprocess import check_output
from typing import Literal,Generator
from typeguard import typechecked
import numpy as np
import tensorflow as tf
from utils.managers import get_simple_logger
from utils.operations import norm_min_max,norm_z_score,read_nii_file,save_nii_file,np_zero_close,np_sequence_reduce_min_max,reduce_same
from utils.operations import get_subranges
from datasets.brats import bratsbase
#--------------------------------------------------------#
_logger = get_simple_logger("BraTS Preprocess")
#--------------------------------------------------------#
def _get_all_file_paths(path:str)->Generator[str|bytes, None, None]:
    return (os.fspath(child.absolute()) for child in pathlib.Path(path).glob('**/*') if child.is_file())
def initial_records(path:str): # tb_patients tb_modalities
    tb_patients_buf = []
    tb_patients_getter = operator.itemgetter('patient_id')
    tb_modalities_buf = []
    tb_modalities_getter = operator.itemgetter('patient_id','train_or_validate','modality','remark','is_basic','path','meta_data')
    
    for str_path in _get_all_file_paths(path):
        single_record = bratsbase.SingleRecord(str_path,bratsbase.META_DATA)
        if single_record.is_matched:
            tb_patients_buf.append(tb_patients_getter(single_record))
            tb_modalities_buf.append(tb_modalities_getter(single_record))
    bratsbase.tb_add_delete_modify('insert ignore into tb_patients (`patient_id`) values (%s)',tb_patients_buf,is_many=True)
    bratsbase.tb_add_delete_modify('insert ignore into tb_modalities ('
                        '`patient_id`,`train_or_validate`,`modality`,'
                        '`remark`,`is_basic`,`path`,`meta_data`) '
                        'values (%s,%s,%s,%s,%s,%s,%s)', tb_modalities_buf, is_many=True)
def infer_records(tags_dict:dict[Literal['modality','remark'],str]): # tb_modalities
    tb_modalities_buf = []
    tb_modalities_getter = operator.itemgetter('patient_id','train_or_validate','modality','remark','is_basic','path','meta_data')
    tmp_tb_base = bratsbase.tb_select('select * from tb_modalities where '
                    '`modality` regexp %s '
                    'and `remark` regexp %s order by `patient_id`',('^t1$','^main$'))
    for row in tmp_tb_base:
        single_record = bratsbase.SingleRecord(row['path'],json.loads(row['meta_data']))
        assert single_record.is_matched
        single_record.modify_update(tags_dict)
        if single_record.is_matched:
            tb_modalities_buf.append(tb_modalities_getter(single_record))
    bratsbase.tb_add_delete_modify('insert ignore into tb_modalities ('
                        '`patient_id`,`train_or_validate`,`modality`,'
                        '`remark`,`is_basic`,`path`,`meta_data`) '
                        'values (%s,%s,%s,%s,%s,%s,%s)', tb_modalities_buf, is_many=True)
def get_records(tags_dict:dict[Literal['modality','remark'],str]): # tb_modalities
        return bratsbase.tb_select('select * from tb_modalities where '
                        '`modality` regexp %s '
                        'and `remark` regexp %s order by `patient_id`',(tags_dict['modality'],tags_dict['remark']))
#-----------------preprocess functions definition-----------------#
def resample():
    raise ValueError("BraTS do not need Resample.")
def registration():
    raise ValueError("BraTS do not need Registration.")
def get_reshape_perm(self,target,source):
    return (target.index(item) for item in source)   
def bet():
    if platform.system() != 'Linux':
        raise ValueError(
            f"Bet in FSL must run on Linux instead of {platform.system()}."
        )
    for modality,remark in itertools.product(['t1','t2','t1ce','flair'],['brain','brain_mask']):
        infer_records({'modality':modality,'remark':remark})
    
    
    for modality in ['^t1$','^t2$','^t1ce$','^flair$']:
        inputs = get_records({'modality':modality,'remark':'^main$'})
        outputs = get_records({'modality':modality,'remark':'^brain$'})
        _ = get_records({'modality':modality,'remark':'^brain_mask$'})
        bar = tf.keras.utils.Progbar(len(inputs),width=30,verbose=1,interval=0.5,stateful_metrics=None,unit_name=f"Doing bet on `{modality}`...")
        for x,y in zip(inputs,outputs):
            assert x['patient_id']==y['patient_id']
            bar.add(1)
            reg_cmd = f"bet {x['path']} {y['path']} -m -R -f 0.05"
            # -m lead to gen mask with np.int16 dtype
            reg_info = check_output(reg_cmd,shell=True).decode()
            _logger.info(f"{reg_cmd} checkinfo is:{reg_info}")

def combine_masks():
    for modality,remark in itertools.product(['t1','t2','t1ce','flair'],['brain_mask']):
        infer_records({'modality':modality,'remark':remark})
    for modality,remark in itertools.product(['mask'],['main']):
        infer_records({'modality':modality,'remark':remark})
    def gen():
        yield from zip(
                get_records({'modality':'^t1$','remark':'^brain_mask$'}),
                get_records({'modality':'^t2$','remark':'^brain_mask$'}),
                get_records({'modality':'^t1ce$','remark':'^brain_mask$'}),
                get_records({'modality':'^flair$','remark':'^brain_mask$'}),
                get_records({'modality':'^mask$','remark':'^main$'}))
    def combine_mask_infos(x,y):
        if isinstance(x,str):
            x_img,x_affine,x_header = read_nii_file(x,dtype=np.int16)
            y_img,y_affine,y_header = read_nii_file(y,dtype=np.int16)
            assert x_img.shape==y_img.shape
            assert bratsbase.is_affine_euqal(x_affine,y_affine)
            assert bratsbase.is_header_euqal(x_header,y_header)
            return np_zero_close(x_img*y_img),x_affine,x_header
        x_img,x_affine,x_header = x
        y_img,y_affine,y_header = read_nii_file(y,dtype=np.int16)
        assert x_img.shape==y_img.shape
        assert bratsbase.is_affine_euqal(x_affine,y_affine)
        assert bratsbase.is_header_euqal(x_header,y_header)
        return np_zero_close(x_img*y_img),x_affine,x_header
    bar = tf.keras.utils.Progbar(len(get_records({'modality':'^mask$','remark':'^main$'})),width=30,verbose=1,interval=0.5,stateful_metrics=None,unit_name="Combining masks...")
    for *inputs,outputs in gen():
        assert reduce_same(inputs,map_func=lambda x:x['patient_id'])==outputs['patient_id']
        bar.add(1)
        mask,affine,header = functools.reduce(combine_mask_infos,[item['path'] for item in inputs])
        save_nii_file(mask,outputs['path'],affine=affine,header=header)
        _logger.info(outputs['path'])

@typechecked
def norm_with_mask(foreground_offset:int|float,norm_method:str):
    for modality,remark in itertools.product(['t1','t2','t1ce','flair'],['main',norm_method]):
        infer_records({'modality':modality,'remark':remark})
    for modality,remark in itertools.product(['mask'],['main']):
        infer_records({'modality':modality,'remark':remark})
    for modality in ['^t1$','^t2$','^t1ce$','^flair$']:
        def gen():
            yield from zip(get_records({'modality':modality,'remark':'^main$'}),
                        get_records({'modality':'^mask$','remark':'^main$'}),
                        get_records({'modality':modality,'remark':f"^{norm_method}$"}))
        if norm_method == 'min_max_norm':
            norm_func =  norm_min_max
            use_global_min_max=True
        elif norm_method == 'individual_min_max_norm':
            norm_func = norm_min_max
            use_global_min_max = False 
        elif norm_method == 'z_score_norm':
            _logger.warning(f"z_score 不仅将有效区域进行标准化~N({foreground_offset},1.0) 也将背景归0 有可能会影响模型性能")
            norm_func = norm_z_score
            use_global_min_max = False 
        elif norm_method == 'z_score_and_min_max_norm':
            norm_func = norm_min_max
            use_global_min_max=True
            def gen():
                yield from zip(get_records({'modality':modality,'remark':'^z_score_norm$'}),
                        get_records({'modality':'^mask$','remark':'^main$'}),
                        get_records({'modality':modality,'remark':f"^{norm_method}$"}))
        else: # 其他归一化方法
            raise  ValueError("") #TODO 

        if use_global_min_max:
            _logger.warning(f"基于全局的min_max需要考虑到各个样本是否配准到同一模板, 为配准")
            bar = tf.keras.utils.Progbar(len(get_records({'modality':modality,'remark':'^main$'})),
                width=30,verbose=1,interval=0.5,stateful_metrics=None,unit_name=f"Computing global min max on `{modality}`...")
            def sequence_gen():
                for inputs,mask,outputs in gen():
                    bar.add(1)
                    _input,*_ = read_nii_file(inputs['path'])
                    _mask,*_ = read_nii_file(mask['path'],dtype=np.int16)
                    yield np_zero_close(_input*_mask)
            global_min_max = np_sequence_reduce_min_max(sequence_gen())
            tf.print(global_min_max)
        else:
            global_min_max = None
        bar = tf.keras.utils.Progbar(len(get_records({'modality':modality,'remark':'^main$'})),
                width=30,verbose=1,interval=0.5,stateful_metrics=None,unit_name=f"Computing norm on `{modality}` ...")
        for inputs,mask,outputs in gen():
            bar.add(1)
            _input,affine,header = read_nii_file(inputs['path'])
            _mask,*_ = read_nii_file(mask['path'],dtype=np.int16)
            output = norm_func(x=_input,
                                global_min_max=global_min_max,
                                mask=_mask,
                                foreground_offset=foreground_offset,
                                dtype=np.float32)
            save_nii_file(output,outputs['path'],affine=affine,header=header)
            _logger.info(outputs['path'])  

@typechecked
def del_files(target_type:Literal['mask','brain','brain_mask',
                                'individual_min_max_norm',
                                'min_max_norm','z_score_norm',
                                'z_score_and_min_max_norm']): # 缩减逻辑
    if target_type == 'mask':
        del_buf = get_records({'modality':'^mask$','remark':'^main$'})
    else:
        del_buf = tuple(
            itertools.chain(
                *(get_records({'modality':modality,'remark':f'^{target_type}$'}) for modality in ['^t1$','^t2$','^t1ce$','^flair$'])
            )) 
    bar = tf.keras.utils.Progbar(len(del_buf),
                width=30,verbose=1,interval=0.5,stateful_metrics=None,unit_name=f" Delete files correspond to `{target_type}` ")
    for item in del_buf:
        bar.add(1)
        try:
            os.remove(item['path'])
            _logger.info(f"remove {item['path']}")
        except FileNotFoundError:
            _logger.info(f"{item['path']} does not exist! Ignored") 

def append_meta_data(): 
    buf = []
    bar = tf.keras.utils.Progbar(len(get_records({'modality':'^mask$','remark':'^main$'})),
                width=30,verbose=1,interval=0.5,stateful_metrics=None,unit_name="Append meta_data on mask ...")
    for item in get_records({'modality':'^mask$','remark':'^main$'}):
        bar.add(1)
        _input,*_ = read_nii_file(item['path'])
        indices = np.nonzero(_input)
        basic_meta_data = json.loads(item['meta_data'])
        basic_meta_data['VALID_RANGES'] = tuple((int(indice.min()),int(indice.max())) for indice in indices)
        buf.append((json.dumps(basic_meta_data),item['patient_id'],item['train_or_validate'],item['modality'],item['remark']))
    
    bratsbase.tb_add_delete_modify('update tb_modalities set meta_data=%s where'
    '`patient_id`=%s and `train_or_validate`=%s and `modality`=%s and `remark`=%s',buf,is_many=True)

def initial_patches(
    stored_dir:str,
    modality:Literal['individual_min_max_norm',],
    patch_sizes:tuple[int,...],
    patch_overlap_tolerance:tuple[tuple[int,int],...]):

    records = get_records({'modality':'^(t1ce|t1|t2|flair)$','remark':f'^{modality}$'})+get_records({'modality':'^mask$','remark':'^main$'})
    insert_buf = []
    bar = tf.keras.utils.Progbar(len(records),
                width=30,verbose=1,interval=0.5,stateful_metrics=None,unit_name="Initial info of patches ...")
    for item in records:
        bar.add(1)
        mask_record = bratsbase.tb_select('select * from tb_modalities where '
                        '`patient_id`=%s and `train_or_validate`=%s and `modality`=%s and `remark`=%s',
                        (item['patient_id'],item['train_or_validate'],'mask','main')
                        )[0]
        meta_data = json.loads(mask_record['meta_data'])
        # print("the valid_ranges: ", valid_ranges)
        valid_ranges = meta_data['VALID_RANGES']
        total_ranges = [(0,item-1) for item in meta_data['SIZE']]
        patch_ranges_per_axis = [get_subranges(*item) for item in zip(total_ranges,valid_ranges,patch_sizes,patch_overlap_tolerance)]
        patch_nums = functools.reduce(lambda x1,x2:x1*len(x2),patch_ranges_per_axis,1)

        for i,patch_ranges in enumerate(itertools.product(*patch_ranges_per_axis)):
            serialized_patch_ranges = '-'.join(f"{x1}x{x2}" for x1,x2 in patch_ranges)
            stamp_buf = [item['patient_id'],
            item['train_or_validate'],
            item['modality'],
            item['remark'],
            serialized_patch_ranges]

            patch_meta_data = meta_data|{
                            'TOTAL_RANGES':total_ranges,
                            'VALID_RANGES':valid_ranges,
                            'PATCH_SIZES':patch_sizes,
                            'PATCH_OVERLAP_TOLERANCE':patch_overlap_tolerance,
                            'PATCH_RANGES':patch_ranges,
                            'PATCH_INDEX':i,
                            'MAX_INDEX':patch_nums-1}
            patch_path = pathlib.Path(stored_dir+'\\'+'_'.join(stamp_buf)+'.npy')
            insert_buf.append((*stamp_buf[:-1],i,json.dumps(patch_ranges),patch_path,json.dumps(patch_meta_data)))
    bratsbase.tb_add_delete_modify('insert ignore into tb_patches ('
                        '`patient_id`,`train_or_validate`,`modality`,'
                        '`remark`,`patch_index`,`patch_range`,`patch_path`,`patch_meta_data`) '
                        'values (%s,%s,%s,%s,%s,%s,%s,%s)',insert_buf,is_many=True)

def divide_patches(path:str):
    patches = bratsbase.tb_select('select * from tb_patches where `modality` regexp "(t1ce|t1|t2|flair|mask)$"')
    bar = tf.keras.utils.Progbar(len(patches),
                width=30,verbose=1,interval=0.5,stateful_metrics=None,unit_name="Dividing patches ...")
    for i,item in enumerate(patches):
        bar.add(1)
        original = bratsbase.tb_select('select * from tb_modalities where '
                        '`patient_id`=%s and `train_or_validate`=%s and `modality`=%s and `remark`=%s',
                        (item['patient_id'],item['train_or_validate'],item['modality'],item['remark'])
                        )[0]
        if item['modality']=='mask':
            img,*_ = read_nii_file(original['path'],dtype=np.int32)
        else:
            img,*_ = read_nii_file(original['path'],dtype=np.float32)
        patch_range = json.loads(item['patch_range'])
        patch_slice = tuple(slice(x1,x2+1) for x1,x2 in patch_range)
        save_path = pathlib.Path(f"{path}/{item['patch_path']}")
        np.save(save_path,img[patch_slice])     
  
if __name__=='__main__':


    # initial_records(path="D:\\Datasets\\BraTS\\BraTS2021\\Datas")
    # # bet()
    # combine_masks()
    # del_files('z_score_and_min_max_norm')
    # del_files('z_score_norm')
    # del_files('min_max_norm')
    # preprocess.del_files('individual_min_max_norm')
    # preprocess.del_files('mask')
    # preprocess.del_files('brain_mask')
    # preprocess.del_files('brain')
    # append_meta_data()
    initial_patches("D:\\Datasets\\BraTS\\BraTS2021\\patch_records",'individual_min_max_norm',patch_sizes=[64,64,64],patch_overlap_tolerance=((0.2,0.3),(0.2,0.3),(0.2,0.3)))

    # bratsbase.tb_select('select * from tb_modalities where '
    #                     '`modality` regexp %s '
    #                     'and `remark` regexp %s order by `patient_id`',(tags_dict['modality'],tags_dict['remark']))
    # divide_patches(path="D:\\Datasets\\BraTS\\BraTS2021\\patch_records")

    # preprocess.bet()
    # preprocess.combine_masks()
    # preprocess.norm_with_mask(foreground_offset=0.001,norm_method='individual_min_max_norm')
    # preprocess.norm_with_mask(foreground_offset=0.001,norm_method='min_max_norm')
    # preprocess.norm_with_mask(foreground_offset=0.0,norm_method='z_score_norm')
    # norm_with_mask(foreground_offset=0.001,norm_method='z_score_and_min_max_norm')



    # preprocess.del_files('mask')
    # preprocess.combine_masks()
    # preprocess.del_files('z_score_norm')
    # preprocess.del_files('min_max_norm')
    # preprocess.del_files('z_score_and_min_max_norm')
    # preprocess.del_files('individual_min_max_norm')
    # preprocess.norm_with_masks(foreground_offset=0.001,norm_method='individual_min_max_norm')
    # preprocess.norm_with_masks(foreground_offset=0.001,norm_method='min_max_norm')
    # preprocess.norm_with_masks(foreground_offset=0.0,norm_method='z_score_norm')
    # preprocess.norm_with_masks(foreground_offset=0.001,norm_method='z_score_and_min_max_norm')
    