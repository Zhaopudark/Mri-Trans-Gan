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
Usually we need to preprocess the above files to get our desired ones. So, naming the preprocessed 
file is inevitable. Therefore, we make the following naming convention:

files from or for single modality, such as:
    some files got form t1 modality {some discription} such as {brain} {brain_mask} {...}
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1_{some discription}_.nii.gz
    ...
files made for all modality, such as:
    `mask` that indicate concerned regions on all registrated modalitis
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_mask_.nii.gz

So, we define `mask` `seg` as `shared` modality (Can be shared by the other modalities),
and attach 4 main TAGS type to BraTS Datas:

    training_type:  'Training'  or 'Validation'
    patient_id:     'BraTS2021_00000' 'BraTS2021_00002' ...
    modality:       'flair','t1','t1ce','t2','shared'
    info:           'main' 'norm' 'mask' 'seg' ...

Only `training_type` and `modality` have limited tags. We use tuple to 
confirm an order of TAGS, ('training_type','patient_id','modality','info')

TAGS ('Training','BraTS2021_00000','t1','main') 
    indicate to
    original `t1` modality file of patient `BraTS2021_00000`, i.e.,
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1.nii.gz

TAGS ('Training','BraTS2021_00000','t1','brain') 
    indicate to
    user difined `t1` modality's `brain` file of patient `BraTS2021_00000`, i.e.,
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1_brain.nii.gz

TAGS ('Training','BraTS2021_00000','t1','norm') 
    indicate to
    user difined `t1` modality's `norm` file of patient `BraTS2021_00000`, i.e.,
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_t1_norm.nii.gz

TAGS ('Training','BraTS2021_00000','shared','seg') 
    indicate to
    original `shared` modality's `seg` file of patient `BraTS2021_00000`, i.e.,
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_seg.nii.gz

TAGS ('Training','BraTS2021_00000','shared','mask') 
    indicate to
    user difined `shared` modality's `mask` file of patient `BraTS2021_00000`, i.e.,
    ...\\RSNA_ASNR_MICCAI_BraTS2021_TrainingData\\BraTS2021_00000\\BraTS2021_00000_mask.nii.gz
"""
import re 
import numpy as np
#-----------------------------------config for BraTS------------------------------------------#
FILE_PATTERN = r'.*RSNA_ASNR_MICCAI_BraTS(?P<year>\d*)_(?P<training_type>Training|Validation)Data(?:\\{1}|[/]{1})(?P<patient_id>BraTS\d+_\d+)(?:\\{1}|[/]{1})(?P=patient_id)_(?P<modality>flair|t1ce|t1|t2)?(?:_)?(?P<info>\w*)?(?P<suffix>\.nii\.gz|\.csv|\.*)'
BASE_PATTERN = r'(?P<prefix>.*)(?P<patient_id>BraTS\d+_\d+)(?:_)(?P<modality>flair|t1ce|t1|t2)(?P<suffix>\.nii\.gz)' # used when gen path (by re.sub) from a base existed and correct path 
REPL_BASE_PATTERN = r'\g<prefix>\g<patient_id>_{}\g<suffix>' # used when gen path (by re.sub) from a base existed and correct path 
TAGS = ('training_type','patient_id','modality','info')
TAGS_ORDERS = (('Training','Validation'),None,('flair','t1','t1ce','t2'),None) # used for sort
AXES_FORMAT = ('coronal', 'sagittal', 'vertical')
AXES_DIRECTION_FORMAT = ("R:L", "A:P", "I:S")
SIZE = (240,240,155)
#-----------------------------------functions for BraTS------------------------------------------#
__path_load_prog = re.compile(FILE_PATTERN)
__path_gen_prog = re.compile(BASE_PATTERN)

def get_tags_from_path(path:str):
    _result = __path_load_prog.match(path)
    if _result:
        result = [_result.group(tag_name) for tag_name in TAGS]
    else:
        result = None
    # result = Tags(__path_load_prog.match(path))
    match result:
        case None:
            pass 
        case [_,_,(None|''),_]:
            result[2] = 'shared'
        case [_,_,_,(None|'')]: # info is None or ''
            result[3] = 'main'
        case _:
            pass
    return result

def gen_stamp_from_tags(tags:tuple[str,...]):
    match tags: #rules
        case [training_type,patient_id,_,_]:
            return f"{training_type}_{patient_id}"
        case _:
            raise ValueError(" ") # TODO
def gen_key_tag_from_tags(tags:tuple[str,...]):
    match tags: #rules
        case [_,_,'shared',info]:
            return info
        case [_,_,modality,_]: # [_,_,modality,'main'] or [_,_,modality,other]
            return modality
        case _:
            raise ValueError(" ") # TODO
def get_base_tags_from_tags(tags:tuple[str,...]):
    match tags: #rules
        case [training_type,patient_id,_,_]:
            return (training_type,patient_id,'t1','main')
        case _:
            raise ValueError(" ") # TODO
def gen_path_from_tags(tags:tuple[str,...],base_path:str):
    match tags: #rules
        case [_,_,'shared',info]:
            return __path_gen_prog.sub(REPL_BASE_PATTERN,base_path).format(f"{info}")
        case [_,_,modality,info]:# not shared
            return __path_gen_prog.sub(REPL_BASE_PATTERN,base_path).format(f"{modality}_{info}")
        case _:
            raise ValueError(" ") # TODO
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
