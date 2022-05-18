import os 
import sys
import platform
import logging

import pytest
import itertools
import numpy as np
import nibabel as nib
import re
from datasets.brats.brats_data import BraTSDataPathCollection
import functools
PATH = "D:\\Datasets\\BraTS\\BraTS2021_new"
file_pattern = re.compile(r'RSNA_ASNR_MICCAI_BraTS(?P<year>\d*)_(?P<training_type>Training|Validation)Data(?:\\{1}|[/]{1})(?P<patient_id>BraTS\d+_\d+)(?:\\{1}|[/]{1})(?P=patient_id)_(?P<modality>flair|t1ce|t1|t2)?(?:_)?(?P<info>\w*)?(?P<suffix>\.nii\.gz|\.csv|\.*)$')
data_path_collection = BraTSDataPathCollection(path=PATH)
keys_list = ['training_type','patient_id','modality','info']
@pytest.mark.parametrize('training_type',['Training','Validation'])
@pytest.mark.parametrize('modality',['t1','t2','t1ce','flair'])
@pytest.mark.parametrize('info',['main','brain','brain_mask','min_max_norm','z_score_norm','z_score_and_min_max_norm','individual_min_max_norm'])
def test_individual_paths(training_type,modality,info):
    
    for item in data_path_collection[(training_type,None,modality,info)]:
        matched = file_pattern.search(item[1])
        assert matched.group('training_type')==training_type
        assert matched.group('modality')==modality
        if info=='main':
            assert matched.group('info')==""
        else:
            assert matched.group('info')==info
@pytest.mark.parametrize('training_type',['Training','Validation'])
@pytest.mark.parametrize('modality',['shared'])
@pytest.mark.parametrize('info',['mask','seg'])
def test_shared_paths(training_type,modality,info):
    for item in data_path_collection[(training_type,None,modality,info)]:
        matched = file_pattern.search(item[1])
        assert matched.group('training_type')==training_type
        if (training_type=='Training')and(info=='seg'):
            assert os.path.exists(item[1])
        assert matched.group('modality') is None or matched.group('modality')==""
        assert matched.group('info')==info

@pytest.mark.parametrize('training_type',['Training','Validation'])
def test_individual_paths_num(training_type):
    buf = []
    for i,item in enumerate(data_path_collection[(training_type,None,('t1','t2','t1ce','flair'),'main')]):
        pass 
    buf.append(i)
    for i,item in enumerate(data_path_collection[(training_type,None,('t1','t2','t1ce','flair'),'brain')]):
        pass 
    buf.append(i)
    for i,item in enumerate(data_path_collection[(training_type,None,('t1','t2','t1ce','flair'),'brain_mask')]):
        pass 
    buf.append(i)
    for i,item in enumerate(data_path_collection[(training_type,None,('t1','t2','t1ce','flair'),'min_max_norm')]):
        pass 
    buf.append(i)
    for i,item in enumerate(data_path_collection[(training_type,None,('t1','t2','t1ce','flair'),'z_score_norm')]):
        pass 
    buf.append(i)
    for i,item in enumerate(data_path_collection[(training_type,None,('t1','t2','t1ce','flair'),'z_score_and_min_max_norm')]):
        pass 
    buf.append(i)
    for i,item in enumerate(data_path_collection[(training_type,None,('t1','t2','t1ce','flair'),'individual_min_max_norm')]):
        pass 
    buf.append(i)

    def _reduce_same(x1,x2):
        assert x1==x2 
        return x2 
    out = functools.reduce(_reduce_same,buf)
    assert out == buf[0]

@pytest.mark.parametrize('training_type',['Training','Validation'])
def test_shared_paths_num(training_type):
    buf = []
    for i,item in enumerate(data_path_collection[(training_type,None,('shared',),'mask')]):
        pass 
    buf.append(i)
    for i,item in enumerate(data_path_collection[(training_type,None,('shared',),'seg')]):
        pass 
    buf.append(i)
    for i,item in enumerate(data_path_collection[(training_type,None,('shared',),'xxxxxx')]):
        pass 
    buf.append(i)
    def _reduce_same(x1,x2):
        assert x1==x2 
        return x2 
    out = functools.reduce(_reduce_same,buf)
    assert out == buf[0]

def test_reduce_datas_and_inverse_reduce_datas():
    input_datas = data_path_collection.get_individual_datas('main',['mask'])
    length1 = len(input_datas)
    input_datas = BraTSDataPathCollection.reduce_datas(input_datas)
    def check(x1,x2):
        assert x1==x2 
        return x2
    length2 = functools.reduce(check,map(len,input_datas.values()))
    mask_paths = input_datas.pop('mask')
    mask_paths = BraTSDataPathCollection.inverse_reduce_datas({'mask':mask_paths})
    assert len(list(mask_paths)) == length2 == length1

    input_datas = data_path_collection.get_individual_datas('main',['mask'])
    output_datas = data_path_collection.get_individual_datas('brain_mask',['segggg'])
    x = list(BraTSDataPathCollection.inverse_reduce_datas(BraTSDataPathCollection.reduce_datas(input_datas[:])))
    y = list(BraTSDataPathCollection.inverse_reduce_datas(BraTSDataPathCollection.reduce_datas(x)))
    for item1,item2 in zip(x,y):
        assert item1 == item2

    for inputs,outputs in zip(input_datas,output_datas):
        assert inputs.name==outputs.name
    input_datas = BraTSDataPathCollection.reduce_datas(input_datas)
    _ = input_datas.pop('mask')
    input_datas = BraTSDataPathCollection.inverse_reduce_datas(input_datas)
    output_datas = BraTSDataPathCollection.reduce_datas(output_datas)
    _ = output_datas.pop('segggg')
    output_datas = BraTSDataPathCollection.inverse_reduce_datas(output_datas)
    for inputs,outputs in zip(input_datas,output_datas):
        for (k1,v1),(k2,v2) in zip(inputs.items(),outputs.items()):
            assert k1==k2 
