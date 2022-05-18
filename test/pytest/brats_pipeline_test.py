import pytest
import numpy as np
from datasets.brats.brats_pipeline import BraTSBasePipeLine,BraTSDividingWrapper,BraTSPatchesWrapper


def test_BraTSBasePipeLine():
    d = BraTSBasePipeLine(
        path="D:\\Datasets\\BraTS\\BraTS2021_new",
        record_path="D:\\Datasets\\BraTS\\BraTS2021_new\\records",
        norm_method='individual_min_max_norm',
        seed=None)
    g = d()
    for i,item in enumerate(g()):
        pass
    assert (i+1)==len(d.datas)
def test_BraTSDividingWrapper():
    d = BraTSBasePipeLine(
        path="D:\\Datasets\\BraTS\\BraTS2021_new",
        record_path="D:\\Datasets\\BraTS\\BraTS2021_new\\records",
        norm_method='individual_min_max_norm',
        seed=None)
    d1 = BraTSDividingWrapper(d,dividing_rates=(0.7,0.2,0.1))
    g1,g2,g3 = d1()
    for i,item in enumerate(g1()):
        pass
    assert (i+1)==len(d1.datas[0])
    for i,item in enumerate(g2()):
        pass
    assert (i+1)==len(d1.datas[1])
    for i,item in enumerate(g3()):
        pass
    assert (i+1)==len(d1.datas[2])


def test_BraTSPatchesWrapper():
    d = BraTSBasePipeLine(
        path="D:\\Datasets\\BraTS\\BraTS2021_new",
        record_path="D:\\Datasets\\BraTS\\BraTS2021_new\\records",
        norm_method='individual_min_max_norm',
        seed=0)
    d1 = BraTSPatchesWrapper(d,cut_ranges=((155//2-8,155//2+7),(0,239),(0,239)),patch_sizes=(16,128,128),patch_nums=(1,3,3))
    g1 = d1()
    for i,item in enumerate(g1()):
        pass
    assert (i+1)==len(d1.datas)

    d2 = BraTSDividingWrapper(d,dividing_rates=(0.7,0.2,0.1))
    d3 = BraTSPatchesWrapper(d2,cut_ranges=((155//2-8,155//2+7),(0,239),(0,239)),patch_sizes=(16,128,128),patch_nums=(1,3,3))
    g1,g2,g3 = d3()
    for i,item in enumerate(g1()):
        pass
    assert (i+1)==len(d3.datas[0])
    for i,item in enumerate(g2()):
        pass
    assert (i+1)==len(d3.datas[1])
    for i,item in enumerate(g3()):
        pass
    assert (i+1)==len(d3.datas[2])

def test_patch_combine():
    d0 = BraTSBasePipeLine(path="D:\\Datasets\\BraTS\\BraTS2021_new",record_path="D:\\Datasets\\BraTS\\BraTS2021_new\\records",seed=None)
    d = BraTSBasePipeLine(path="D:\\Datasets\\BraTS\\BraTS2021_new",record_path="D:\\Datasets\\BraTS\\BraTS2021_new\\records",seed=None)
    d2 = BraTSPatchesWrapper(d,cut_ranges=((155//2-8,155//2+7),(0,239),(0,239)),patch_sizes=(16,128,128),patch_nums=(1,3,3))
    generator_func = d2()
    def gen(g):
        for item in g:
            t1 = item['t1']
            t2 = item['t2']
            t1ce = item['t1ce']
            flair = item['flair']
            mask = item['mask']
            m = item['patch_mask']
            v = item['patch_padding_vector']
            yield {'t1':{'img':t1,
                        'mask':m,
                        'padding_vector':v},
                    't2':{'img':t2,
                        'mask':m,
                        'padding_vector':v},
                    't1ce':{'img':t1ce,
                        'mask':m,
                        'padding_vector':v},
                    'flair':{'img':flair,
                        'mask':m,
                        'padding_vector':v},
                    'mask':{'img':mask,
                        'mask':m,
                        'padding_vector':v},
                    }

    for item,item0 in zip(d2.patch_combine_generator(gen(generator_func())),d0()()):
        assert item['t1']['img'].shape == item['t1']['mask'].shape == (16,240,240)
        assert item['t2']['img'].shape == item['t2']['mask'].shape == (16,240,240)
        assert item['t1ce']['img'].shape == item['t1ce']['mask'].shape == (16,240,240)
        assert item['flair']['img'].shape == item['flair']['mask'].shape == (16,240,240)
        assert item['mask']['img'].shape == item['mask']['mask'].shape == (16,240,240)

        assert np.sum(item['t1']['mask'])==16*240*240
        assert np.sum(item['t2']['mask'])==16*240*240
        assert np.sum(item['t1ce']['mask'])==16*240*240
        assert np.sum(item['flair']['mask'])==16*240*240
        assert np.sum(item['mask']['mask'])==16*240*240

        assert np.isclose(np.mean(item['t1']['img']-item0['t1'][155//2-8:155//2+7+1,0:240,0:240]),0.0)
        assert np.isclose(np.mean(item['t2']['img']-item0['t2'][155//2-8:155//2+7+1,0:240,0:240]),0.0)
        assert np.isclose(np.mean(item['t1ce']['img']-item0['t1ce'][155//2-8:155//2+7+1,0:240,0:240]),0.0)
        assert np.isclose(np.mean(item['flair']['img']-item0['flair'][155//2-8:155//2+7+1,0:240,0:240]),0.0)
        assert np.isclose(np.mean(item['mask']['img']-item0['mask'][155//2-8:155//2+7+1,0:240,0:240]),0.0)
def test_patch_combine_dividing():
    d0 = BraTSBasePipeLine(path="D:\\Datasets\\BraTS\\BraTS2021_new",record_path="D:\\Datasets\\BraTS\\BraTS2021_new\\records",seed=None)
    d1 = BraTSDividingWrapper(d0,dividing_rates=(0.7,0.2,0.1))

    d0 = BraTSBasePipeLine(path="D:\\Datasets\\BraTS\\BraTS2021_new",record_path="D:\\Datasets\\BraTS\\BraTS2021_new\\records",seed=None)
    d1 = BraTSDividingWrapper(d0,dividing_rates=(0.7,0.2,0.1))
    d2 = BraTSPatchesWrapper(d1,cut_ranges=((155//2-8,155//2+7),(0,239),(0,239)),patch_sizes=(16,128,128),patch_nums=(1,3,3))

    def gen(g):
        for item in g:
            t1 = item['t1']
            t2 = item['t2']
            t1ce = item['t1ce']
            flair = item['flair']
            mask = item['mask']
            m = item['patch_mask']
            v = item['patch_padding_vector']
            yield {'t1':{'img':t1,
                        'mask':m,
                        'padding_vector':v},
                    't2':{'img':t2,
                        'mask':m,
                        'padding_vector':v},
                    't1ce':{'img':t1ce,
                        'mask':m,
                        'padding_vector':v},
                    'flair':{'img':flair,
                        'mask':m,
                        'padding_vector':v},
                    'mask':{'img':mask,
                        'mask':m,
                        'padding_vector':v},
                    }
    for g1,g2 in zip(d1(),d2()):
        for item,item0 in zip(d2.patch_combine_generator(gen(g2())),g1()):
            assert item['t1']['img'].shape == item['t1']['mask'].shape == (16,240,240)
            assert item['t2']['img'].shape == item['t2']['mask'].shape == (16,240,240)
            assert item['t1ce']['img'].shape == item['t1ce']['mask'].shape == (16,240,240)
            assert item['flair']['img'].shape == item['flair']['mask'].shape == (16,240,240)
            assert item['mask']['img'].shape == item['mask']['mask'].shape == (16,240,240)

            assert np.sum(item['t1']['mask'])==16*240*240
            assert np.sum(item['t2']['mask'])==16*240*240
            assert np.sum(item['t1ce']['mask'])==16*240*240
            assert np.sum(item['flair']['mask'])==16*240*240
            assert np.sum(item['mask']['mask'])==16*240*240

            assert np.isclose(np.mean(item['t1']['img']-item0['t1'][155//2-8:155//2+7+1,0:240,0:240]),0.0)
            assert np.isclose(np.mean(item['t2']['img']-item0['t2'][155//2-8:155//2+7+1,0:240,0:240]),0.0)
            assert np.isclose(np.mean(item['t1ce']['img']-item0['t1ce'][155//2-8:155//2+7+1,0:240,0:240]),0.0)
            assert np.isclose(np.mean(item['flair']['img']-item0['flair'][155//2-8:155//2+7+1,0:240,0:240]),0.0)
            assert np.isclose(np.mean(item['mask']['img']-item0['mask'][155//2-8:155//2+7+1,0:240,0:240]),0.0)


