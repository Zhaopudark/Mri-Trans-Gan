import itertools
import os
import copy
import functools
import operator
import random
import math
from typing import Any,Literal
import numpy as np 
import nibabel as nib

def read_nii_file(path:str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img:nib.Nifti2Image = nib.load(path)
    affine = img.affine
    header = img.header
    img = np.array(img.dataobj[:,:,:],dtype=img.get_data_dtype())
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

# def data_dividing(datas:list[Any],rates:tuple[float,...],random:random.Random|None=None,allow_omit=False)->list[list[Any]]:
#     assert all(x>=0 for x in rates)
#     assert sum(rates)<=1.0
#     data_range = list(range(len(datas)))
#     if random is not None:
#         random.shuffle(data_range)
#     length = len(data_range)
#     end_indexes = []
#     for rate in rates:
#         increase_indexes = round(length*rate)
#         assert increase_indexes>1
#         if not end_indexes:
#             end_indexes.append(increase_indexes)
#         else:
#             end_indexes.append(end_indexes[-1]+increase_indexes)
#     if allow_omit: # obey rates,
#         assert end_indexes[-1]<=length
#     else: # if false, the last dividing rate will be ignored and the last dividing part will contain all remaining elements
#         end_indexes[-1]=length
#     range_buf = []
#     previous_index = 0
#     for end_index in end_indexes:
#         range_buf.append(sorted(data_range[previous_index:end_index]))
#         previous_index = end_index
#     return  [[datas[index] for index in ranges] for ranges in range_buf]

def _get_dividing_nums(N:int,r:list[int|float],basic_func:Literal['round','floor']='round')->list[int]:
    """
    Divide `N` elements into len(`r`) parts, according to rate in `r`, 
    without repetition or omission(leaking),
    making the divided result closest to the scale determind by `r*N`.
    
    Args: 
        N: the total numbers of elements
        r: rates=[r_1,r_2,...,r_n] where r_i represent `i-th` divided part should have about r_i*N elements
        basic_func: `round` or `floor`, two different implementations (the same output), 
                    see https://little-train.com/posts/66b1be3d.html
    Return:
        y = [y_1,y_2,...,y_n], list of integer datas, where y_i represts the `i-th` divided part's 
            element number, i.e., `dividing nums`
    Define a dividing error = (|y_1-r_1*N|^p+|y_2-r_2*N|^p+...+|y_n-r_n*N|^p)^(1/p),
        there should be 
        sum(y)==N and y = argmin(error)
    According to https://little-train.com/posts/66b1be3d.html, the are 2 methods to calulate y
    At first, there should be:
        N>=len(r)>=1
        all(0<rate<=1 for rate in r)
        sum(r)==1.0 
    If basic_func == 'round'
        1. calculate `x` = [x_1,x_2,...,x_n] where x_i = round(r_i*N)-r_i*N
            calculate `y` = [y_1,y_2,...,y_n] where y_i = round(r_i*N) as the esitimated y
        2. get the sorted `ranks`(indices) of x by order from `small to large`,
            `ranks` = [rank_1,rank_2,...,rank_n]
            rank_i means if sort x, x[rank_i] is the i-th elements
        3. get `m` = N -(round(r_1*N)+round(r_2*N)+...+round(r_n*N))
        4. calculate a `bias_list` to modify x and get y
            if m>0 then `bias_list` = [1,1,...1,0,0,...,0], the first |m| elements are 1, the rest are 0 
            if m=0 then `bias_list` = [0,0,...,0]
            if m<0 then `bias_list` = [0,0,...,0,-1,-1,...,-1] , the last |m| elements are -1, the rest are 0 
        5. modify `y` = [y_1,y_2,...,y_n], where y[ranks[i]] = y[ranks[i]]+bias_list[i]
    if basic_func == 'floor':
        1. calculate `x` = [x_1,x_2,...,x_n] where x_i = floor(r_i*N)-r_i*N
            calculate `y` = [y_1,y_2,...,y_n] where y_i = round(r_i*N)
        2. get the sorted `ranks`(indices) of x by order from `small to large`,
            `ranks` = [rank_1,rank_2,...,rank_n]
            rank_i means if sort x, x[rank_i] is the i-th elements
        3. get `m` = N -(floor(r_1*N)+floor(r_2*N)+...+floor(r_n*N) )
        4. calculate a `bias_list` to modify x and get y
            if m>0 then `bias_list` = [1,1,...1,0,0,...,0], the first |m| elements are 1, the rest are 0 
            if m=0 then `bias_list` = [0,0,...,0]
        5. modify `y` = [y_1,y_2,...,y_n], where y[ranks[i]] = y[ranks[i]]+bias_list[i]
    Here the `y` is the target list of integer datas.
    Just slect y_i elements into the `i-th` divided part, we can divide N elements into n parts,
        without repetition or omission, achieving the smallest dividing error. 

    NOTE For determinacy, the sorting function used should be stable.
        Luckily, python's built-in sorted() function is a stable one,
        see https://docs.python.org/3/library/functions.html?highlight=sorted#sorted.

    """
    n = len(r)
    assert all(0<rate<=1 for rate in r)
    assert math.isclose(sum(r),1.0)
    assert 1<=n<=N
    x = [] # buf for index and `estimated-N*rate`
    y = [] # estimated list, y:list[estimated]
    if basic_func == "round":
        for i,rate in enumerate(r):
            estimated = round(N*rate)
            x.append((i,estimated-N*rate))
            y.append(estimated)
        x.sort(key=lambda i_v:i_v[-1])
        ranks = [item[0] for item in x]
        m = N-sum(y)
        bias_list = [1]*abs(m)+[0]*(n-abs(m)) if m>=0 else [0]*(n-abs(m))+[-1]*abs(m)
    elif basic_func == "floor":
        for i,rate in enumerate(r):
            estimated = math.floor(N*rate)
            x.append((i,estimated-N*rate))
            y.append(estimated)
        x.sort(key=lambda i_v:i_v[-1])
        ranks = [item[0] for item in x]
        m = N-sum(y)
        assert m>=0
        bias_list = [1]*m+[0]*(n-m)
    else:
        raise ValueError(f"Unsupported mode:{basic_func}")
    # appliy bias for get each region's length, i.e., modify `y` (esitimated)
    for rank,bias in zip(ranks,bias_list):
        y[rank] += bias
    return y

def datas_dividing(datas:list[Any],rates:list[int|float],seed:int|None=None,mode:Literal['round','floor']='round')->tuple[list[Any],...]:
    """
    Dividing a list of elements into several parts without repetition or omission(leaking), 
        according to rates, to achieve the smallest dividing error as far as possible.
    Args:
        datas: input list of elements that need divide
        rates: list of rate numbers, represtent the dividing rate, rates[i] means the i+1 part will get about 
               len(datas)*rates[i] elements more details see function `_get_dividing_nums`'s arg `r`
        seed: if not `None`, an independent random shuffle with `seed` will be applied to realize random dividing
              NOTE: random is only for "which data in which parts", where the datas in a same parts should matain 
                    the original relative order, we matain this order by operating on indexes-level
        mode: `round` or `floor`, two different implementations (the same output)
               more details see function `_get_dividing_nums`'s arg `basic_func`
    Return:
        tuple of divided datas
        
    For safety, we only operate the copy of datas.
    For envinient, we operate on indexes-level insead of element-level.
        
    >>> datas_dividing(list(range(3)),[0.51,0.24,0.25])
    >>> ([0], [1], [2])
    """
    _datas = copy.deepcopy(datas)
    _datas_indexes = list(range(len(_datas)))
    if seed is not None:
        random_func = random.Random(seed)
        random_func.shuffle(_datas_indexes)
    dividing_nums = _get_dividing_nums(N=len(_datas_indexes),r=rates,basic_func=mode)
    indices_buf = [] # slices buf / indices buf
    begin = 0 
    for num in dividing_nums: # get indices
        indices_buf.append(slice(begin,begin+num,1))
        begin = begin+num
    out_buf = []
    for indices in indices_buf:
        sorted_indices = sorted(operator.getitem(_datas_indexes,indices)) # get selected indexes, sort it
        out_buf.append([_datas[i] for i in sorted_indices]) # get sorted elements by sorted indexes
    return tuple(out_buf)

def alloc_ones_from_center(total:int,length:int,mode:Literal['min_interval','max_interval']):
    assert 0<=total<=length
    if total>=(length/2):
        reverse_flag = True
        total = length-total
    else:
        reverse_flag = False
    buf = [0,]*length
    if total >=1 :
        if total==1: 
            interval = 0
        else:
            interval_buf = [i for i in range(length) if (i*(total-1)+1<=length)and((i+1)*(total-1)+1>length or i*total+1>length)]
            if mode == 'max_interval':
                interval = max(interval_buf)
            elif mode == 'min_interval':
                interval = min(interval_buf)
        inner_length = interval*(total-1)+1
        left_length = (length-inner_length)//2 # left_length == begin_index
        for i in range(total):
            buf[left_length+i*interval] = 1
    return [1-item for item in buf] if reverse_flag else buf



def _get_sub_rigions(total_length:int,sub_region_length:int,overlap_tolerance:tuple[float,float]):
    """
    将一个连续的n元素整数区间S(集合) 例如 S={0,1,2,...,n-1}划分为t个连续的子区间 S1={0,1,2...}, ...,St={...,n-2,n-1}
    要求:
        1. S1 S2 ... St 的并集与 S等价
        2. t 尽可能小, 等价于 overlap 尽可能小
        3. overlap 呈中心对称的均匀排布, 即 
    显然, min_t = ceil(total_length/sub_region_length) 对应于最小的 overlap
    max_t = 
    Dividing a region such as a list [0,1,2,...,total_length-1]
    ] into several sub-ranges and try not to overlap them as far as possible.
    1. Should cover all ranges

    NOTE in this func, right side of all section repesent the actual available one
    if `ranges=(a,b)`, means a closed interval of Integer, which contains numbers of a,a+1,a+2,...,b-1,b

    Args:
        ranges: a tuple represent a closed interval to be divided 
        length: each sub-range's length, i.e., its elements's total numbers
        overlap_tolerance: if a new calculated sub-range has the overlap with existing sub-range that more than `overlap_tolerance`, it will be deprecated. The calculation method see `mode`
        TODO mode: currently, only support `center_first`,
            consider ranges=(a,b)
            get sub-ranges' nums as (b-a+1)//length

    """
    assert total_length>=sub_region_length
    assert (min_overlap_rate:=min(overlap_tolerance))>=0
    max_overlap_rate = max(overlap_tolerance)
    
    min_overlaps = round(sub_region_length*min_overlap_rate)
    max_overlaps = round(sub_region_length*max_overlap_rate)

    min_t = math.ceil(total_length/sub_region_length)
    max_t = total_length-sub_region_length+1 # such as conv's kernel moving procedure
    for t in range(min_t,max_t+1):
        if (min_t*sub_region_length-(min_t-1)*max_overlaps)<=\
            total_length<=\
            (min_t*sub_region_length-(min_t-1)*min_overlaps):
            # can find result 
            nums = total_length - (min_t*sub_region_length-(min_t-1)*max_overlaps)
            # divide nums to such (min_t-1) overlap regions 
            
            overlaps = [max_overlaps-nums//(min_t-1)-decrease for decrease  in alloc_ones_from_center(nums%(min_t-1),(min_t-1),'max_interval') ] # 使最内侧的 overlap regions 优先-1 
            
            buf = [(0,sub_region_length-1)] 
            for single_overlaped in overlaps:
                l = buf[-1][-1]-single_overlaped+1
                r = l+sub_region_length-1
                buf.append((l,r))
            return tuple(buf)   
    return None 



    x*sub_region_length-(x-1)*min_overlaps
    


    if overlap_tolerance is None: # no overlap
        nums, overflow = divmod(total_length, sub_region_length)
        begin = overflow//2



    total_length = ranges[1]-ranges[0]+1
    assert 1<=length<=total_length
    if overlap_tolerance is None:
        nums = total_length//length
        overflow = total_length%length
        begin = overflow//2
        begins = [begin+1+num*length for num in range(nums)]
        
        return tuple((item,item+length-1) for item in begins)
    else:
        
        """
        num+1 is the number of sub-range's 
        num is the total of all overlaped regions among sub-ranges

        Try to find a minimun `num`
            from 
                total_length//length+1 (beacuse if num==total_length//length, there isn't any overlap)
            to 
                total_length-length+1 (beacuse jusr like convolution with stride 1, the most overlap is overlap length-1 elements)
            to achieve an `overlap_rate` in `overlap_tolerance`
        Consider 
            overlaped_i is the length of overlaped regions between each 2 sub-ranges
            overlaped = overlaped_1+overlaped_2+...+overlaped_num
            usually, (num+1)*length-overlaped==total_length cannot be ensured, since they are all integers.
            so we allow the error among overlaped_1 to overlaped_num can be within 1
        So, we try to find a minimun `num` of overlaped_i, with a difference of only 1 to achieve `overlap_tolerance`

        """
        assert overlap_tolerance[0]<=overlap_tolerance[1]
        
        for num in range(total_length//length,(total_length-length+2)): # 
            overlaped = (num+1)*length-total_length
            overlaped_min = overlaped//num
            overlaped_overflow = overlaped%num
            if (overlap_tolerance[0]<=overlaped_min/length)and\
                ((overlaped_min+1)/length<=overlap_tolerance[1]): # find a suitable `num`
                # distribute `overlaped_overflow` into "overlaped_1 to overlaped_num"  with `1` as the smallest unit and with a centrosymmetric manner 
                overlaped = [overlaped_min+maybe_one_or_zero for maybe_one_or_zero in alloc_ones_from_center(overlaped_overflow,num,'min_interval')]
                buf = [(0,length-1)] 
                for single_overlaped in overlaped:
                    l = buf[-1][-1]-single_overlaped+1
                    r = l+length-1
                    buf.append((l,r))
                return tuple(buf)     
        # raise ValueError(f"Cannot find valid subranges within overlap_tolerance:{overlap_tolerance}")
        return None


def get_subranges(total_ranges:tuple[int,int],valid_ranges:tuple[int,int],length:int,overlap_tolerance:tuple[float,float]):
    assert total_ranges[1]>=valid_ranges[1]>=valid_ranges[0]>=total_ranges[0]>=0
    max_total_length = total_ranges[1]-total_ranges[0]+1
    min_total_length = valid_ranges[1]-valid_ranges[0]+1
    for total_length in range(min_total_length,max_total_length+1):
        sub_regions = _get_sub_rigions(total_length,length,overlap_tolerance)
        if sub_regions is not None:
            l = (max_total_length-total_length)//2
            return tuple((region_l+l,region_r+l) for region_l,region_r in sub_regions)

    # buf = itertools.product(range(a+1),range(b+1))

    # totoal
    # for a,b in buf:
        
    #     sub_ranges = get_subranges((0,valid_ranges[1]+b-(valid_ranges[0]-a)),length,overlap_tolerance)
    #     if sub_ranges is not None:
    #         return [[item[0]+valid_ranges[0]-a,item[1]+valid_ranges[0]-a] for item in sub_ranges]
    raise ValueError(f"Cannot find valid subranges within overlap_tolerance:{overlap_tolerance}")

if __name__ == "__main__":
    print(alloc_ones_from_center(0,5,'min_interval'))
    print(alloc_ones_from_center(1,5,'max'))
