
import os
import re 
import ast
import itertools
import functools
import platform
from typing import Callable,Iterable,Any
from typeguard import typechecked

@typechecked
def flatten(items:Iterable):
    for item in items:
        match item:
            case str(item)|bytes(item)|int(item)|float(item)|complex(item):
                yield item
            case item if isinstance(item,Iterable):
                yield from flatten(item)
            case item:
                raise ValueError(f"{item} in unexpected type:{type(item)}.")
@typechecked
def norm_tuple(tuple_like:Any,depth:int=0):
    """
    converted `tuple_like` object to a maybe-nested tuple
    each `Iterable` type in nested structure will be converted
    do not convert first level's `non-Iterable` elements
    such as 
    >>> x = [1,-2,['abc',b'a',[True,6.0,[None,"a"]]],{"a":2},{'a','b'}]
    >>> norm_tuple(x)
    >>> (1, -2, ('abc', b'a', (True, 6.0, (None, 'a'))), ('a',), ('a', 'b'))

    but `non-Iterable` inputs will be converted to a single value tuple
    such as:
    >>> x = True
    >>> norm_tuple(x)
    >>> (True,)
    """
    match tuple_like,depth:
        case x,d if (isinstance(x,(str,bytes))) or (not isinstance(x,Iterable)):
            if d==0:
                return (x,)
            else:
                return x
        case x,d if isinstance(x,Iterable):
            return tuple([norm_tuple(item,d+1) for item in x])
        case x:
            raise ValueError(f"{x} in unexpected type:{type(x)}.")
@typechecked
def get_tuple_from_string(x:str)->tuple:
    return norm_tuple(ast.literal_eval(x))

@typechecked
def reduce_with_map(reduce_func:Callable,inputs:Iterable[Any],map_func:Callable[[Any],Any]=None):
    if map_func is not None:
        def gen(items):
            for item in items:
                yield map_func(item)
        inputs = gen(inputs)
    return functools.reduce(reduce_func,inputs)
    
@typechecked
def reduce_same(inputs:Iterable[Any],map_func:Callable[[Any],Any]=None):
    def _check(x1,x2):
        if x1!=x2:
            raise ValueError(f"Elements in sequence is not the same: `{x1}`!=`{x2}`")
        return x1 
    return reduce_with_map(_check,inputs,map_func=map_func)
    
# @typechecked
# def find_first_target(maybe_iterable,target_type:type=str):
#     if isinstance(maybe_iterable,target_type):
#         return maybe_iterable
#     elif isinstance(maybe_iterable,Iterable):
#         if isinstance(maybe_iterable,dict):
#             for k,v in maybe_iterable.items():
#                 if isinstance(v,target_type):
#                     return v
#                 elif isinstance(v,(dict,set,tuple,list)):
#                     try:
#                         return find_first_target(v,target_type)
#                     except ValueError:
#                         continue
#                 else:
#                     continue
#         else:
#             for item in maybe_iterable:
#                 if isinstance(item,target_type):
#                     return item
#                 elif isinstance(item,(dict,set,tuple,list)):
#                     try:
#                         return find_first_target(item,target_type)
#                     except ValueError:
#                         continue
#                 else:
#                     continue
#     raise ValueError(f"Cannot find target_type {target_type}")

#-------------------functions for deal with nested dict-------------------------#
def _key_sort_func(key:str,order:tuple[str,...]|None):
    """
    key sort function 
    mostly used when sort a dict by key
    mapping key to a specific `order` representation by order

    since `str order` is very convenient, this function will map 
    key to a special `str order` instead of `int order`

    Args:
        key: a str that need to be mapped
        order: `None` or `tuple of str` that indicate the order
    the rule is:
        if order is None, use `str order`, so just map key itself
        if order is tuple, assign priority according to index of order first, then use `str order` for rest keys that  not in  the tuple
    """
    if order is None:
        return key 
    elif key in order:
        return chr(order.index(key))
    else:
        return chr(len(order))+key

@typechecked
def nested_dict_key_sort(maybe_nested_dict:dict,key_orders:Iterable[tuple[str,...]|None],bar:Callable=None)->dict:
    """
    sort a maybe_nested_dict level by level with corresponding key order
    Args:
        maybe_nested_dict: 
            a dict which may be a nested dict, 
            its key in any nested level should be `str` and the innermost nested level's value should be not dict.
            (This sounds weird beacuse if the innermost value is dict, it's not the innermost one)
            The dict structure need not be balanced, since sorting procedure only works when a nested level is a dict but not other type.
            such as 
                {
                    'a':{
                        'a1':'a1_content'
                        'a2':'a2_content'
                    }
                    'b':{

                    }
                    'c':'c_content'
                }
            the 'c_content' will not be sorted since its str not dict.
        key_orders: 
            an Iterable object that specifies each nested level's key order when each iteration. It' iteration length 
            need not be equal to maybe_nested_dict's nested level nums.
            If it's earyl exhausted, `None` will be used for next nested levels.
            If it's still surplus when innermost nested level , the remaining values will be discarded.
        bar: a bar function, called when finish a sorted of maybe_nested_dict's nested level
             If bar function is progress bar drawer function for each iteration, the progress bar will be drawn
    """
    key_orders = iter(key_orders) # if has support next(), iter will return itself
    try:
        current_order = next(key_orders)
    except StopIteration:
        key_orders = iter((None,))
        current_order = next(key_orders)
    
    maybe_nested_dict = dict(sorted(maybe_nested_dict.items(),key=lambda kv:_key_sort_func(kv[0],current_order)))
    if bar is not None:
        bar()
    for (key,value),tee_key_orders in zip(
        maybe_nested_dict.items(),
        itertools.tee(key_orders,len(maybe_nested_dict.keys()))):
        if isinstance(value,dict):
            maybe_nested_dict[key] = nested_dict_key_sort(value,tee_key_orders,bar)
    return maybe_nested_dict


@typechecked
def check_nested_dict(maybe_nested_dict:dict,
                      keys:tuple[None|int|str|tuple[str,...],...],
                      previous_keys:list[str]|list=None,
                      value_func:Callable=None,
                      bar:Callable=None):
    """
    check a maybe_nested_dict's key:value by corresponding keys level by level
    if the key,value is not in a nested level, add them with {} or value_func's return value
    Args:
        maybe_nested_dict: 
            a dict may be in nest structure. 
        keys: 
            a tuple contains each target key, corresponding to each nested level of `maybe_nested_dict`.
            Here, len(keys) should be equal to maybe_nested_dict's nested level nums, beacuse we should know if it's the innermost nested level and should add last value instead of add another dict to nest
            TODO rectify potential errors when len(keys) is not equal to maybe_nested_dict's nested level nums
            The maybe_nested_dict need not be balanced, i.g.,
                {
                    'a':{
                        'a1':'a1_content'
                        'a2':'a2_content'
                    }
                    'b':{

                    }
                    'c':'c_content'
                } 
                can also be checked.
            The specific rule is:
                for key in keys,
                if key is str:
                    check the key whether in current nested level
                        If key not in current nested level, new `key:value` will be added.
                        If current nested level is the innermost one, the added value will be value_func's return value.
                        If current nested level is not the innermost one, the value will be a {}, means add a empty dict as next nested level.
                if key is int or None:
                    check all key:value 
                    if current nested level is just a empty dict, add "%placeholder%":value to it 
                    the value is determinded by above rules.
                if key is tuple of str
                    check each str of key by above rules
            So, unbalanced maybe_nested_dict does not influence the check beahvior, beacuse 
                "%placeholder%":value 
                "%placeholder%":{}
                'key':{}
                'key':value
                may be used to patch up the missing parts.            
        previous_keys: record previous keys, used by value_func and recursive
        value_func: a function generate a value for the innermost nested level's key. The value must not be a dict, otherwise itdoes not belong to the the innermost nested level.
        bar: a bar function, called when finish a single key:value check of maybe_nested_dict
             If bar function is progress bar drawer function for each iteration, the progress bar will be drawn
    """
    current_key = keys[0]
    not_innermost_nested = len(keys)>1
    if previous_keys is None:
        previous_keys = []
    match current_key:
        case str(key):
            if not_innermost_nested:
                if key not in maybe_nested_dict:
                    maybe_nested_dict[key] = {} # checked and add {}
                if isinstance(maybe_nested_dict[key],dict): 
                    # next check only if the next is dict ()
                    # so here can deal with not balanced `maybe_nested_dict`
                    next_keys = keys[1:]
                    check_nested_dict(maybe_nested_dict[key],next_keys,previous_keys[:]+[key],value_func,bar)
                if bar is not None:
                    bar()
            else:
                if key not in maybe_nested_dict:
                    maybe_nested_dict[key] = value_func(tuple(previous_keys[:]+[key])) # checked and add last value
                assert not isinstance(maybe_nested_dict[key],dict) # preserve the nest dict architecture
        case tuple(key)| int(key) | (None as key): # deal with by the above str() case through recursive invocation
            if not isinstance(key,tuple):# int | None
                current_key_list = list(maybe_nested_dict.keys()) if maybe_nested_dict else ["%placeholder%"]
            else:#tuple[str,...]
                current_key_list = list(key)
            for current_key in current_key_list:
                tmp_key = tuple([current_key]+list(keys[1:])) if not_innermost_nested else (current_key,)  
                check_nested_dict(maybe_nested_dict,tmp_key,previous_keys[:],value_func,bar)


@typechecked
def gen_key_value_from_nested_dict(
        maybe_nested_dict:dict,
        keys:tuple[None|int|str|tuple[str,...],...],
        previous_keys:list[str]|list=None,):
    """
    yield key:value of a maybe_nested_dict by corresponding keys level by level
    
    value is the innermost nested level's value
    key is the tuple of keys to locate the value
    for example 
        (k1,k2,k3,k4):v
        means at least a 4-level-nested dict is existed in maybe_nested_dict
        {
            k1:{
                k2:{
                    k3:{
                        k4:v
                    }
                }
            }
        }
        when input keys = (k1,k2,k3,k4)
        yield (k1,k2,k3,k4),v

    Args:
        maybe_nested_dict: a dict may be in nest structure
        keys: a tuple contains each target key, corresponding to each nested level of `maybe_nested_dict`.
            The specific rule is:
                for key in keys,
                if key is str:
                    the value in key:value may be still a nested structure or the innermost nested level 
                    if here is the innermost nested level, means the value in key:value is not a dict
                        yeild the value 
                    if the value in key:value is still a nested structure (dict):
                        yield from the value by the same relu
                if key is int:
                    yield from the first `int(key)` values by above relu, 
                    if `int(key)` more than exist key nums, yield from all values by above relu.
                if key is None:
                    yield from all values by above relu.
                if key is tuple of str
                    yield from each key's vaule by above rules
                So, unbalanced maybe_nested_dict does not influence the yield beahvior, beacuse 
                empty dict will be ignored with yield anything

    """
    current_key = keys[0]
    not_innermost_nested = len(keys)>1
    if previous_keys is None:
        previous_keys = []
    match current_key:
        case str(key):
            assert key in maybe_nested_dict
            if not_innermost_nested:
                if isinstance(maybe_nested_dict[key],dict): 
                    # yield only if the next is dict ()
                    # so here can deal with not balanced `maybe_nested_dict`
                    next_keys = keys[1:]
                    yield from gen_key_value_from_nested_dict(maybe_nested_dict[key],next_keys,previous_keys[:]+[key])
            else:
                assert not isinstance(maybe_nested_dict[key],dict) # preserve the nest dict architecture
                yield tuple(previous_keys[:]+[key]),maybe_nested_dict[key] 
        case tuple(key) | int(key) | (None as key): # deal with by the above str() case through recursive invocation
            if not isinstance(key,tuple):# int | None
                current_key_list =  list(maybe_nested_dict.keys())[0:abs(key) if isinstance(key,int) else len(maybe_nested_dict.keys())]
                 #empty dict can also be supported
            else:#tuple[str,...]
                current_key_list = list(key)
            for current_key in current_key_list:
                tmp_key = tuple([current_key]+list(keys[1:])) if not_innermost_nested else (current_key,)  
                yield from gen_key_value_from_nested_dict(maybe_nested_dict,tmp_key,previous_keys[:])

@typechecked
def dict_flatten(maybe_nested_dict:dict[str,Any],key_separator:str='|'):
    buf = []
    for key,value in maybe_nested_dict.items():
        if isinstance(value,dict):
            buf.extend((f"{key}{key_separator}{inner_key}",inner_value) for inner_key,inner_value in dict_flatten(value,key_separator))
        else:
            buf.append((key,value))
    return buf 

def _nested_dict_k_v_assign(maybe_nested_dict:dict[str,Any],keys:tuple[str,...]=None,value:Any=None):
    key = keys[0]
    if len(keys)>1:
        buf = maybe_nested_dict.get(key, {})
        _nested_dict_k_v_assign(buf,keys=keys[1::],value=value)
        maybe_nested_dict[key] = buf
    else:
        maybe_nested_dict[key] = value

@typechecked
def dict_flatten_reverse(key_value:list[tuple[str,Any]],buf_dict:dict=None,key_separator:str='|'):
    if buf_dict is None:
        buf_dict = {}
    for key,value in key_value:
        keys = tuple(key.split(key_separator))
        _nested_dict_k_v_assign(maybe_nested_dict=buf_dict,keys=keys,value=value)
    return buf_dict
# @typechecked
# def gen_values_from_nested_dict(maybe_nested_dict:dict,keys:tuple[None|int|str|tuple[str,...],...],previous_keys:list[str]|list=None,value_func:Callable=None):
#     # return any number of values of a n-level-nested dict by keys, as long as the value's key are met
#     # when store, keys should be str
#     # when reading, key can be None(all keys in this level are met) or int(first n keys in this level are met)
#     # if key not in dict, new sub dict will be added and the final value will be generated by value_func
#     current_key = keys[0]
#     if previous_keys is None:
#         previous_keys = []
#     if isinstance(current_key,str):
#         if current_key in maybe_nested_dict:
#             if len(keys)>1:
#                 next_keys = keys[1:]
#                 yield from gen_values_from_nested_dict(maybe_nested_dict[current_key],next_keys,previous_keys[:]+[current_key],value_func)
#             else:
#                 yield maybe_nested_dict[current_key]
#         elif len(keys)>1:
#                 next_keys = keys[1:]
#                 maybe_nested_dict[current_key] = {}
#                 yield from gen_values_from_nested_dict(maybe_nested_dict[current_key],next_keys,previous_keys[:]+[current_key],value_func)
#         else:       
#             maybe_nested_dict[current_key] = value_func(previous_keys[:]+[current_key])
#             yield maybe_nested_dict[current_key]
#     elif isinstance(current_key,int) or current_key is None:
#         if current_key is None:
#             current_key = len(maybe_nested_dict.keys())
#             if current_key==0:
#                 logging.getLogger(__name__).warning(f"`None` in key of `{tuple(list(previous_keys[:])+list(keys[:]))}` do not have any items.")
#         for i,(key,value) in enumerate(maybe_nested_dict.items()):
#             if i+1 > current_key:
#                 break
#             if len(keys)>1:
#                 next_keys = keys[1:]
#                 yield from gen_values_from_nested_dict(value,next_keys,previous_keys[:]+[key],value_func)
#             else:
#                 yield maybe_nested_dict[key]
#     else:
#         for key in current_key: # manually product
#             tmp_key = tuple([key]+list(keys[1:])) if len(keys)>1 else (key,)
#             yield from gen_values_from_nested_dict(maybe_nested_dict,tmp_key,previous_keys[:],value_func)