import ast
from typing import Any,Literal,Iterable

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


def dict_flatten_reverse(key_value:list[tuple[str,Any]],buf_dict:dict=None,key_separator:str='|'):
    if buf_dict is None:
        buf_dict = {}
    for key,value in key_value:
        keys = tuple(key.split(key_separator))
        _nested_dict_k_v_assign(maybe_nested_dict=buf_dict,keys=keys,value=value)
    return buf_dict

def order_mapping(key:Any,order:tuple[Any,...]|None=None,mode:Literal['key_order_first','alphabetical_first']="key_order_first"):
    """
    if 
    observe the index order in `order`
    mapping key to a specific `order`:
        1. observe order first
        2. if order is None, use alphabetical order

    since `str order` is very convenient, this function will map 
    key to a special `str order` instead of `int order`
    
    Args:
        key: a str that need to be mapped
        order: `None` or `tuple of str` that indicate the order
        mode: indicate the order_mapping mode,
            if `key_order_first`, observe `key`'s index order in `order`, then alphabetical order if `order` is None or does not contain `key`
            if `alphabetical_first`, observe `key`'s alphabetical order (transfer to str)
    """
    if mode == 'key_order_first':
        if order is None:
            return str(key) 
        elif key in order:
            return chr(order.index(key))
        else:
            return chr(len(order))+str(key)
    elif mode == 'alphabetical_first':
        return str(key)
    else:
        raise ValueError(f"mode:{mode} should in {['key_order_first','alphabetical_first']}")

def flatten(items:Iterable):
    for item in items:
        match item:
            case str(item)|bytes(item)|int(item)|float(item)|complex(item):
                yield item
            case item if isinstance(item,Iterable):
                yield from flatten(item)
            case item:
                raise ValueError(f"{item} in unexpected type:{type(item)}.")

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

def get_tuple_from_str(x:str)->tuple:
    try:
        value = ast.literal_eval(x)
        return norm_tuple(value)
    except ValueError:
        return norm_tuple(x)