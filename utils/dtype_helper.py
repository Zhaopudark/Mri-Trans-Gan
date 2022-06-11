
import ast
import copy
import itertools
import functools
import collections
from typing import Callable,Iterable,Any, Mapping
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
def get_tuple_from_str(x:str)->tuple:
    try:
        value = ast.literal_eval(x)
        return norm_tuple(value)
    except ValueError:
        return norm_tuple(x)


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
    

#-------------------functions for deal with nested dict-------------------------#
def key_sort_func(key:str,order:tuple[str,...]|None):
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

# @typechecked
def nested_dict_key_sort(maybe_nested_dict:dict,key_orders:Iterable[tuple[str,...]|None])->dict:
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
    
    maybe_nested_dict = dict(sorted(maybe_nested_dict.items(),key=lambda kv:key_sort_func(kv[0],current_order)))
    for (key,value),tee_key_orders in zip(
        maybe_nested_dict.items(),
        itertools.tee(key_orders,len(maybe_nested_dict.keys()))):
        if isinstance(value,dict):
            maybe_nested_dict[key] = nested_dict_key_sort(value,tee_key_orders)
    return maybe_nested_dict


# @typechecked
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


# @typechecked
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

#--------------------------------------------------------------------------------------------------------------#
class ListContainer(collections.UserList):
    def __init__(self,data) -> None:
        if isinstance(data,ListContainer):
            super().__init__(data.data)
        elif isinstance(data,list): # 可能用户本就希望存储该列表 为了保持其完整性 打包成元组后存储
            super().__init__(data)
        else:
            super().__init__([data])
    def get_norm(self): # return a copy for safety
        # return most original data (non-user-defined) type (str,dict,list...)
        return self.data[:]
    def serialize(self): # return a copy for safety
        # return each serialized type of  element in self.data (which can be eval() to original type)
        # return [f"\'{item}\'" for item in self.data]
        buf = []
        for item in self.data:
            if isinstance(item,str):
                buf.append(f"\'{item}\'")
            else:
                buf.append(f"{item}")
        return buf
 

#--------------------------------------------------------------------------------------------------------------#
class DictList(collections.UserDict):
    def __init__(self,initial_data) -> None:
        if isinstance(initial_data,DictList):
            self.data = initial_data.data
        elif isinstance(initial_data,dict):
            self.data = initial_data
            for key in self.data:
                self.data[key] = ListContainer(self.data[key])
        else:
            raise ValueError(" ")
    @staticmethod
    def __reduce_out_place(dict1:dict,dict2:dict): # not in-place
        return {key:dict1[key]+dict2[key] for key in dict2 if key in dict1}
    @staticmethod
    def __broadcast_out_place(data:dict,info): # not in-place
        buf = {}
        for key in data:
            assert len(data[key].data)==1
            buf[key] = ListContainer((data[key].data[0],info))
        return buf
        # return {key:data[key]+ListContainer(info) for key in data}
    def __add__(self,other): # should not be in-place
        if isinstance(other,self.__class__):
            return self.__class__(self.__reduce_out_place(self.data,other.data))
        else:
            return self.__class__(self.__broadcast_out_place(self.data,other))
    def __radd__(self, other):
        raise NotImplementedError
    def __iadd__(self, other):
        raise NotImplementedError
    # def get_norm(self): # return a copy for safety
    #     # return most original data (non-user-defined) type (str,dict,list...)
    #     return {key: self.data[key].get_norm() if hasattr(self.data[key], 'get_norm') 
    #             else copy.deepcopy(self.data[key]) 
    #             for key in self.data}
    def serialize(self): # return a copy for safety
        # return serialized type of self.data[key] (which can be eval() to original type)
        return {key: self.data[key].serialize() if hasattr(self.data[key], 'serialize') 
                else copy.deepcopy(self.data[key]) 
                for key in self.data}
 
class NestedDict(collections.UserDict):
    """
    {
        names:{keys:values}
    }
    dict[str,dict[str,...dict[str,str]]]
    """
    __slots__=["data"]
    class _MetaData(collections.UserString):...
    
    def _norm_tuple(self,maybe_tuple:str|tuple[str,...])->tuple:
        if isinstance(maybe_tuple,tuple):
            return maybe_tuple
        # try:
        #     value = ast.literal_eval(maybe_tuple)
        #     return value if isinstance(value,tuple) else (value, )
        # except ValueError:
        return (maybe_tuple,) 
    def append(self,arg:Iterable|Mapping|str|tuple[str,...],item=None):
        match arg,item:
            case _mapping,None if isinstance(_mapping,Mapping):
                for key,value in _mapping.items():
                    if key in self: # not first add
                        # if isinstance(value,Mapping):
                        if (isinstance(self[key],self.__class__))and(isinstance(value,Mapping)): # not leaf node
                            self[key].append(value,None)
                        elif isinstance(self[key],self.__class__):
                            pass # 提前耗尽则不更新
                        elif isinstance(value,Mapping):
                            self[key].append(value,None)
                        else:
                            self[key] = self[key]+value
                    else: # first add
                        self[key] = self.__class__().append(value,None) if isinstance(value,Mapping) else self.__class__._MetaData(value)
            case _iterable,None if isinstance(_iterable,Iterable):
                for key,value in _iterable:
                    self.append(key,value) 
            case (current_key,*next_keys),value if value is not None:
                if current_key in self: # not first add
                    if isinstance(self[current_key],self.__class__): # not leaf node
                        self[current_key].append(tuple(next_keys),value)
                    else:# extend leaf node's current value
                        self[current_key] = self[current_key]+value
                else: # extend a leaf node with a tree
                    self[current_key] = self.__class__().append(tuple(next_keys),value) if next_keys else self.__class__._MetaData(value)      
                 
            case str(key),str(value):
                self[key] = self.__class__._MetaData(value)
            case unmatched:
                raise ValueError(f"unmatched: {unmatched}") # TODO
        return self   
    def _check_leaf(self,key):
        try:
            return isinstance(super().__getitem__(key),self.__class__)
        except KeyError:
            return False
        
    def __getitem__(self,maybe_tuple_key:str|tuple[str,...])->dict|list|Any:
        keys = self._norm_tuple(maybe_tuple_key)
        current_key = keys[0]
        if self._check_leaf(current_key) and (len(keys)>1):
            return super().__getitem__(current_key)[keys[1:]]
        else:
            return super().__getitem__(current_key)
    
    def get_items(self,superior_keys=None,maybe_tuple_key:str|tuple[str|tuple[str,...],...]=None,estimate_func:Callable[[tuple[str|tuple[str,...],...]],Any]=None):
        if superior_keys is None:
            superior_keys = []
        else:
            superior_keys = list(superior_keys)
        keys = self._norm_tuple(maybe_tuple_key)
        match keys[0]:
            case str(current_key):
                current_keys = [current_key]
            case tuple(current_key):
                current_keys = current_key
            case None:
                current_keys = list(self.keys())
            case int(current_key):
                if current_key<0:
                    _slice = slice(-current_key,0,-1)
                else:
                    _slice = slice(0,current_key,1)
                current_keys = list(self.keys())[_slice]
            case _:
                raise ValueError(" ")
        for current_key in current_keys:
            if self._check_leaf(current_key) and (len(keys)>1):
                yield from self[current_key].get_items(tuple(superior_keys+[current_key]),keys[1:],estimate_func)
            else:
                try:
                    yield tuple(superior_keys+[current_key]),self[current_key].data
                except KeyError:
                    if estimate_func is None:
                        pass # do not yield the data that not involved (append)
                    else:
                        # estimate the data by keys
                        estimated = estimate_func(tuple(superior_keys+[current_key]))
                        yield tuple(superior_keys+[current_key]),self.__class__._MetaData(estimated).data

    # -------------------functions for deal with nested dict-------------------------#
    # @typechecked
    def _key_sort_func(self,key:str,order:tuple[str,...]|None):
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
    # @typechecked
    def sort(self,key_orders:tuple[tuple[str,...]|None]|None)->None:
        """
        sort the maybe_nested_dict level by level with corresponding key order
        Args:
            key_orders: 
                an Iterable object that specifies each nested level's key order when each iteration. It' iteration length 
                need not be equal to maybe_nested_dict's nested level nums.
                If it's earyl exhausted, `None` will be used for next nested levels.
                If it's still surplus when innermost nested level , the remaining values will be discarded.
        """
        match key_orders:
            case None:
                current_order = None
                next_order = None
            case tuple(orders):
                current_order = orders[0]
                next_order = orders[1:] if len(orders)>1 else None
        for key in self.keys():
            if isinstance(self[key],NestedDict):
                self[key].sort(next_order) 
        self.data = dict(sorted(self.items(),key=lambda kv:self._key_sort_func(kv[0],current_order)))