import copy
import collections
from typing import Iterable,Mapping,Any,Callable

from .process_dtype import order_mapping

class ListContainer(collections.UserList):
    def __init__(self,data) -> None:
        if isinstance(data,ListContainer):
            super().__init__(data.data)
        elif isinstance(data,list): # 可能用户本就希望存储该列表 为了保持其完整性 打包后存储
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

    def serialize(self): # return a copy for safety
        # return serialized type of self.data[key] (which can be eval() to original type)
        return {key: self.data[key].serialize() if hasattr(self.data[key], 'serialize') 
                else copy.deepcopy(self.data[key]) 
                for key in self.data}
 
class NestedDict(collections.UserDict):
    """
    我应当完善这个NestedDict 
    提供
        __init__
        append
        get_items
        sort()

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
    
    def get_items(self,
        superior_keys=None,maybe_tuple_key:str|tuple[str|tuple[str,...],...]=None,
        estimate_func:Callable[[tuple[str|tuple[str,...],...]],Any]=None):
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
        self.data = dict(sorted(self.items(),key=lambda kv:order_mapping(kv[0],order=current_order)))
