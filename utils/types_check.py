import builtins
__all__ = [ 
    'type_check',
]
def _get_inner(x,type):
    if type==list:
        return x[0]
    elif type==tuple:
        return x[0]
    elif type==dict:
        return list(x.values())[0]
    else:
        raise ValueError(f"Unsupported type {type}")
def type_check(x,mode,meta_instance_or_type=None):#
    """
    判断x是否是基于mode列表中的list tuple dict将最小元进行嵌套的组织形式 是则返回true 否则返回false 
    当最小元未指定时 只需要x可以依据mode列表拆解即可返回true
    当最小元指定时 不仅需要x可以依据mode列表拆解 也需要最后的拆解得到的最小元与meta_instance_or_type指定的类型一致才可返回true
    为了方便 meta_instance_or_type 可以接收实例或者类名皆可
    """
    if not isinstance(mode,list):
        mode = [mode]
    if len(mode)>=2:
        if type(x)==mode[0]:
            _x = _get_inner(x,mode[0])
            return type_check(_x,mode[1::],meta_instance_or_type)
        else:
            return False
    elif len(mode)==1:
        if type(x)==mode[0]:
            if meta_instance_or_type is not None:
                _x = _get_inner(x,mode[0])
                if type(meta_instance_or_type)==type:
                    # 给定的元一个类型 可以直接使用
                    meta_type = meta_instance_or_type
                else:
                    # 给的是一个instance
                    meta_type = type(meta_instance_or_type)
                if type(_x) == meta_type:
                    return True
                else:
                    return False
            else: #不考虑最小元的类型
                return True
        else:
            return False
    else:
        raise ValueError(f"Unsupported mode {mode}")
#--------------------------------------------------------------#
def _parse_spec(str_spec):
    return [getattr(builtins, x) for x in str_spec.rstrip('>').split('<')]
def _parse_spec2(str_spec):
    return [x for x in str_spec]
def _validate(value, spec):
    for t in spec:
        if not isinstance(value, t):
            return False
        try:
            value = value[0] # 没用 不适用于与字典的混合嵌套
        except TypeError:
            pass
    return True
#-------------------------------------------------------------#
if __name__ =='__main__':
    dic = {'b1':{'a1':1,'a2':2,'a3':3},'b2':{'a1':4,'a2':5,'a3':6}}
    dic2 = [{'a1':1,'a2':2,'a3':3},{'a1':4,'a2':5,'a3':6}]
    dic3 = {'b1':[1,2,3],'b2':[4,5,6]}
    import numpy as np 
    a = np.array(0)
    print(type_check(dic,[dict]))
    print(type_check(dic,[dict,dict,dict,dict]))
    print(type_check(dic2,[list,dict]))
    print(type_check(dic3,[dict,list]))
    print(type_check(dic,[dict],meta_instance_or_type=dict))
    print(type_check(dic,[dict],meta_instance_or_type={}))
    print(type_check(dic,[dict],meta_instance_or_type=()))
    print(type_check(dic,[dict],meta_instance_or_type=set()))
    print(type_check(dic,[dict],meta_instance_or_type=[]))
    dic = {'b1':{'a1':a+1,'a2':a+2,'a3':a+3},'b2':{'a1':a+4,'a2':a+5,'a3':a+6}}
    print(type_check(dic,[dict,dict],meta_instance_or_type=a+1))
    print(type_check(dic,[dict],meta_instance_or_type=a+1))
    print(type_check(dic,[dict],meta_instance_or_type=np.int32))
    print(type_check(dic,[dict,dict],meta_instance_or_type=np.int32))
    import random
    a = [1,2,3,4,5]
    b = [1,2,3,4,5]
    random.seed(1)
    random.shuffle(a)
    print(a)
    random.seed(1)
    random.shuffle(a)
    print(a)
    random.seed(1)
    random.shuffle(a)
    print(a)
    print(_validate([10], _parse_spec('list<int>')))
    print(_validate([10], _parse_spec('int')))
    print(_validate([['something']], _parse_spec('list<list<str>>')))
    print(_validate([10], _parse_spec2([list,int])))
    print(_validate([10], _parse_spec2([int])))
    print(_validate([['something']], _parse_spec2([list,list,str])))


