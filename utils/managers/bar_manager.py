
import functools
import imp
import logging
import inspect
from inspect import Parameter
from typing import Callable, Literal
from typeguard import typechecked

# bar backend
import alive_progress as alive_progress
import tensorflow.keras as keras

@typechecked
def func_bar_injector(func:Callable=None,total:int=None,title:str='UnknownBar',backend:Literal['keras','alive_progress']='alive_progress'):
    if func is None:
        return functools.partial(func_bar_injector,total=total,title=title,backend=backend)
    bar_param = Parameter('bar',Parameter.KEYWORD_ONLY,default=None,annotation=Callable)
    parameters = inspect.signature(func).parameters
    if ('bar' not in parameters):
        logging.getLogger(__name__).warning(f"{func.__name__} does not have kw `bar` ,this func will not be injected by `bar` drawer.")
        return func
    elif str(parameters['bar']) != str(bar_param):
        logging.getLogger(__name__).warning(f"{func.__name__}'s `bar` is not `{bar_param}`, this func will not be injected by `bar` drawer.")
        return func
    else:
        pass
    if backend == 'keras':
        @functools.wraps(func)
        def wrapped(*args,**kwargs):
            _bar = keras.utils.Progbar(total,width=30,verbose=1,interval=0.5,stateful_metrics=None,unit_name=title)
            _bar = functools.partial(_bar.add,1)
            return func(*args,**(kwargs|{'bar':_bar}))
        return wrapped
    elif backend == 'alive_progress':
        @functools.wraps(func)
        def wrapped(*args,**kwargs):
            with alive_progress.alive_bar(total=total,ctrl_c=False,title=title) as _bar: 
                return func(*args,**(kwargs|{'bar':_bar}))
        return wrapped
    else:
        raise ValueError(f"The backend should be `keras` or `alive_bar`, not {backend}")
    

    

# @typechecked
# def alive_bar_configue(total=None,title:str='alive_bar'):
#     return alive_bar(total=total,ctrl_c=False,title=total)
#     # parameters['bar'].replace(default=_bar)

#     # binded_args = inspect.signature(func).bind_partial(bar=_bar) # No need!!!
#     # return functools.partial(func,*binded_args.args,**binded_args.kwargs)
#     # return functools.partial(func,bar=_bar)

# with alive_bar(ctrl_c=False, title='Checking data paths before generation:',) as bar:  
#             check_nested_dict(self.data,keys,previous_keys=[],value_func=self._map_keys2path,bar=bar)
#             self.data = nested_dict_key_sort(self.data,BraTSBase.KEY_ORDERS.value,bar)