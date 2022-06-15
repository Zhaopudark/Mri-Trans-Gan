
import functools
import logging
from typing import Callable
from alive_progress import alive_bar
from typeguard import typechecked
from tensorflow.keras.utils import Progbar
import inspect
from inspect import Parameter

@typechecked
def func_bar_injector(func:Callable=None,total_number:int=None,bar_name:str='UnknownBar'):
    if func is None:
        return functools.partial(func_bar_injector,total_number=total_number,bar_name=bar_name)
    bar_param = Parameter('bar',Parameter.KEYWORD_ONLY,default=None,annotation=Callable)
    parameters = inspect.signature(func).parameters
    if ('bar' not in parameters):
        logging.getLogger(__name__).warning(f"{func.__name__} does not have kw `bar` ,this func will not be injected by `bar` drawer.")
        return func
    elif str(parameters['bar']) != str(bar_param):
        logging.getLogger(__name__).warning(f"{func.__name__}'s `bar` is not `{bar_param}`, this func will not be injected by `bar` drawer.")
        return func
    _bar = Progbar(total_number,width=30,verbose=1,interval=0.5,stateful_metrics=None,unit_name=bar_name)
    _bar = functools.partial(_bar.add,1)

    @functools.wraps(func)
    def wrapped(*args,**kwargs):
        return func(*args,**(kwargs|{'bar':_bar}))
    return wrapped
    # parameters['bar'].replace(default=_bar)

    # binded_args = inspect.signature(func).bind_partial(bar=_bar) # No need!!!
    # return functools.partial(func,*binded_args.args,**binded_args.kwargs)
    # return functools.partial(func,bar=_bar)

# with alive_bar(ctrl_c=False, title='Checking data paths before generation:',) as bar:  
        #     check_nested_dict(self.data,keys,previous_keys=[],value_func=self._map_keys2path,bar=bar)
        #     self.data = nested_dict_key_sort(self.data,BraTSBase.KEY_ORDERS.value,bar)
