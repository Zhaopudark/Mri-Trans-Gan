import random
import logging

def random_datas(datas:list|dict[str,list],random:random.Random|None=None):
    if random is not None:
        if isinstance(datas,dict):
            state = random.getstate()
            for key in datas:
                random.setstate(state)
                random.shuffle(datas[key])
        else:
            random.shuffle(datas) # since datas has been shuffled, the next shuffle will not be the same
        logging.getLogger(__name__).info("Random complete!")
    return datas

def get_random_from_seed(seed:int|None=None):
    return random.Random(seed) if seed is not None else None
