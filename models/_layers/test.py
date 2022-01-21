def decorator(num):
    print(num)
    def dec2(cls):
        print(cls)
        return cls
    return dec2

def decorator2(cls):
    print(cls)
    return cls
    
@decorator(1)
class Model(object):
    test_val = 0
    def __init__(self):
        pass

@decorator2
class SubModel(Model):
    def __init__(self):
        pass

# if __name__ == '__main__':
    # model = SubModel()