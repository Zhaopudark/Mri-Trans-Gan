def add_to(num, target=[]):
    target.append(num)
    return target
target = []
print(add_to(1,target))
print(add_to(2,target))
print(add_to(3,target))


def add_to(num, target=()):
    target[str(num)]=num
    return target
target = {}
print(add_to(1))
print(add_to(2))
print(add_to(3))

def add_to(num, target={}):
    target[str(num)]=num
    return target
target = {}
print(add_to(1))
print(add_to(2))
print(add_to(3))

def add_to(num, target={}):
    target[str(num)]=num
    return target
target = {}
print(add_to(1,target))
print(add_to(2,target))
print(add_to(3,target))
