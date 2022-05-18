import numpy as np 
import tempfile
from operator import itemgetter

a = np.random.normal(size=[1,2,3])
b = np.random.normal(size=[1,2,3])
c = np.random.normal(size=[1,2,3])

with tempfile.NamedTemporaryFile(suffix=".npz") as fp:
    print(fp.name)
    np.savez(fp,a=a,b=b,c=c)
    result = np.load(fp)
    print(result.files)
    x = itemgetter(*('a','b','c'))
    print(x(result))
    np.savez(fp,d=c)
    result = np.load(fp)
    print(result.files)

f = np.load("D:\\Datasets\\BraTS\\BraTS2021_new\\records5\\Training_BraTS2021_00002.npz")
for item in f.files:
    print(item,f[item].shape)