"""
将若干通用性的2D 3D图片处理仍无放到这里。
提供 图片文件的读写方法实现
但是"写"方法不在此处运行 而是被模型或logs调用
"""
import os 
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image
import tensorflow as tf
import numpy as np 
import io
from collections import Iterable
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from utils.types_check import type_check
__all__ = [
    "ImageSaver",
    "Drawer",
]
class ImageSaver():
    def __init__(self,path) -> None:
        self._path = path
    def saving(self,img_dict):
        """
        {path1:img1
         path2:img2
         path3:img3
         ...}
        """
        if type_check(img_dict,[dict]):
            for path,img in img_dict.items():
                im = Image.fromarray(img[0,:,:,:].numpy())
                if self._path[-1]=="\\" or self._path[-1]=="/":
                    tmp_path = self._path[0:-1]
                else:
                    tmp_path = self._path[::]
                if os.path.exists(tmp_path):
                    pass
                else:
                    os.makedirs(tmp_path)
                if path[0]=="\\" or self._path[0]=="/":
                    tmp_path = tmp_path+path
                else:
                    tmp_path = tmp_path+"/"+path
                im.save(tmp_path+".png")
        else:
            raise ValueError("img_dict must be a dict with depth=1")

class Drawer():
    def __init__(self):
        pass
    def _dict2img(self,imgs:dict):
        img_num = len(imgs.keys())
        sub_window_w_num = math.ceil(math.sqrt(float(img_num)))
        sub_window_h_num = sub_window_w_num
        pic_scale = sub_window_w_num*2
        plt.figure(figsize=(pic_scale,pic_scale)) # 图片大一点才可以承载像素
        for i,(img_name,img) in enumerate(imgs.items()):
            plt.subplot(sub_window_w_num,sub_window_h_num,i+1)
            plt.title(img_name)
            plt.imshow(img,cmap='gray') # TODO
            plt.axis('off')
        buf = io.BytesIO() # 在内存中读写 方便很多
        plt.savefig(buf,format="png") # TODO
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        img = np.array(img)
        img_shape = img.shape
        img = tf.reshape(img,shape=(1,img_shape[0],img_shape[1],img_shape[2]))# fig size 一致
        # cv2.imshow('image', img[0,:,:,:].numpy())
        # cv2.waitKey(0)
        buf.close()
        return img #B H W C形式
    def draw_from_dict(self,img_dict,process_func=lambda x:x):
        """
        根据传入的字典 进行绘图。
        要求字典格式为2重字典
        {path1:{img_name1:img1,img_name2:img2,...,}
         path2:{img_name1:img1,img_name2:img2,...,}
         path3:{img_name1:img1,img_name2:img2,...,}
         ...}
        返回绘图后的字典 
        {path1:img1
         path2:img2
         path3:img3
         ...}
        """
        if type_check(img_dict,[dict,dict]):
            buf_0 = {}
            for path,imgs in img_dict.items():
                buf_1 = {}
                for img_name,img in imgs.items():
                    buf_1[img_name] = process_func(img)
                buf_0[path] = buf_1
            buf_2 = {}
            for path,imgs in buf_0.items():
                buf_2[path] =  self._dict2img(imgs)
            return buf_2
        else:
            raise ValueError("img_dict must be a dict(dict)")
    