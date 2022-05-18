import sys
import os
from models import _mri_trans_gan_model
__all__ = [ 
    'ModelSelector',
]
class ModelSelector():
    def __init__(self,args):
        self.__model_name = args.model_name.lower()
        if self.__model_name == 'mri_trans_gan':
            # self.__model = _mri_trans_gan_model.MriTransGan
            self.__model = _mri_trans_gan_model.MriTransGan
            # self.__model = _mri_trans_gan_model_modified.MriTransGan
        else:
            raise ValueError("Unsupported model!")
    
    def model(self,*agrs,**kwargs):
        return self.__model(*agrs,**kwargs)
    