import sys
import os
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)
sys.path.append(os.path.join(base,'../'))
from networks import mri_trans_gan,mri_trans_gan_v2,mri_trans_gan_v3,mri_trans_gan_v4
__all__ = [ 
    "NetworkSelector"
]
class NetworkSelector():
    def __init__(self,args):
        self._architectures = {}
        _architecture_name = args.architecture_name.lower()
        if _architecture_name == "mri_trans_gan":
            self._architectures["generator"] = mri_trans_gan.Generator
            self._architectures["discriminator"] = mri_trans_gan.Discriminator
        elif _architecture_name == "mri_trans_gan_v2":
            self._architectures["generator"] = mri_trans_gan_v2.Generator
            self._architectures["discriminator"] = mri_trans_gan_v2.Discriminator
        elif _architecture_name == "mri_trans_gan_v3":
            self._architectures["generator"] = mri_trans_gan_v3.Generator
            self._architectures["discriminator"] = mri_trans_gan_v3.Discriminator
        elif _architecture_name == "mri_trans_gan_v4":
            self._architectures["generator"] = mri_trans_gan_v4.Generator
            self._architectures["discriminator"] = mri_trans_gan_v4.Discriminator
        else:
            raise ValueError("Unsupported architecture {}!".format(self.__architecture_name))
    @property
    def architectures(self):
        return self._architectures

if __name__ == "__main__":
    # print(hasattr(networks,"mri_trans_gan"))
    # print(dir([networks]))
    # print(getattr(networks,"__name__"))
    # print(getattr(networks,"__name__"))
    class A():
        def __init__(self) -> None:
            pass
    a = A()
    a.architecture_name= "mri_trans_gan_v3"
    b = NetworkSelector(a)