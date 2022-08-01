
from . import mri_trans_gan_v5

class NetworkSelector():
    def __init__(self,args):
        self._architectures = {}
        _architecture_name = args['architecture_name'].lower()
        if _architecture_name == 'mri_trans_gan_v5':
            self._architectures['generator'] = mri_trans_gan_v5.Generator
            self._architectures['discriminator'] = mri_trans_gan_v5.Discriminator
        else:
            raise ValueError(f"Unsupported architecture {self.__architecture_name}!")
    @property
    def architectures(self):
        return self._architectures

if __name__ == '__main__':
    # print(hasattr(networks,'mri_trans_gan'))
    # print(dir([networks]))
    # print(getattr(networks,'__name__'))
    # print(getattr(networks,'__name__'))
    class A():
        def __init__(self) -> None:
            pass
    a = A()
    a.architecture_name= 'mri_trans_gan_v3'
    b = NetworkSelector(a)