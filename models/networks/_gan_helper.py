"""
实现和GAN理论到实践相关的若干共性规则
"""
class GeneratorHelper():
    def __init__(self,name,args):
        self.name = name
        self.domain = sorted(map(float,args.domain))
    def output_activation_name(self):
        """依据指定值域,确定生成器的生成值域,即最后一层的激活函数"""
        if self.domain == [0.0,1.0]:
            return 'sigmoid'
        elif self.domain == [-1.0,1.0]:
            return 'tanh'
        else:
            return 'domain_shift_sigmoid'
            # raise ValueError("Unsupported domain")
 
class DiscriminatorHelper():
    def __init__(self,name,args):
        self.name = name
        self.gan_loss_name = args.gan_loss_name
    def output_sigmoid_flag(self):
        if self.gan_loss_name.lower() in ['wgan',"wgan-gp"]:
            print("Discriminator's last layer will not be limited (clipped) by sigmoid")
            return False
        elif self.gan_loss_name.lower() in ['vanilla','lsgan','rsgan']:
            print("Discriminator's last layer will be limited (clipped) by sigmoid")
            return True
        else: 
            raise ValueError(f"Unsupported loss name: {self.gan_loss_name}")

