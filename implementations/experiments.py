import os 
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base,'../'))
class ArgsRecoder():
    """
    记录与默认值不符合命令行参数
    形成类似于时间戳的标记mark
    用以帮助标记实验名称(无序,所以不直接生成实验名称,仅作实验名称与地址的辅助)
    """
    def __init__(self) -> None:
        self.args_recoder = {}
        self.concerned_list = []
    def __call__(self,name,parser_added,concerned=True):
        _arg_recoder = {}
        _arg_recoder["default"] = parser_added.default #未设置则默认为None
        self.args_recoder[name] = _arg_recoder
        if concerned:
            self.concerned_list.append(name)
        else:
            pass
    def _mark_shorter(self,key,value):
        if type(value)==list:
            buf = ""
            for item in value:
                if type(item)== list:
                    buf += self._mark_shorter(key,item)+"x"
                else: 
                    buf += str(item)+"x"
            buf = buf[0:-1]
        elif type(value)==str:  
            buf = value.split("_")
            buf = ("-").join(buf)
        elif type(value)==int:
            buf = str(value)
        elif type(value)==float:
            buf = str(value)
        elif type(value)==bool:
            buf = str(key).split("_")
            buf = ("-").join(buf)
        else:
            raise ValueError("Unsupported mark type:{} {} {}".format(key,value,type(value)))
        return buf
    def default_check(self,dict_args): #返回基于非默认值形成的标记mark 忽略非concerned的参数值 比如logs路径等
        check_mark = ""
        for key,value in dict_args.items():
            if (value != self.args_recoder[key]["default"])and(key in self.concerned_list):
                check_mark += "_"+self._mark_shorter(key,value)
                print(key,value,"default:",self.args_recoder[key]["default"])
        return check_mark

import argparse
parser = argparse.ArgumentParser(prog='MRI_Trans_GAN',allow_abbrev=False,fromfile_prefix_chars='@')
parser_recoder = ArgsRecoder()
initial_group = parser.add_argument_group('initial')
parser_recoder("action",
    initial_group.add_argument("action",choices=["initial","train","test","debug","train-and-test"],type=str.lower),
    concerned=False)
parser_recoder("workspace",
    initial_group.add_argument("--workspace",type=str),
    concerned=False)
parser_recoder("init",
    initial_group.add_argument("--init",type=str.lower))
parser_recoder("indicator",
    initial_group.add_argument("--indicator",type=str.lower))
parser_recoder("global_random_seed",
    initial_group.add_argument("--global_random_seed",type=int,help="全局随机种子 其余部分"))
parser_recoder("logs_interval",
    initial_group.add_argument("--logs_interval",type=int,default=1000))
parser_recoder("checkpoint_interval",
    initial_group.add_argument("--checkpoint_interval",type=int,default=100))
parser_recoder("checkpoint_max_keep",
    initial_group.add_argument("--checkpoint_max_keep",type=int,default=3))
#----------------------------------------------------------------------------------------#
df_group = parser.add_argument_group('data_flow')
parser_recoder("dataset",
    df_group.add_argument("--dataset",choices=["brats","ixi"],type=str.lower))
parser_recoder("data_random_seed",
    df_group.add_argument("--data_random_seed",type=int,help="数据集随机种子"))
parser_recoder("norm",
    df_group.add_argument("--norm",choices=["min_max","z_score","z_score_and_min_max"],type=str.lower,default="z_score"))
parser_recoder("data_format",
    df_group.add_argument("--data_format",type=str,choices=["DHW","HWD"],default="DHW"))
parser_recoder("cut_ranges",
    df_group.add_argument("--cut_ranges",type=int,nargs='+',action="append",default=[[0,239],[0,239],[69,84]]))
parser_recoder("patch_size",
    df_group.add_argument("--patch_size",type=int,nargs='+',default=[128,128,3]))
parser_recoder("batch_size",
    df_group.add_argument("--batch_size",type=int,default=1))
parser_recoder("patch_nums",
    df_group.add_argument("--patch_nums",type=int, nargs='+',default=[3,3,6]))
parser_recoder("domain",
    df_group.add_argument("--domain", type=float, nargs='+',default=[-5.0,37.0]))
#----------------------------------------------------------------------------------------#
ms_group = parser.add_argument_group('model_structure')
parser_recoder("model_name",
    ms_group.add_argument("--model_name",type=str))
parser_recoder("capacity_vector",
    ms_group.add_argument("--capacity_vector",type=int,default=32))
parser_recoder("res_blocks_num",
    ms_group.add_argument("--res_blocks_num",type=int,default=9))
parser_recoder("self_attention_G",
    ms_group.add_argument("--self_attention_G",action="store_true"))
parser_recoder("self_attention_D",
    ms_group.add_argument("--self_attention_D",action="store_true"))
parser_recoder("dimensions_type",
    ms_group.add_argument("--dimensions_type",type=str,choices=["2D","3D","3D_special"],default="3D_special"))
parser_recoder("architecture_name",
    ms_group.add_argument("--architecture_name",type=str))
parser_recoder("up_sampling_method",
    ms_group.add_argument("--up_sampling_method",type=str,choices=["transpose","up_conv","sub_pixel_up"],default="up_conv"))
# parser.add_argument("--down_sampling_method",type=str,default="default")
#---------------------------------------------------------------------------------------#
rec_loss_group = parser.add_argument_group('reconstruction_loss')
parser_recoder("CC",
    rec_loss_group.add_argument("--CC",action="store_true"))
parser_recoder("CC_l",
    rec_loss_group.add_argument("--CC_l",type=float,default=0.0))
parser_recoder("MAE",
    rec_loss_group.add_argument("--MAE",action="store_true"))
parser_recoder("MSE",
    rec_loss_group.add_argument("--MSE",action="store_true"))
parser_recoder("MGD",
    rec_loss_group.add_argument("--MGD",action="store_true"))
parser_recoder("Per_Reuse_D",
    rec_loss_group.add_argument("--Per_Reuse_D",action="store_true"))
##
_group = rec_loss_group.add_mutually_exclusive_group()
parser_recoder("Per",
    _group.add_argument('--Per', action='store_true'))
parser_recoder("Per_2D",
    _group.add_argument('--Per_2D',action='store_true'))
##
parser_recoder("Sty",
    rec_loss_group.add_argument("--Sty",action="store_true"))
parser_recoder("transfer_learning_model",
    rec_loss_group.add_argument("--transfer_learning_model",type=str,choices=["vgg16","vgg19"],default="vgg16"))
#----------------------------------------------------------------------------------------#
gl_group = parser.add_argument_group('GAN_LOSS')
parser_recoder("gan_loss_name",
    gl_group.add_argument("--gan_loss_name",type=str,default="Vanilla"))
parser_recoder("wgp_penalty_l",
    gl_group.add_argument("--wgp_penalty_l",type=float))
parser_recoder("wgp_initial_seed",
    gl_group.add_argument("--wgp_initial_seed",type=int,help="WGAN-GP中e序列的初始化随机种子"))
parser_recoder("wgp_random_always",
    gl_group.add_argument("--wgp_random_always",action="store_true",help="如果wgp_random_always=False 则每次都已相同的向量e指导WGAN-GP的loss计算 反之每次e都将重新随机"))
#----------------------------------------------------------------------------------------#
sn_group = parser.add_argument_group('spectral_normalization')
parser_recoder("spectral_normalization",
    sn_group.add_argument("-sn","--spectral_normalization",action="store_true")) 
parser_recoder("sn_random_seed",
    sn_group.add_argument("--sn_random_seed",type=int))
parser_recoder("sn_iter_k",
    sn_group.add_argument("--sn_iter_k",type=int,default=1))
parser_recoder("sn_clip_flag",
    sn_group.add_argument("--sn_clip_flag",action="store_true"))
parser_recoder("sn_clip_range",
    sn_group.add_argument("--sn_clip_range",type=float,default=100.0))
#----------------------------------------------------------------------------------------#
training_group = parser.add_argument_group('training')
parser_recoder("epochs",
    training_group.add_argument("--epochs",type=int,default=400))
parser_recoder("steps",
    training_group.add_argument("--steps",type=int,default=80000))
parser_recoder("G_learning_rate",
    training_group.add_argument("--G_learning_rate",type=float,default=1e-4))
parser_recoder("D_learning_rate",
    training_group.add_argument("--D_learning_rate",type=float,default=4e-4))
#-------------------------------------------------------------------------------------------------------------#
optm_group = parser.add_argument_group('optimizer')
parser_recoder("optimizer_name",
    optm_group.add_argument("--optimizer_name",type=str.lower,default="adam"))
parser_recoder("optimizer_random_seed",
    optm_group.add_argument("--optimizer_random_seed",type=int,help="优化器随机种子"))
parser_recoder("adam_beta_1",
    optm_group.add_argument("--adam_beta_1",type=float,default=0.0))
parser_recoder("adam_beta_2",
    optm_group.add_argument("--adam_beta_2",type=float,default=0.9))
parser_recoder("rmsprop_rho",
    optm_group.add_argument("--rmsprop_rho",type=float,default=0.9))
parser_recoder("rmsprop_momentum",
    optm_group.add_argument("--rmsprop_momentum",type=float,default=0.0))
parser_recoder("sgd_momentum",
    optm_group.add_argument("--sgd_momentum",type=float,default=0.0))
#-------------------------------------------------------------------------------------------------------------#
lrs_group = parser.add_argument_group('learning_rate_schedule')
parser_recoder("learning_rate_schedule",
    lrs_group.add_argument("--learning_rate_schedule",type=str,choices=["cosine_decay","cosine_decay_restarts","exponential_decay","inverse_time_decay","piecewise_constant_decay","poly_nomial_decay","custom_decay"],default="custom_decay"))
# 目前互斥参数组不支持 add_argument_group() 的 title 和 description 参数。
# lrs_group = parser.add_mutually_exclusive_group()
# lrs_group = lrs_group.add_argument_group('cosine_decay')
parser_recoder("cosine_decay_1",
    lrs_group.add_argument("--cosine_decay_1",type=int,default=80000))
parser_recoder("cosine_decay_2",
    lrs_group.add_argument("--cosine_decay_2",type=float,default=0.0))
# lrs_group = lrs_group.add_argument_group('cosine_decay_restarts')
parser_recoder("cosine_decay_restarts_1",
    lrs_group.add_argument("--cosine_decay_restarts_1",type=int,default=100))
parser_recoder("cosine_decay_restarts_2",
    lrs_group.add_argument("--cosine_decay_restarts_2",type=float,default=2.0))
parser_recoder("cosine_decay_restarts_3",
    lrs_group.add_argument("--cosine_decay_restarts_3",type=float,default=1.0))
parser_recoder("cosine_decay_restarts_4",
    lrs_group.add_argument("--cosine_decay_restarts_4",type=float,default=0.0))
# lrs_group = lrs_group.add_argument_group('exponential_decay')
parser_recoder("exponential_decay_1",
    lrs_group.add_argument("--exponential_decay_1",type=int,default=100))
parser_recoder("exponential_decay_2",
    lrs_group.add_argument("--exponential_decay_2",type=float,default=0.98))
parser_recoder("exponential_decay_3",
    lrs_group.add_argument("--exponential_decay_3",action="store_true"))

parser_recoder("custom_decay_1",
    lrs_group.add_argument("--custom_decay_1",type=int,default=100))
parser_recoder("custom_decay_2",
    lrs_group.add_argument("--custom_decay_2",type=float,default=0.98))
parser_recoder("custom_decay_3",
    lrs_group.add_argument("--custom_decay_3",type=int,default=40000))
parser_recoder("custom_decay_4",
    lrs_group.add_argument("--custom_decay_4",action="store_true"))
# lrs_group = lrs_group.add_argument_group('inverse_time_decay')
parser_recoder("inverse_time_decay_1",
    lrs_group.add_argument("--inverse_time_decay_1",type=int,default=100))
parser_recoder("inverse_time_decay_2",
    lrs_group.add_argument("--inverse_time_decay_2",type=float,default=0.98))
parser_recoder("inverse_time_decay_3",
    lrs_group.add_argument("--inverse_time_decay_3",action="store_true"))
# lrs_group = lrs_group.add_argument_group('piecewise_constant_decay')
parser_recoder("piecewise_constant_decay_1",
    lrs_group.add_argument("--piecewise_constant_decay_1",type=int,nargs='+',default=[20000,40000,60000]))
parser_recoder("piecewise_constant_decay_2",
    lrs_group.add_argument("--piecewise_constant_decay_2",type=float,nargs='+',default=[1.0,0.67,0.33,0.0]))
# lrs_group = lrs_group.add_argument_group('poly_nomial_decay')
parser_recoder("poly_nomial_decay_1",
    lrs_group.add_argument("--poly_nomial_decay_1",type=int,default=100))
parser_recoder("poly_nomial_decay_2",
    lrs_group.add_argument("--poly_nomial_decay_2",type=float,default=0.0001))
parser_recoder("poly_nomial_decay_3",
    lrs_group.add_argument("--poly_nomial_decay_3",type=float,default=1.0))
parser_recoder("poly_nomial_decay_4",
    lrs_group.add_argument("--poly_nomial_decay_4",action="store_true"))
# parser_recoder("learning_rate_decay_method",
#     parser.add_argument("--learning_rate_decay_method",type=str,choices=["linear","exponential","none"],default="exponential"))
# parser_recoder("learning_rate_decay_interval",
#     parser.add_argument("--learning_rate_decay_interval",type=int,default=100))
# parser_recoder("lr_exp_base",
#     parser.add_argument("--lr_exp_base",type=float,default=0.98))
# parser_recoder("learning_rate_decay_begin_rate",
#     parser.add_argument("--learning_rate_decay_begin_rate",type=float,choices=[0.0,0.25,0.5,0.75,1.0],default=0.5))
#-------------------------------------------------------------------------------------------------------------#
nc_group = parser.add_argument_group('numerical_calculation')
parser_recoder("xla",
    nc_group.add_argument("--xla",action="store_true"))
parser_recoder("mixed_precision",
    nc_group.add_argument("--mixed_precision",action="store_true"))

metric_group = parser.add_argument_group('metric')
parser_recoder("metrics_list",
    metric_group.add_argument("--metrics_list",nargs="+",type=str,default=["psnr3d","ssim3d"]))
#----------------------------------------------------------------------------------------#
args = parser.parse_args()
dic_args = vars(args)
args.mark = parser_recoder.default_check(dic_args).strip("_")
args.weight_path = args.workspace+"/init/"+args.init
args.logs_path = args.workspace+"/%(mark)s"%dic_args

# print(args.mark)
import tensorflow as tf
from models.model_selector import ModelSelector 
from datasets.data_pipeline import PipeLine
physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.enable_op_determinism()
# tf.config.experimental.enable_tensor_float_32_execution(False)
# tf.config.experimental.set_synchronous_execution(True)
pipe_line = PipeLine(args)

model_selector = ModelSelector(args=args)
tf.keras.utils.set_random_seed(args.global_random_seed)
model = model_selector.model(args=args,pipe_line=pipe_line)
model.build()
if args.action == "train":
    model.train()
elif args.action == "test":
    model.test()
elif args.action == "debug":
    model.debug()
elif args.action == "train-and-test":
    model.train()
    model.test()
elif args.action == "initial":
    model.weight_initial()
else:
    raise ValueError("Unsupported model action!")



# argsDict = args.__dict__
# with open('setting.txt', 'w') as f:
    # f.writelines('------------------ start ------------------'+'\n')
    # for eachArg, value in argsDict.items():
    #     f.writelines("--"+eachArg + '\n' + str(value) + '\n')
    # f.writelines('------------------- end -------------------')