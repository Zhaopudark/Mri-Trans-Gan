
import os
import logging
from logging import handlers
import argparse
from typeguard import typechecked
import tensorflow as tf
from utils.types import get_tuple_from_str,flatten
from utils.managers import set_global_loggers
class ArgsPlus():
    """
    manager argparse.ArgumentParser
    get args stamp from their default values
    save args into a txt to a target file
    """
    @typechecked
    def __init__(self,args_file:str,parser:argparse.ArgumentParser,args_passby:set=None) -> None:
        self._parser = parser
        self._args = vars(self._parser.parse_args([os.path.normpath(args_file)]))
        self._default_args = vars(self._parser.parse_args([]))
        self._args_passby = set() if args_passby is None else args_passby
    @property
    def args(self):
        return self._args
    @property
    def stamp(self):
        if not hasattr(self,'_stamp'):
            self._stamp = self._get_args_stamp(self._args,self._default_args,self._args_passby)
        return self._stamp
    @typechecked
    def _stamp_shorter(self,key:str,value:str|int|float|bool|list|tuple):
        match value:
            case list(value)|tuple(value):
                value = flatten(value)
                return ("-").join([self._stamp_shorter(key,item) for item in value])
            case str(value):
                return ("-").join(value.split('_'))
            case float(value)|int(value) if not isinstance(value,bool):
                return str(value)
            case bool(value): #约定bool型为store——true 所以以key作为stamp
                return str(key)
            case _:
                raise ValueError(f"Unsupported type:{key} {value} {type(value)}")
    @typechecked
    def _get_args_stamp(self,vars:dict,default_vars:dict,passby_keys:set[str]=None):
        if passby_keys is None:
            passby_keys = set()
        stamp = ''
        for key in vars:
            if key not in passby_keys:
                if (key not in default_vars) or (vars[key]!=default_vars[key]):
                    stamp += f"_{self._stamp_shorter(key,vars[key])}"
                else:
                    pass

        # for (k1,v1),(k2,v2) in zip(vars.items(),default_vars.items()):
        #     assert k1==k2
        #     if (k1 not in passby_keys) and (v1 != v2):
        #         stamp += f"_{self._stamp_shorter(k1,v1)}"
        return stamp.strip("_")
    @typechecked
    def save_args(self,target_path:str):
        target_path = os.path.normpath(target_path)
        dir_path = os.path.dirname(os.path.abspath(target_path))
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        with open(target_path,'w') as f:
            for k,v in self._args.items():
                if isinstance(v,bool):
                    if v:
                        f.writelines(f'--{k}\n')
                elif v is not None:
                    f.writelines(f'--{k}\n')
                    f.writelines(f'{v}\n')
@typechecked
def get_args(arg_path:str)->dict:
    parser = argparse.ArgumentParser(prog='',allow_abbrev=False,fromfile_prefix_chars='@',formatter_class=argparse.RawTextHelpFormatter)
    passby_set = {  'action','workspace','data_dividing_rates',
                    'cut_ranges','patch_sizes','batch_size',
                    'patch_nums','domain','steps','logs_interval',
                    'checkpoint_interval','checkpoint_max_keep'}
    initial_group = parser.add_argument_group('initial')
    initial_group.add_argument('--action',choices=['initial','train','test','debug',"train-and-test"],type=str.lower)
    initial_group.add_argument("--workspace",type=str)
    initial_group.add_argument("--init",type=str.lower)
    initial_group.add_argument("--indicator",type=str.lower)
    initial_group.add_argument("--determinism",action='store_true',help="确定性算法")
    initial_group.add_argument("--global_random_seed",type=int,help="全局随机种子 其余部分")
    initial_group.add_argument("--logs_interval",type=int,default=1000)
    initial_group.add_argument("--checkpoint_interval",type=int,default=100)
    initial_group.add_argument("--checkpoint_max_keep",type=int,default=3)
    #----------------------------------------------------------------------------------------#
    df_group = parser.add_argument_group('data_flow')
    df_group.add_argument("--dataset",choices=['brats','ixi'],type=str.lower)
    df_group.add_argument("--data_random_seed",type=int)
    df_group.add_argument("--data_dividing_rates",type=get_tuple_from_str)
    df_group.add_argument("--data_dividing_seed",type=int)
    df_group.add_argument("--norm",choices=['min_max_norm','z_score_norm','z_score_and_min_max_norm','individual_min_max_norm'],type=str.lower,default='individual_min_max_norm')
    df_group.add_argument("--raw_data_format",type=str,choices=['DHW','HWD'])
    df_group.add_argument("--cut_ranges",type=get_tuple_from_str,metavar=tuple[tuple[int,int],...],help="should in the same `length` with raw_data_format")
    df_group.add_argument("--patch_sizes",type=get_tuple_from_str,metavar=tuple[int,...],help="should in the same `length` with raw_data_format")
    df_group.add_argument("--batch_size",type=int,default=1)
    df_group.add_argument("--patch_nums",type=get_tuple_from_str,metavar=tuple[int,...],help="should in the same `length` with raw_data_format")
    df_group.add_argument("--domain",type=get_tuple_from_str,metavar=tuple[float,float])
    #----------------------------------------------------------------------------------------#
    ms_group = parser.add_argument_group('model_structure')
    ms_group.add_argument("--model_name",type=str)
    ms_group.add_argument("--capacity_vector",type=int,default=32)
    ms_group.add_argument("--res_blocks_num",type=int,default=9)
    ms_group.add_argument("--self_attention_G",action='store_true')
    ms_group.add_argument("--self_attention_D",action='store_true')
    ms_group.add_argument("--dimensions_type",type=str,choices=['2D','3D','3D_special'],default='3D_special')
    ms_group.add_argument("--architecture_name",type=str)
    ms_group.add_argument("--up_sampling_method",type=str,choices=['transpose','up_conv','sub_pixel_up'],default='up_conv')
    # parser.add_argument("--down_sampling_method",type=str,default='default')
    #----------------------------------------------------------------------------------------#
    gl_group = parser.add_argument_group('GAN_LOSS')
    gl_group.add_argument("--gan_loss_name",type=str,default='Vanilla')
    gl_group.add_argument("--wgp_penalty_l",type=float)
    gl_group.add_argument("--wgp_initial_seed",type=int,help="WGAN-GP中e序列的初始化随机种子")
    gl_group.add_argument("--wgp_random_always",action='store_true',help="如果wgp_random_always=False\n则每次都已相同的向量e指导WGAN-GP的loss计算\n反之每次e都将重新随机")
    #----------------------------------------------------------------------------------------#
    sn_group = parser.add_argument_group('spectral_normalization')
    sn_group.add_argument("-sn","--spectral_normalization",action='store_true')
    sn_group.add_argument("--sn_random_seed",type=int)
    sn_group.add_argument("--sn_iter_k",type=int,default=1)
    sn_group.add_argument("--sn_clip_flag",action='store_true')
    sn_group.add_argument("--sn_clip_range",type=float,default=100.0)
    #---------------------------------------------------------------------------------------#
    rec_loss_group = parser.add_argument_group('reconstruction_loss')
    rec_loss_group.add_argument("--CC",action='store_true')
    rec_loss_group.add_argument("--CC_l",type=float,default=0.0)
    rec_loss_group.add_argument("--MAE",action='store_true')
    rec_loss_group.add_argument("--MSE",action='store_true')
    rec_loss_group.add_argument("--MGD",action='store_true')
    rec_loss_group.add_argument("--Per_Reuse_D",action='store_true')
    metric_group = parser.add_argument_group('metric')
    metric_group.add_argument("--metrics",type=get_tuple_from_str,metavar=tuple[str,...],default=('psnr3d','ssim3d'))
    #---------------------------------------------------------------------------------------#
    _group = rec_loss_group.add_mutually_exclusive_group()
    _group.add_argument('--Per', action='store_true')
    _group.add_argument('--Per_2D',action='store_true')
    #---------------------------------------------------------------------------------------#
    rec_loss_group.add_argument("--Sty",action='store_true')
    rec_loss_group.add_argument("--transfer_learning_model",type=str,choices=['vgg16','vgg19'],default='vgg16')
    #----------------------------------------------------------------------------------------#
    training_group = parser.add_argument_group('training')
    training_group.add_argument("--epochs",type=int,default=200)
    training_group.add_argument("--steps",type=int,default=80000*2)
    training_group.add_argument("--G_learning_rate",type=float,default=1e-4)
    training_group.add_argument("--D_learning_rate",type=float,default=4e-4)
    #-------------------------------------------------------------------------------------------------------------#
    optm_group = parser.add_argument_group('optimizer')
    optm_group.add_argument("--optimizer_name",type=str.lower,default='adam')
    # optm_group.add_argument("--optimizer_random_seed",type=int,help='$1')
    optm_group.add_argument("--adam_beta_1",type=float,default=0.0)
    optm_group.add_argument("--adam_beta_2",type=float,default=0.9)
    optm_group.add_argument("--rmsprop_rho",type=float,default=0.9)
    optm_group.add_argument("--rmsprop_momentum",type=float,default=0.0)
    optm_group.add_argument("--sgd_momentum",type=float,default=0.0)
    #-------------------------------------------------------------------------------------------------------------#
    nc_group = parser.add_argument_group('numerical_calculation')
    nc_group.add_argument("--xla",action='store_true')
    nc_group.add_argument("--mixed_precision",action='store_true')
    #-------------------------------------------------------------------------------------------------------------#
    subparsers = parser.add_subparsers(title='learning_rate_schedule',dest='learning_rate_schedule',help='additional help')
    subparser_1 = subparsers.add_parser('cosine_decay',allow_abbrev=False)
    subparser_1.add_argument("--cosine_decay_steps",type=int,default=92601)
    subparser_1.add_argument("--cosine_alpha",type=float,default=0.0)
    subparser_2 = subparsers.add_parser('cosine_decay_restarts',allow_abbrev=False)
    subparser_2.add_argument("--cosine_decay_restarts_first_decay_steps",type=int,default=92601//2)
    subparser_2.add_argument("--cosine_decay_restarts_t_mul",type=float,default=2.0)
    subparser_2.add_argument("--cosine_decay_restarts_m_mul",type=float,default=0.5)
    subparser_2.add_argument("--cosine_decay_restarts_alpha",type=float,default=0.0)
    subparser_3 = subparsers.add_parser('exponential_decay',allow_abbrev=False)
    subparser_3.add_argument("--exponential_decay_steps",type=int,default=100)
    subparser_3.add_argument("--exponential_decay_rate",type=float,default=0.98)
    subparser_3.add_argument("--exponential_decay_staircase",action='store_true')
    subparser_4 = subparsers.add_parser('inverse_time_decay',allow_abbrev=False)
    subparser_4.add_argument("--inverse_time_decay_steps",type=int,default=100)
    subparser_4.add_argument("--inverse_time_decay_rate",type=float,default=0.98)
    subparser_4.add_argument("--inverse_time_decay_staircase",action='store_true')
    subparser_5 = subparsers.add_parser('piecewise_constant_decay',allow_abbrev=False)
    subparser_5.add_argument("--piecewise_constant_decay_boundaries",type=get_tuple_from_str,metavar=tuple[int,...],default=[20000,40000,60000])
    subparser_5.add_argument("--piecewise_constant_decay_values",type=get_tuple_from_str,metavar=tuple[float,...],default=[1.0,0.67,0.33,0.0])
    subparser_6 = subparsers.add_parser('poly_nomial_decay',allow_abbrev=False)
    subparser_6.add_argument("--poly_nomial_decay_steps",type=int,default=100)
    subparser_6.add_argument("--poly_nomial_decay_end_learning_rate",type=float,default=0.0001)
    subparser_6.add_argument("--poly_nomial_decay_power",type=float,default=1.0)
    subparser_6.add_argument("--poly_nomial_decay_cycle",action='store_true')
    subparser_7 = subparsers.add_parser('custom_decay',allow_abbrev=False)
    subparser_7.add_argument("--custom_decay_steps",type=int,default=100)
    subparser_7.add_argument("--custom_decay_rate",type=float,default=0.98)
    subparser_7.add_argument("--custom_decay_start_step",type=int,default=40000)
    subparser_7.add_argument("--custom_decay_staircase",action='store_true')

    args_plus = ArgsPlus(args_file=f'@{arg_path}',parser=parser,args_passby=passby_set)
    weight_path = os.path.normpath(f"{args_plus.args['workspace']}\\init\\{args_plus.stamp}")
    logs_path = os.path.normpath(f"{args_plus.args['workspace']}\\{args_plus.stamp}")
    args_plus.save_args(f"{logs_path}\\args.txt")
    return args_plus.args|{'weight_path':weight_path,'logs_path':logs_path,'stamp':args_plus.stamp}
@typechecked
def experiment_runer(arg_path:str,logging_config_path:str):
    args = get_args(arg_path)
    set_global_loggers(config_path=logging_config_path,target_prefix=args['logs_path'])
    logging.getLogger(__name__).info(f"Current Experiment Stamp: {args['stamp']}")
    
    # tf.config.experimental.enable_op_determinism()
    # tf.config.experimental.enable_tensor_float_32_execution(False)
    # tf.config.experimental.set_synchronous_execution(True)
    # tf.config.run_functions_eagerly(True)
    # tf.debugging.enable_check_numerics()
    # tf.autograph.set_verbosity(10, alsologtostdout=False)
    
    from models.model_selector import ModelSelector 
    tf.keras.utils.set_random_seed(args['global_random_seed'])
    if args['determinism']:
        tf.config.experimental.enable_op_determinism()
        logging.getLogger(__name__).info(f"Determinism has been enabled!")
    model_selector = ModelSelector(args=args)
    model = model_selector.model(args=args)
    model.build()
    if args['action'] == 'train':
        model.train()
    elif args['action'] == 'test':
        model.test()
    elif args['action'] == 'debug':
        model.debug()
    elif args['action'] == "train-and-test":
        model.train()
        model.test()
    elif args['action'] == 'initial':
        model.weight_initial()
    else:
        raise ValueError("Unsupported model action!")

#----------------------------------------------------------------------------------------#
if __name__ == "__main__":
    experiment_runer()
#------------------------------------------------------------------------------------#

    



# argsDict = args.__dict__
# with open('setting.txt', 'w') as f:
#     f.writelines('------------------ start ------------------'+'\n')
#     for eachArg, value in argsDict.items():
#         f.writelines("--"+eachArg + '\n' + str(value) + '\n')
#     f.writelines('------------------- end -------------------')