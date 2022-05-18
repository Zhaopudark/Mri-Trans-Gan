"""
logs 区别于checkpoints 
记录若干训练中的要素
包括metrics losses值 images
"""
import sys
import os
import datetime
import tensorflow as tf 
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base))
from _log4local import LocalLogsMaker as _LocalLogsMaker
from _log4tensorboard import SummaryMaker as _SummaryMaker
__all__ = [
    'LogsMaker',
]
class LogsMaker():
    """
    summary writer 
    为模型训练和测试服务
    训练时  以训练step 记录训练某个阶段的图片,损失函数等监控值
    测试时  以测试样本为step 输出记录值
        对于 图像记录 如有需要 调用图像处理函数 保存 并记录图像 同步到summary中
        对于 metrics 如有需要 将metrics返回的结果dict 保存到summry中 
        对于 训练过程中的loss 如有需要 保存到summary中 
    dict_list
    """
    def __init__(self,counters_dict,path):
        self._counters_dict = counters_dict
        self._path = path
        self._path_dict = self._get_paths_dict(self._path)
        self._time_stamp_register = {}
        self.local_logs_maker = _LocalLogsMaker(paths_dict=self._path_dict,counters_dict=self._counters_dict)
        self.summary_maker = _SummaryMaker(paths_dict=self._path_dict,counters_dict=self._counters_dict)
    def _get_paths_dict(self,path):
        if path[-1]=="\\" or path[-1]=="/":
            _path = path[0:-1]
        else:
            _path = path[0::]
        tmp_dict = {}
        tmp_dict['train'] = _path+"/train_logs"
        tmp_dict['test'] = _path+"/test_logs"
        return tmp_dict
    def _get_time_stamp(self):
        if 'time' not in self._time_stamp_register.keys():
            self._time_stamp_register['time'] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return self._time_stamp_register['time']
    def wirte(self,training,log_types,log_contents):
        time_stamp = self._get_time_stamp()
        self.local_logs_maker.write(training,time_stamp,log_types,log_contents)   
        self.summary_maker.write(training,time_stamp,log_types,log_contents)