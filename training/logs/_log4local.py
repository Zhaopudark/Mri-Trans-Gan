import sys
import os
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../../'))
from utils.csv_process import CsvWriter
from utils.image.image_process import ImageSaver
from utils.types_check import type_check
__all__ = [
    "LocalLogsMaker",
]
class LocalLogsMaker():
    def __init__(self,paths_dict,counters_dict) -> None:
        """
        logs的interval由于涉及到模型推理 应当放到模型的文件中进行实现 不应该在此处实现
        原因如下:
            如果要在此处实现 就太过复杂了 
            要么每一步都推理仅按照interval记录推理结果 浪费大量算力 
            要么传递推理函数 但推理函数太过多变 抽象困难 
        不同于checkpoint保存的便捷 固定性 logs的interval 统一到模型的构建中实现才是最佳选择
        """
        self._paths_dict = paths_dict
        self._counters_dict = counters_dict
    def write(self,training,time_stamp,log_types,log_contents):
        if training:
            for log_type,log_content in zip(log_types,log_contents):
                _path = self._get_log_types_dir(path=self._paths_dict["train"],log_type=log_type,time_stamp=None)
                _writer = self._get_contents_writer(log_type)
                _ = _writer(_path,log_content)
        else:
            for log_type,log_content in zip(log_types,log_contents):
                _path = self._get_log_types_dir(path=self._paths_dict["test"],log_type=log_type,time_stamp=time_stamp)
                _writer = self._get_contents_writer(log_type)
                _ = _writer(_path,log_content)
    def _get_log_types_dir(self,path,log_type,time_stamp=None): #分出该功能 统一管理时间戳 避免混乱
        if log_type is None:
            tmp_path = ""
        elif log_type=="loss": 
            tmp_path = "/loss"
        elif log_type=="metric":
            tmp_path = "/metric"
        elif log_type=="image":
            tmp_path = "/image"
        else:
            raise ValueError("Unsupported log_types:{}".format(log_type))
        if time_stamp is not None:
            return path+"/"+time_stamp+tmp_path
        else:
             return path+tmp_path
    def _get_contents_writer(self,log_type):
        if log_type=="loss": # 在 loss 文件夹下记录csv step--losses 为内容加入时间戳一列
            return self._write_loss
        elif log_type=="metric":# 在 metric 文件夹下记录csv step--metric 为内容加入时间戳一列
            return self._write_metric
        elif log_type=="image":# 在 metric 文件夹下记录images 可以存在子文件夹
            return self._save_image
        else:
            raise ValueError("Unsupported log_types:{}".format(log_type))
    def _write_loss(self,path,loss_content):
        _path = path+"/loss"+"_step_"+str(int(self._counters_dict["step"].numpy()))+".csv"
        csv_writer = CsvWriter(file_name=_path,manner="dict")
        csv_writer.writing(rows=loss_content)
        del csv_writer
    def _write_metric(self,path,metric_content):
        _path = path+"/metric"+"_step_"+str(int(self._counters_dict["step"].numpy()))+".csv"
        csv_writer = CsvWriter(file_name=_path,manner="dict")
        csv_writer.writing(rows=metric_content)
        del csv_writer
    def _save_image(self,path,img_content):
        _path = path+"/image"+"_step_"+str(int(self._counters_dict["step"].numpy()))
        img_saver = ImageSaver(_path)
        img_saver.saving(img_dict=img_content)
        del img_saver
    