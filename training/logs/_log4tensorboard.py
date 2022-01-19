import tensorflow as tf 
import sys
import os
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../../'))
from utils.types_check import type_check
__all__ = [
    "SummaryMaker",
]
class SummaryMaker():
    def __init__(self,paths_dict,counters_dict) -> None:
        """
        logs的interval由于涉及到模型推理 应当放到模型的文件中进行实现 不应该在此处实现
        原因如下:
            如果要在此处实现 就太过复杂了 
            要么每一步都推理 但仅按照interval记录推理结果 浪费大量算力 
            要么传递推理函数 但推理函数太过多变 抽象困难 
        不同于checkpoint保存的便捷 固定性 logs的interval 统一到模型的构建中实现才是最佳选择
        """
        self._paths_dict = paths_dict
        self._counters_dict = counters_dict
        self._summary_writer_register = {} #避免反复del 
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
            tmp_path = "/summary"
        elif log_type=="metric":
            tmp_path = "/summary"
        elif log_type=="image":
            tmp_path = "/summary"
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
        if type_check(loss_content,[dict]):
            pass
        else:
            raise ValueError("loss_content must be a dict")
        _path = path+"/loss"
        if _path in self._summary_writer_register.keys():
            pass 
        else:
            self._summary_writer_register[_path] = tf.summary.create_file_writer(_path)
        with self._summary_writer_register[_path].as_default():
            for key,value in loss_content.items():
                tf.summary.scalar(name=key,data=value,step=self._counters_dict["step"])  
    def _dict_dict_transpose(self,dict_dict):
        out_dict = {}
        for key1,value1 in dict_dict.items():
            for key2,value2 in value1.items():
                if key2 not in out_dict.keys():
                    out_dict[key2] = {key1:value2}
                else:
                    out_dict[key2][key1] = value2
        return out_dict
    def _list_dict2dict_dict(self,list_dict):
        out_dict = {}
        for _dict in list_dict:
            buf = {}
            for key,value in _dict.items():
                if key == list(_dict.keys())[0]:#永远将第一个键值对视为index指示 记录为key
                    out_key = value
                else:
                    buf[key] = value
            out_dict[out_key] = buf 
        return out_dict                
    def _write_metric(self,path,metric_content):
        if type_check(metric_content,[dict,dict]):
            _metric_content = metric_content
        elif type_check(metric_content,[list,dict]):
            _metric_content = self._list_dict2dict_dict(metric_content)
        else:
            raise ValueError("loss_content must be a dict")
        if len(_metric_content.keys())<len(list(_metric_content.values())[0].keys()):
            pass
        else:
            _metric_content = self._dict_dict_transpose(_metric_content)
        for rows_name,row_results in _metric_content.items():
            _path = path+"/metric/"+rows_name
            if _path in self._summary_writer_register.keys():
                pass 
            else:
                self._summary_writer_register[_path] = tf.summary.create_file_writer(_path)
            with self._summary_writer_register[_path].as_default():
                for column_name,value in row_results.items():
                    tf.summary.scalar(name=column_name,data=value,step=self._counters_dict["step"])                
    def _save_image(self,path,img_content):
        _path = path+"/image"
        if _path in self._summary_writer_register.keys():
            pass
        else:
            self._summary_writer_register[_path] = tf.summary.create_file_writer(_path)
        with self._summary_writer_register[_path].as_default():
            for key,value in img_content.items():
                tf.summary.image(name=key,data=value,step=self._counters_dict["step"])

if __name__ == "__main__":
    dic = {"b1":{"a1":1,"a2":2,"a3":3},"b2":{"a1":4,"a2":5,"a3":6}}
    print(SummaryMaker._dict_dict_transpose(SummaryMaker,dic))
    print(int("123132"))

 