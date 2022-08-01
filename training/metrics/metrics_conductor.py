
from . import psnr
from . import ssim
__all__ = [
    'MetricsConductor',
]
def metrics_select(metric_name):
    if metric_name == 'psnr3d':
        return psnr.PeakSignal2NoiseRatio3D() 
    elif metric_name == 'ssim3d':
        return ssim.Ssim3D()
    else:
        raise ValueError(f"Unsupported metric {metric_name}") 

class MetricsConductor():
    def __init__(self,metrics):
        self._metrics = []
        for item in metrics:
            self._metrics.append(metrics_select(metric_name=item))
        """
        [[psnr1,psnr2],
         [ssim1,ssim2],
         [mae1,mae2]]
        """
    def calculate(self,dict_in_dict,func_apply_list,mean=False): # 得到在所有给出样本中的均值 一般仅在训练的验证中使用
        out_buf = []
        for key,value in dict_in_dict.items():
            buf = {}
            buf['key'] = key
            for metric in self._metrics:
                for func_apply in func_apply_list:
                    buf[f"{metric.name} {func_apply.name}"] = func_apply(metric,value)
            out_buf.append(buf)
        if mean:
            buf = {}
            buf['key'] = 'mean'
            for index,result in enumerate(out_buf): # index是从0开始的
                for name,value in result.items():
                    if name == 'key':
                        continue
                    if name not in buf.keys():
                        buf[name] = value
                    else:
                        buf[name] = buf[name]*(index/(index+1))+value/(index+1) #为了防止溢出
            out_buf.append(buf)
        else:
            pass
        return out_buf
class MetricsConductorV2():
    """
    Calculate metrics 
    针对单组数据
    针对批量数据：
        
    """
    def __init__(self,metrics_name_list:list[str]) -> None:
        pass

