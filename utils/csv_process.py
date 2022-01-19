import csv
import os
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base))
from types_check import type_check
__all__ = [
    "CsvWriter",
]
def _csv_lists_writer(file_name,head_list,row_lists):
    for index in range(len(file_name)-1,-1,-1):
        if file_name[index]=="\\" or file_name[index]=="/":
            break
    dir_path =  file_name[0:index]
    if os.path.exists(dir_path):
        pass 
    else:
        os.makedirs(dir_path)
    if os.path.exists(file_name):
        with open(file_name,mode='a',newline='') as file:
            file_csv = csv.writer(file)
            file_csv.writerows(row_lists)
    else:
        with open(file_name,mode='w',newline='') as file:
            file_csv = csv.writer(file)
            file_csv.writerow(head_list)
            file_csv.writerows(row_lists)

def _csv_dicts_writer(file_name,head_list,row_dicts):
    for index in range(len(file_name)-1,-1,-1):
        if file_name[index]=="\\" or file_name[index]=="/":
            break
    dir_path =  file_name[0:index]
    if os.path.exists(dir_path):
        pass 
    else:
        os.makedirs(dir_path)
    if os.path.exists(file_name):
        with open(file_name,mode='a',newline='')as file:
            file_csv = csv.DictWriter(file,head_list)
            file_csv.writerows(row_dicts)
    else:
        with open(file_name,mode='w',newline='')as file:
            file_csv = csv.DictWriter(file,head_list)
            file_csv.writeheader()
            file_csv.writerows(row_dicts)
class CsvWriter():
    def __init__(self,file_name,manner="dict"):
        if not isinstance(file_name,str):
            raise ValueError("file_name must be a string.")
        if ".csv" in file_name:
            pass
            
        elif "." in file_name:
            index = file_name.find(".")
            file_name = file_name[0:index]+".csv"
        else:
            file_name = file_name[:]+".csv"
        self.file_name = file_name
        self.manner = manner
    def writing(self,headers=None,rows=None):
        if self.manner == "list":
            if type_check(headers,[list]):
                pass 
            else:
                raise ValueError("Header must be a list with depth=1.")
            if type_check(rows,[list,list]):
                pass
            elif type_check(rows,[list]):
                rows = [rows] 
            else:
                raise ValueError("Rows must be a list with depth = 2 or 1.")
            _csv_lists_writer(file_name=self.file_name,head_list=headers,row_lists=rows)
        elif self.manner == "dict":
            if type_check(rows,[list,dict]):
                pass 
            elif type_check(rows,[dict]):
                rows = [rows]
            else:
                raise ValueError("Rows must be a list with depth=0 or 1.")
            if headers is None:
                headers = list(rows[0].keys()) 
            else:
                if type_check(headers,[list]):
                    pass 
                else:
                    raise ValueError("Header must be a list with depth=1.")
            _csv_dicts_writer(file_name=self.file_name,head_list=headers,row_dicts=rows)
        else:
            raise ValueError("Unsupported CSV writer manner: {}".format(self.manner))
if __name__ == "__main__":
    headers = ['class','name','sex','height','year']
    rows = [[1,'xiaoming','male',168,23],
            [1,'xiaohong','female',162,22],
            [2,'xiaozhang','female',163,21],
            [2,'xiaoli','male',158,21]]
    rows2 = [{'class':1,'name':'xiaoming','sex':'male','height':168,'year':23},
             {'class':1,'name':'xiaohong','sex':'female','height':162,'year':22},
             {'class':2,'name':'xiaozhang','sex':'female','height':163,'year':21},
             {'class':2,'name':'xiaoli','sex':'male','height':158,'year':21}]
    _csv_lists_writer(file_name="./test.csv", head_list=headers, row_lists=rows)
    _csv_dicts_writer(file_name="./test2.csv", head_list=headers, row_dicts=rows2)
