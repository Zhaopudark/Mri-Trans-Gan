import csv 
import os 
import numpy as np
path = "F:\\Training\\Codes_tmp\\mri_trans_gan_3x128x128_brats_WGAN-GP_10.0_sn_True_new_per_13"
csv_files_list = []
for (dirName,_, fileList) in os.walk(path):
    for filename in fileList:
        if ".csv" in filename.lower():   
            csv_files_list.append(os.path.join(dirName,filename))

print(csv_files_list)
for csv_file in  csv_files_list:
    with open(csv_file) as f:
        f_csv = csv.reader(f)
        hearders = next(f_csv)
        # print(hearders)
        rows = []
        for row in f_csv:
            # print(row)
            rows.append(row)
        a = np.array(rows,dtype=np.float32)
        a = np.delete(a,0,axis=-1)# 删除最后一轴的第0维
        print(np.mean(a,axis=0))
        print(np.std(a,axis=0))
s = 0
for i in range(100,10000+1,2):
    s+=i 
print(s)