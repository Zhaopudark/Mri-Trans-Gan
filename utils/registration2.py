from subprocess import check_output
import os
import shutil
def acc():
    print("????")
path_T1 = "G:\\Datasets\\IXI\\IXI-T1\\"
buf_T1 = []
for (dirName, subdirList, fileList) in os.walk(path_T1):
    for filename in fileList:
        if "t1.nii.gz" in filename.lower(): 
            buf_T1.append(os.path.join(dirName,filename))
print(buf_T1)

for t1_path in buf_T1:
    fixed_path   = "G:\\Datasets\\IXI\\IXI-T2\\"+t1_path[23:-9]+"T2.nii.gz"
    moving_path  = t1_path[:]
    mat_out_path =  "G:\\Datasets\\IXI\\IXI-T12T2\\"+t1_path[23:-10]+".mat"
    nii_out_path =  "G:\\Datasets\\IXI\\IXI-T12T2\\"+t1_path[23::]
    # print(fixed_path)
    # print(moving_path)
    # print(mat_out_path)
    # print(nii_out_path)
    if os.path.exists(fixed_path):
        reg_cmd = r'greedy -d 3 -a -m NMI -i {} {} -dof 6 -o {} -ia-image-centers -n 30x30x0'.format(fixed_path, moving_path, mat_out_path)
        print(reg_cmd)
        reg_info = check_output(reg_cmd, shell=True).decode()
        print(reg_info)
        apply_cmd = r'greedy -d 3  -rf {} -rm {} {} -ri LINEAR -r {}'.format(fixed_path, moving_path, nii_out_path, mat_out_path)
        apply_info = check_output(apply_cmd, shell=True).decode()
        print(apply_info)
