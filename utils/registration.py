from subprocess import check_output
import os
import shutil
# for patient in os.listdir(root_dir):
#     fixed_path = os.path.join(patient_path, fixed)
#     moving_path = ''
#     mat_out_path = ''
#     nii_out_path = ''
#     # 复制配准参考图像
#     shutil.copy(fixed_path, dst_dir)
#     for modal in modalities:
#         # 不是配准参考图像
#         if modal != fixed:
#             moving_path = os.path.join(patient_path, modal)
#             mat_out_path = os.path.join(mat_dir, modal+'.mat')
#             nii_out_path = os.path.join(dst_dir, 'reg_'+modal)

#             # 构造命令
#             reg_cmd = r'greedy -d 3 -a -m NMI -i {} {} -dof 6 -o {} -ia-image-centers -n 30x30x0'.format(fixed_path, moving_path, mat_out_path)
#             apply_cmd = r'greedy -d 3  -rf {} -rm {} {} -ri LINEAR -r {}'.format(fixed_path, moving_path, nii_out_path, mat_out_path)

#             # registration
#             print(reg_cmd)
#             try:
#                 reg_info = check_output(reg_cmd, shell=True).decode()
#                 print(reg_info)
#             except:

#             # apply
#             print(apply_cmd)
#             try:
#                 apply_info = check_output(apply_cmd, shell=True).decode()
#                 print(apply_info)
#             except:



# fixed_path = "H:\\Datasets\\IXI\\IXI-T1\\IXI662-Guys-1120-T1.nii.gz"
# moving_path = "H:\\Datasets\\IXI\\IXI-T2\\IXI662-Guys-1120-T2.nii.gz"
# mat_out_path = "E:\\Datasets\\IXI\\IXI662-Guys-1120.mat"
# nii_out_path = "E:\\Datasets\\IXI\\IXI662-Guys-1120.nii.gz"
# reg_cmd = r'greedy -d 3 -a -m NMI -i {} {} -dof 6 -o {} -ia-image-centers -n 30x30x0'.format(fixed_path, moving_path, mat_out_path)
# print(reg_cmd)
# reg_info = check_output(reg_cmd, shell=True).decode()
# print(reg_info)
# apply_cmd = r'greedy -d 3  -rf {} -rm {} {} -ri LINEAR -r {}'.format(fixed_path, moving_path, nii_out_path, mat_out_path)
# apply_info = check_output(apply_cmd, shell=True).decode()
# print(apply_info)

path_T2 = "G:\\Datasets\\IXI\\IXI-T2\\"
buf_T2 = []
for (dirName, subdirList, fileList) in os.walk(path_T2):
    for filename in fileList:
        if "t2.nii.gz" in filename.lower(): 
            buf_T2.append(os.path.join(dirName,filename))

print(buf_T2)

for t2_path in buf_T2:
    fixed_path   = "G:\\Datasets\\IXI\\IXI-T1\\"+t2_path[23:-9]+"T1.nii.gz"
    moving_path  = t2_path[:]
    mat_out_path =  "G:\\Datasets\\IXI\\IXI-T22T1\\"+t2_path[23:-10]+".mat"
    nii_out_path =  "G:\\Datasets\\IXI\\IXI-T22T1\\"+t2_path[23::]

    # print(fixed_path   )
    # print(moving_path  )
    # print(mat_out_path )
    # print(nii_out_path )
    if os.path.exists(fixed_path):
        reg_cmd = r'greedy -d 3 -a -m NMI -i {} {} -dof 6 -o {} -ia-image-centers -n 30x30x0'.format(fixed_path, moving_path, mat_out_path)
        print(reg_cmd)
        reg_info = check_output(reg_cmd, shell=True).decode()
        print(reg_info)
        apply_cmd = r'greedy -d 3  -rf {} -rm {} {} -ri LINEAR -r {}'.format(fixed_path, moving_path, nii_out_path, mat_out_path)
        apply_info = check_output(apply_cmd, shell=True).decode()
        print(apply_info)
