import os

root = "/GPUFS/sysu_gbli_1/zhaoxinkai/data/Crohn_avi"
for name in os.listdir("/GPUFS/sysu_gbli_1/zhaoxinkai/data/Crohn_avi"):
    # print(os.path.join(root, name[:-23]))
    os.rename(os.path.join(root, name), os.path.join(root, name[:-23]))