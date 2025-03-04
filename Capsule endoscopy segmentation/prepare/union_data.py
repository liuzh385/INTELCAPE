import os
import shutil

source_dirs = "/mnt/minio/node77/liuzheng/RJ/small_jpg"
target_dir = "/mnt/minio/node77/liuzheng/RJ/Data/small_all_jpg"
source_dirs = os.listdir(source_dirs)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    

# for source_dir in source_dirs:
#     for 
tmp = source_dirs[0]
tmp = os.path.join(source_dirs[0], tmp)
for root, _, files in os.walk(tmp):
    for file in files:
        src_file_path = os.path.join(root, file)
        dst_file_path = os.path.join(target_dir, source_dirs[0], file)
        print(dst_file_path)