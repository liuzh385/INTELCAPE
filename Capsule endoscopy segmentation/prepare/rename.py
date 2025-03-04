import os


def rename_files_in_folder(root_folder):
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        # print(subfolder_path)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                old_file_path = os.path.join(subfolder_path, file)
                new_file_name = f"{subfolder}_{file}"
                new_file_path = os.path.join(subfolder_path, new_file_name)
                # print(new_file_name)
                if os.path.exists(new_file_path):
                    # print(f"文件已存在，跳过: {new_file_path}")
                    continue
                
                os.rename(old_file_path, new_file_path)
                # print(f"重命名: {old_file_path} -> {new_file_path}")
            print("over")
        
        


if __name__ == "__main__":
    root_folder = "/mnt/minio/node77/liuzheng/RJ/Data/small_jpg"
    rename_files_in_folder(root_folder)
    print("重命名完成")