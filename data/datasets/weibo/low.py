import os
# 定义函数进行批量重命名
def batch_rename(files_path):
    # 遍历目标目录下所有文件
    for file_name in os.listdir(files_path):
        lowercase_filename = file_name.lower()
        # 重命名文件
        os.rename(os.path.join(files_path, file_name), os.path.join(files_path, lowercase_filename))

# 调用函数进行批量重命名
batch_rename('rumor_images')
