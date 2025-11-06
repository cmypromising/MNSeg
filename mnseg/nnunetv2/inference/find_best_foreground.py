# 写一个脚本，在一个文件夹中，找到包含前景最多元素的前五个.nii.gz文件，并输出它的路径
import os
import SimpleITK as sitk
import numpy as np

def find_best_foreground(root):
    """在指定目录下找到前景像素最多的前5个.nii.gz文件
    
    Args:
        root: 文件目录路径
        
    Returns:
        list: 包含前5个文件路径和对应前景像素数的列表
    """
    path_list = [f for f in os.listdir(root) if f.endswith('.nii.gz')]
    path_list.sort()
    
    file_foregrounds = []
    for path in path_list:
        full_path = os.path.join(root, path)
        mri = sitk.ReadImage(full_path)
        mri_array = sitk.GetArrayFromImage(mri)
        # 计算非零像素数量作为前景
        foreground_count = np.count_nonzero(mri_array)
        file_foregrounds.append((full_path, foreground_count))
    
    # 按前景像素数量降序排序并取前5个
    file_foregrounds.sort(key=lambda x: x[1], reverse=True)
    return file_foregrounds[:5]

if __name__ == '__main__':
    root = '/home/promising/NAS_DATA/data/bigneuron/origin_data/data/000003_sitk'
    top_files = find_best_foreground(root)
    
    print("前景像素最多的5个文件:")
    for i, (file_path, count) in enumerate(top_files, 1):
        print(f"{i}. 文件路径: {file_path}")
        print(f"   前景像素数: {count}")