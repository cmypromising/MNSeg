"""
这个脚本负责3D图像的导入和导出
"""
import os, cv2, sys
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed 
IMAGE_SUFFIXES = ['.jpg', '.jpeg', '.tiff', '.bmp', '.png', '.tif']
Image2DName_Length = 6
def load_image_3d(image_root):
    """
    从文件夹 image_root 中加载其中包含的 3D 图像数据
    :param image_root: 一个保存二维图像序列的路径，这些二维图像是一个三维图像的分片表示
    :return:
    """
    image_name_list = os.listdir(image_root)
    image_name_list.sort()
    image_3d = []
    for image_name in image_name_list:
        _, suffix = os.path.splitext(image_name)
        if suffix not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(os.path.join(image_root, image_name), 0)
        image_3d.append(image)
    return np.array(image_3d)

def save_image_3d(image_3d, image_save_root, dim = 0, suffix = '.tiff'):
    """
    这个函数将三维矩阵保存为对应张数的二维图像
    :param image_3d: np.multiarray, 三维矩阵
    :param image_save_root: string, 保存路径
    :param dim: int, 三维矩阵的切片维度
    :param suffix: string, 保存的二维图像的后缀名
    :return:
    """
    assert dim < len(image_3d.shape)
    assert suffix in IMAGE_SUFFIXES
    if not os.path.isdir(image_save_root):
        os.makedirs(image_save_root)
    #shape_ = [image_3d.shape[i] for i in range(len(image_3d.shape)) if image_3d.shape[i] != 1]
    #image_3d = image_3d.reshape(shape_)
    #shape_ = [image_3d.shape[i] for i in range(len(image_3d.shape)) if i != dim]
    #shape_.insert(0, image_3d.shape[dim])
    #print(image_3d.shape)
    if len(os.listdir(image_save_root)):
        os.system('rm ' + os.path.join(image_save_root,'*'))
    depth = image_3d.shape[dim]
    for index in range(depth):
        image_full_name = os.path.join(image_save_root, str(index).zfill(Image2DName_Length)) + suffix
        flag = cv2.imwrite(image_full_name, image_3d[index])
        if flag == False:
            print('{} to save {}-th image in {}'.format(flag, index, image_save_root))
            break
        time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
        sys.stdout.write('\r{}-------- saving {} / {} 2D image ...'.format(time, index, depth))
        sys.stdout.flush()
    time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
    sys.stdout.write('\n{}-------- finish saving !\n'.format(time))


def save_image_3d(image_3d, image_save_root, dim=0, suffix='.tiff', max_workers=None):  
    """  
    高效地将三维矩阵保存为对应张数的二维图像  
    
    :param image_3d: np.multiarray, 三维矩阵  
    :param image_save_root: string, 保存路径  
    :param dim: int, 三维矩阵的切片维度  
    :param suffix: string, 保存的二维图像的后缀名  
    :param max_workers: int, 并发保存的最大线程数(默认为CPU核心数)  
    :return: 保存成功的图像数量  
    """  
    # 参数验证  
    assert dim < len(image_3d.shape), "切片维度超出矩阵维度"  
    assert suffix in IMAGE_SUFFIXES, f"不支持的文件后缀: {suffix}"  
    
    # 确保保存路径存在  
    os.makedirs(image_save_root, exist_ok=True)  
    
    # 清空目标目录  
    for file in os.listdir(image_save_root):  
        os.remove(os.path.join(image_save_root, file))  
    
    # 获取切片深度  
    depth = image_3d.shape[dim]  
    
    # 并发保存图像  
    def save_single_image(index):  
        """保存单张图像的内部函数"""  
        try:  
            image_full_name = os.path.join(  
                image_save_root,   
                f"{str(index).zfill(Image2DName_Length)}{suffix}"  
            )  
            flag = cv2.imwrite(image_full_name, image_3d[index])  
            return flag, index  
        except Exception as e:  
            print(f"保存第 {index} 张图像时发生错误: {e}")  
            return False, index  
    
    # 使用线程池并发保存  
    successful_saves = 0  
    start_time = datetime.now()  
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  
        # 提交所有保存任务  
        future_to_index = {  
            executor.submit(save_single_image, index): index   
            for index in range(depth)  
        }  
        
        # 实时进度追踪  
        for future in as_completed(future_to_index):  
            flag, index = future.result()  
            if flag:  
                successful_saves += 1  
                
                # 实时进度输出  
                time_str = f'[{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}]'  
                sys.stdout.write(  
                    f'\r{time_str}-------- 保存进度: {successful_saves} / {depth} 2D图像 ...'  
                )  
                sys.stdout.flush()  
    
    # 保存完成后的最终输出  
    end_time = datetime.now()  
    duration = (end_time - start_time).total_seconds()  
    
    print(f"\n[{end_time.strftime('%Y-%m-%d_%H-%M-%S')}]-------- 保存完成!")  
    print(f"总耗时: {duration:.2f}秒, 成功保存 {successful_saves}/{depth} 张图像")  
    
    return successful_saves  

if __name__ == '__main__':
    create_edge_label(image_root="E:\\NeCuDa\\DataBase_5_new")
    # test()
    # test_get_edge()