"""
这个脚本对单个神经元进行分割
"""

import os, sys
import torch
from thop import profile
import copy
import numpy as np
import SimpleITK as sitk
from skimage.morphology import skeletonize_3d

from mnseg.nnunetv2.constant import RESULT_SAVE_PATH, DataNeuron_Root, MODEL_ROOT, Block_Size
from mnseg.nnunetv2.tools.image_3D_io import save_image_3d, load_image_3d
from skimage.measure import label as Label
from mnseg.nnunetv2.inference.huge_image_partition_assemble import NeuronImage_Huge

from mnseg.nnunetv2.inference.divide_neuron_image import Image3D_PATH
from mnseg.nnunetv2.constant import Image2DName_Length, Image3DName_Length
from datetime import datetime
from mnseg.nnunetv2.tools.printer import print_my
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

debug = True
if debug:
    nnUNet_results = "/home/promising/NAS_DATA/nnunet/nnUNet_results"

class Segmentation_SingleNeuron():
    """
    对单个神经元图像进行 切分、逐块分割、拼接、保存 等操作
    """
    def __init__(self, neuron_name, neuron_image_path, predictor, batchsize = 8, ):
        """
        :param neuron_image_path: 神经元图像保存路径，它的终端子目录名称通常是其指代的神经元名称；其中包含一个子目录，如 'image'
        :param batchsize: 批次大小
        """
        self.neuron_name = neuron_name
        self.num_class = 2
        self.predictor = predictor
        # param = self._count_parameters_in_M(self.predictor.network)
        # print(f"param is :{param}M")
        # self._count_flopsAndParam(self.predictor.network)
        self._data(neuron_image_path = neuron_image_path, batchsize = batchsize)

    def _count_parameters_in_M(self, model):
        """
        计算模型的参数量，并以M为单位输出
        :param model: PyTorch模型
        :return: 模型的总参数量（单位：百万）
        """
        total_params = sum(p.numel() for p in model.parameters())
        total_params_in_M = total_params / 1e6  # 转换为百万
        return total_params_in_M

    def _count_flopsAndParam(self, model):
        # 确保模型在 GPU 上
        model = model.cuda()
        # 创建输入张量并将其放到 GPU 上
        input = torch.randn(1, 1, 32, 128, 128).cuda()
        # 计算 FLOPs 和参数量
        flops, params = profile(model, inputs=(input,))
        # 转换为百万（M）
        flops /= 1e6
        params /= 1e6
        # 打印结果
        print(f"FLOPs: {flops:.2f}M, Params: {params:.2f}M")

    def _data(self, neuron_image_path, batchsize):
        # 将神经元图像切分并生成 pytorch 能处理的数据
        # self.singleneuron_dataset = SingleNeuron_Dataset(neuron_image_path = neuron_image_path)
        self.neuron_image = Image3D_PATH(neuron_image_path)
        self.neuron_image.divide_regular(block_size=Block_Size)
        self.neuron_blocks = self.neuron_image.block_list
        self.block_num = len(self.neuron_blocks)
        self.sitk_neuron_image_path = neuron_image_path + '_sitk'

        # 将分割后的神经元block存储成nii.gz格式并保存路径。
        if not os.path.exists(self.sitk_neuron_image_path):
            print("该神经元无sitk数据，开始生成...")
            os.makedirs(self.sitk_neuron_image_path, exist_ok=True)
            for index, block in enumerate(self.neuron_blocks):
                print(f"processing----{index}/{len(self.neuron_blocks)}")
                image_sitk = sitk.GetImageFromArray(block)
                save_image_path = os.path.join(self.sitk_neuron_image_path, 'neuron_' + str(index).zfill(6) + '_0000' + '.nii.gz')
                sitk.WriteImage(image=image_sitk, fileName=save_image_path)
        del self.neuron_blocks
        self.image_3d = self.neuron_image.image_3d
        self.block_locations = self.neuron_image.location_list     #获取规则切割神经元图像后的每个图像块起始坐标列表
        self.block_sizes = self.neuron_image.size_list             #获取规则切割时候每个图像块的实际尺寸
        self.image_shape = self.neuron_image.shape()                          #获取神经元尺寸

    def segment_block(self):
        """
        逐块分割
        :return:
        """
        self.block_label_pre = None   # 保存每个图像块的分割结果
        block_list, props_list = [], []
        for i in range(self.block_num):
            nii_block_path = join(self.sitk_neuron_image_path, 'neuron_' + str(i).zfill(6) + '_0000' + '.nii.gz')
            block, props = SimpleITKIO().read_images([nii_block_path])
            block_list.append(block)
            props_list.append(props)

        self.block_label_pre = self.predictor.predict_from_list_of_npy_arrays(block_list,
                                                        None,
                                                        props_list,
                                                        None, 2, save_probabilities=False,
                                                        num_processes_segmentation_export=2)

        time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
        sys.stdout.write('\n{}-------- finish segmenting for the neuron image "{}"\n'.format(time, self.neuron_name))

    def merge(self):
        """
        将图像块的分割结果按照位置坐标进行合并
        :return:
        """
        print_my('-------- merging data ....')
        self.label_pre = np.zeros(shape = self.image_shape)
        assert len(self.block_label_pre) == len(self.block_locations) == len(self.block_sizes)
        for index in range(self.block_num):
            label_pre = self.block_label_pre[index]
            time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
            sys.stdout.write('\r{}-------- merging {} - {} block '.format(time, index, len(self.block_label_pre)))
            sys.stdout.flush()
            z_start, y_start, x_start = self.block_locations[index]
            depth, height, width = self.block_sizes[index]
            z_stop = z_start + depth
            y_stop = y_start + height
            x_stop = x_start + width
            if (depth, height, width) != Block_Size:
                self.label_pre[z_start:z_stop, y_start:y_stop, x_start:x_stop] += label_pre[:depth, :height, :width]
            else:
                self.label_pre[z_start:z_stop, y_start:y_stop, x_start:x_stop] += label_pre
        time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
        sys.stdout.write('\n{}-------- finish merging - 0...'.format(time))
        self.label_pre[self.label_pre != 0] = 255
        self.label_pre = self.label_pre.astype(np.uint8)
        time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
        sys.stdout.write('\n{}-------- finish merging - 1...\n'.format(time))

    def segment(self, final_process = False):
        """
        按照神经元图像的分割结果提取原图像中的神经元部分
        :param final_process: True: 进行求最大连通域操作，能够将某些孤立点去除，但也会将一些孤立纤维片段去除；对于尺寸比较大的图像，这个操作速度很慢，不建议使用
        :return:
        """
        print_my('-------- in segmenting - 0')
        self.image_pre = torch.tensor(self.image_3d).cuda()
        print_my('-------- in segmenting - 1')
        for i in range(self.image_pre.shape[0]):
            self.image_pre[i][self.label_pre[i] != 255] = 0
        self.image_pre = self.image_pre.cpu().numpy()
        print_my('-------- in segmenting - 2')
        if final_process:
            self._get_theMax_connectedRegion()
            self.image_final = copy.copy(self.image_3d)
            self.image_final[self.label_pre_final != 1] = 0

    def run(self, final_process = False):
        """
        :param final_process: True: 进行求最大连通域操作，能够将某些孤立点去除，但也会将一些孤立纤维片段去除
        :return:
        """
        self.segment_block()      #对每个图像块进行图像分割，预测其中属于神经元的像素点
        self.merge()        #将预测结果进行合并处理，获得整个神经元图像的预测结果
        self.segment(final_process = final_process)

    def _get_theMax_connectedRegion(self, connectivity = 3):
        """
        返回预测标签中的最大连通域
        :return:
        """
        # raise RuntimeError('这个子函数还是不要用了！')
        print('---------- geting the max connected region ...')
        connected_label, num = Label(self.label_pre, connectivity = connectivity, background = 0, return_num = True)
        max_label = 0
        max_num = 0
        for index in range(1, num+1):
            if np.sum(connected_label == index) > max_num:
                max_num = np.sum(connected_label == index)
                max_label = index
        self.label_pre_final = (connected_label == max_label)

    def save(self, save_path, label = False, final_process = False):
        """
        :param final_process: True: 进行求最大连通域操作，能够将某些孤立点去除，但也会将一些孤立纤维片段去除
        :return:
        """
        print_my('-------- saving image data ... ')
        save_image_3d(self.image_pre, image_save_root = os.path.join(save_path, 'image_pre_init'))
        if final_process:
            save_image_3d(self.image_final, image_save_root = os.path.join(save_path, 'image_pre_final'))
        if label:
            print_my('\n-------- saving label data ... ')
            save_image_3d(self.label_pre, image_save_root = os.path.join(save_path, 'label_pre_init'))

            label_enhanced_image_pre = ((self.image_pre.astype(np.int16) + self.label_pre.astype(np.int16))/2).astype(np.uint8)
            skeleton_label = skeletonize_3d(self.label_pre)
            skeleton_enhanced_image_pre = np.where(skeleton_label!=0, self.label_pre, self.image_pre)
            save_image_3d(image_3d=label_enhanced_image_pre, image_save_root=os.path.join(save_path, 'label_enhanced_image_pre'))
            save_image_3d(image_3d=skeleton_enhanced_image_pre, image_save_root=os.path.join(save_path, 'skeleton_enhanced_image_pre'))

            if final_process:
                self.label_pre_final = np.array(self.label_pre_final * 255, dtype = np.uint8)
                save_image_3d(self.label_pre_final, image_save_root = os.path.join(save_path, 'label_pre_final'))

                label_enhanced_image_pre = (
                            (self.image_final.astype(np.int16) + self.label_pre_final.astype(np.int16)) / 2).astype(np.uint8)
                skeleton_label = skeletonize_3d(self.label_pre_final)
                skeleton_enhanced_image_pre = np.where(skeleton_label != 0, self.label_pre_final, self.image_final)
                save_image_3d(image_3d=label_enhanced_image_pre,
                              image_save_root=os.path.join(save_path, 'label_enhanced_image_pre_final'))
                save_image_3d(image_3d=skeleton_enhanced_image_pre,
                              image_save_root=os.path.join(save_path, 'skeleton_enhanced_image_pre_final'))



class Segmentation_HugeNeuron():
    """
    对单个超大尺寸神经元图像进行 划分、逐子图像分割、整合、保存 等操作
    混合、融合、配合、结合、组合、整合
    """
    def __init__(self, neuron_name, neuron_image_root, net_pretrained, batchsize=8):
        """
        :param neuron_name: 超大神经元名称
        :param neuron_image_root: 神经元图像保存路径，这个路径的终端目录通常是当前神经元名称 neuron_name
        :param net_pretrained: 已经加载了预训练模型的 pytorch 网络
        :param batchsize: 批次大小
        """
        self.num_class = 2
        self.neuron_name = neuron_name
        self.neuron_image_root = neuron_image_root
        self.huge_neuron = NeuronImage_Huge(neuron_name = self.neuron_name,
                                            neuron_image_root = self.neuron_image_root)
        self.net_pretrained = net_pretrained
        self.batchsize = batchsize

    def run(self, save_path, label = False):
        """
        对超大尺寸的神经元图像进行：
        1.划分
        2.逐个子图分割
        3.合并分割后的子图
        :return:
        """
        self.partition()
        self.segment()
        self.assemble(save_root = save_path, label = label)

    def partition(self):
        self.huge_neuron.partition()
        self.partitioned_image_root = self.huge_neuron.partition_save_root

    def segment(self):
        save_path = self.partitioned_image_root
        for index, neuron_subimage_name in enumerate(self.huge_neuron.subimage_name_list):
            print_my('')
            subimage_path = os.path.join(self.partitioned_image_root, neuron_subimage_name)
            seg_neuron = Segmentation_SingleNeuron(neuron_name = neuron_subimage_name + ' / '
                                                                 + str(self.huge_neuron.image3d_number_partitioned).zfill(Image3DName_Length),
                                                   neuron_image_path = subimage_path,
                                                   batchsize = self.batchsize)
            seg_neuron.run()
            seg_neuron.save(save_path = os.path.join(save_path, neuron_subimage_name), label = True)
            print_my('')

    def assemble(self, save_root, label = True):
        self.huge_neuron.assemble(source_root = self.partitioned_image_root,
                                  save_root = save_root, leaf_path = 'image_pre_init')
        if label:
            self.huge_neuron.assemble(source_root = self.partitioned_image_root,
                                      save_root = save_root, leaf_path = 'label_pre_init')

def get_predictor(model_path):
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_path,
        use_folds=(1,),
        checkpoint_name='checkpoint_best.pth',
    )
    return predictor

def denoise_neuron_test(net_name, model_path='./', parameter='origin', neuron_name_list=None, batchsize = 8, final_process=False):
    """
        :param net_name:
        :param wavename:
        :param model_epoch:
        :param batchsize:
        :param gpus:
        :return:
        """
    # 预训练模型
    predictor = get_predictor(model_path=model_path)
    # 保存路径
    save_root = os.path.join(RESULT_SAVE_PATH, net_name, parameter, '29')
    run_times = dict()
    for neuron_name in neuron_name_list:
        t0 = datetime.now()
        neuron_image_path = os.path.join(DataNeuron_Root, neuron_name)
        print_my('processing {}'.format(neuron_image_path))
        seg_neuron = Segmentation_SingleNeuron(neuron_name=neuron_name, neuron_image_path=neuron_image_path,
                                               predictor=predictor, batchsize=batchsize)
        seg_neuron.run(final_process=final_process)
        save_path = os.path.join(save_root, neuron_name)
        t1 = datetime.now()
        print('segmentating {} took {} secs'.format(neuron_name, t1 - t0))
        run_times[neuron_name] = t1 - t0
        seg_neuron.save(save_path=save_path, label=True, final_process=final_process)
        del seg_neuron
    for neuron_name in run_times:
        print('{} ==> {}'.format(neuron_name, run_times[neuron_name]))


if __name__ == '__main__':
    neuron_name = '000019'
    neuron_image_path = os.path.join(DataNeuron_Root, neuron_name)
    seg = Segmentation_SingleNeuron(neuron_name=neuron_name,
                                    neuron_image_path=neuron_image_path,
                                    batchsize=8)
    seg.segment_block()