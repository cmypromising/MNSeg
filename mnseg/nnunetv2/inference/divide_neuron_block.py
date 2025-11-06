import torch
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from skimage.morphology import skeletonize_3d
import SimpleITK as sitk
import os
from mnseg.nnunetv2.tools.image_3D_io import save_image_3d, load_image_3d


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

def neuron_name_list(block_root, block_files):
    """
    生成完整的神经元数据路径名列表
    :return:
    """
    assert os.path.isfile(block_files)
    neuron_name_list = open(block_files).readlines()
    neuron_name_list = [line.strip() for line in neuron_name_list]
    neuron_name_list = [line for line in neuron_name_list if not line.startswith('#')]
    return [line for line in neuron_name_list if os.path.isdir(os.path.join(block_root, line))]

if __name__ == '__main__':
    model_name = "test1_xxl_enhance++_1(1_5)ce_1dc_1cl(4)"
    # 读取image_root:
    block_root = "/home/promising/NAS_DATA/datasets/bigneuron/DataBase_5_new"
    block_files = "/home/promising/NAS_DATA/datasets/bigneuron/DataBase_5_new/test.txt"
    block_list = neuron_name_list(block_root, block_files)
    predictor = get_predictor(model_path="/home/promising/NAS_DATA/nnunet/nnUNet_results/Dataset114_BigNeuron_enhance++/nnUNetTrainerMNSeg__nnUNetPlans__3d_fullres")
    # print(block_list)
    save_root = "/home/promising/NAS_DATA/data/bigneuron/test_data"
    for block_name in block_list:

        block_path = os.path.join(block_root, block_name)
        image_path = os.path.join(block_path, "image")
        label_path = os.path.join(block_path, "label")
        print("image_path", image_path)
        # 加载图像数据
        image = load_image_3d(image_root=image_path)
        label = (load_image_3d(image_root=label_path) * 255).astype(np.uint8)

        # 使用 SimpleITK 获取图像属性
        image_sitk = sitk.GetImageFromArray(image)

        # 只保留 'spacing' 键
        image_props = {
            "spacing": image_sitk.GetSpacing()
        }
        image_tensor = np.expand_dims(image, axis=0)
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32).cpu()
        # 调用预测函数，传递图像数据和属性
        block_label_pre = predictor.predict_single_npy_array(image_tensor, image_props)

        print("label_pre_shape:", block_label_pre.shape)
        # 输出预测结果
        # print("block_label_pre", block_label_pre)
        # print("label_pre unique", np.unique(block_label_pre))

        block_label_pre[block_label_pre != 0] = 255
        block_label_pre = block_label_pre.astype(np.uint8)
        ske_label = skeletonize_3d(label)
        ske_label_pre = skeletonize_3d(block_label_pre)
        print("ske_label:", ske_label.shape)
        print("ske_label_pre:", ske_label_pre.shape)


        save_path = os.path.join(save_root, block_name)
        save_image_path = os.path.join(save_path, "image")
        save_label_path = os.path.join(save_path, "label")
        save_label_pre_path = os.path.join(save_path, "label_pre")
        save_ske_label = os.path.join(save_path, "ske_label")
        save_ske_label_pre = os.path.join(save_path, "ske_label_pre")
        save_image_3d(image_3d=image, image_save_root=save_image_path, suffix=".tiff")
        save_image_3d(image_3d=label, image_save_root=save_label_path, suffix=".tiff")
        save_image_3d(image_3d=block_label_pre, image_save_root=save_label_pre_path, suffix=".tiff")
        save_image_3d(image_3d=(ske_label.astype(np.uint8))*255, image_save_root=save_ske_label, suffix=".tiff")
        save_image_3d(image_3d=(ske_label_pre.astype(np.uint8))*255, image_save_root=save_ske_label_pre, suffix=".tiff")




