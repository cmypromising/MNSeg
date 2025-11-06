"""
这个脚本保存项目涉及到的一些常数
"""
import torch, os

#神经元图像像素类别数，只有前景类和背景类两种
NUM_CLASSES = 2

# 神经元图像数据是以二维图像序列形式保存的，其后缀名是一致的
ImageSuffixes = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

# 需要被划分的尺寸，当神经元图像的任意一个维度大小超过这个尺寸的对应维度大小时候，将其进行划分
ImageSize_NoMoreThan = (256, 1024, 1024)    # 当图像任何一个尺寸超过这个大小时候，则认为其是超大型的，不适合低配置的计算机处理；
                                            # 将该图像划分后处理，划分大小为 ImageSice_PartitionTo
ImageSize_PartitionTo = (192, 768, 768)     # 将过大的图像划分的目标大小
#ImageSize_NoMoreThan = (128, 512, 512)
#ImageSize_PartitionTo = (96, 384, 384)
ImageSize_Overlap = (0, 0, 0)             # 超大尺寸图像在划分时候相邻子图像之间的重叠大小

Image2DName_Length = 6
Image3DName_Length = 4

#输入网络进行训练或去噪处理的图像块大小
Block_Size = (32, 128, 128)     #(z, y, x) -- depth, height, width
Block_Size_ = 32 * 128 * 128

Block_Size_CWMBS = (256, 256, 256)
#训练数据保存路径
DataTrain_Root = '/home/promising/NAS_DATA/datasets/bigneuron/DataBase_6_enhance_image'
# DataTrain_Root = '/home/promising/NAS_DATA/datasets/bigneuron/DataBase_6_enhance_image'
TrainSource = os.path.join(DataTrain_Root, 'train.txt')
TestSource = os.path.join(DataTrain_Root, 'test.txt')
Mean_TrainData = 0.029720289547802054   # 这是基于 BigNeuron 数据生成的图像块的均值和方差
Std_TrainData = 0.04219472495471814

#完整神经元图像保存路径
DataNeuron_Root = '/home/daixinle/NAS_DATA/data/bigneuron/origin_data/data' 

RESULT_SAVE_PATH = '/home/daixinle/NAS_DATA/data/bigneuron/neuron_denoised' if torch.cuda.is_available() \
    else 'D:\\data\\BigNeuron-test\\segmentation'
BLOCK_RESULT_SAVE_PATH = '/home/daixinle/NAS_DATA/data/bigneuron/neuron_block_denoised'

DataNeuron_Root_HUST = '/data/daixinle/DATA/BigNeuronData/NeuronData_HUST/NeuronImage_from_15107/0_original_data' if torch.cuda.is_available() \
    else 'in the MOBILE DISK'
RESULT_SAVE_PATH_HUST = '/home/promising/NAS_DATA/data/bigneuron/neuron_denoised' if torch.cuda.is_available() \
    else 'D:\\data\\BigNeuron-test\\segmentation'

ROOT_STANDARD = '/home/promising/NAS_DATA/data/bigneuron/origin_data/data' if torch.cuda.is_available() \
    else 'D:\\pyProject\\BigNeuron\\data'

MODEL_ROOT = '/home/promising/NAS_DATA/data/bigneuron/weight' if torch.cuda.is_available() else NotImplementedError

VAA3D_ROOT = 'D:\\V3d\\Vaa3D_V3.601_Windows_MSVC_64bit'
VAA3D_PATH = os.path.join(VAA3D_ROOT, 'vaa3d_msvc.exe')
VAA3D_PLUGINS = os.path.join(VAA3D_ROOT, 'plugins')


NEURON_NAME_LIST = ['000000', '000011', '000025', '000036', '000047', '000059', '000072', '000083', '000094',
                    '000001', '000012', '000026', '000037', '000048', '000060', '000073', '000084', '000095',
                    '000002', '000013', '000027', '000038', '000049', '000061', '000074', '000085',
                    '000003', '000015', '000028', '000039', '000050', '000064', '000075', '000086',
                    '000004', '000016', '000029', '000040', '000051', '000065', '000076', '000087',
                    '000005', '000017', '000030', '000041', '000052', '000066', '000077', '000088',
                    '000006', '000018', '000031', '000042', '000053', '000067', '000078', '000089',
                    '000007', '000019', '000032', '000043', '000054', '000068', '000079', '000090',
                    '000008', '000022', '000033', '000044', '000055', '000069', '000080', '000091',
                    '000009', '000023', '000034', '000045', '000056', '000070', '000081', '000092',
                    '000010', '000024', '000035', '000046', '000057', '000071', '000082', '000093']

NEURON_NAME_TRAIN = [
    '000000', '000001', '000002', '000004', '000005', '000006',
    '000010', '000013', '000016', '000019', '000020', '000023',
    '000024', '000025', '000027', '000028', '000029', '000030',
    '000032', '000033', '000034', '000035', '000036', '000038',
    '000039', '000040', '000041', '000042', '000043', '000045',
    '000048', '000049', '000052', '000053', '000059', '000061',
    '000062', '000064', '000066', '000067', '000068', '000070',
    '000071', '000072', '000073', '000074', '000075', '000077',
    '000078', '000080', '000081', '000082', '000084', '000085',
    '000086', '000087', '000088', '000089', '000091', '000092',
    '000093', '000095'
]

NeuronNet = ['mnseg']

NEURON_NAME_TEST = ['000011', '000015', '000018', '000031', '000047', '000050']

NEURON_NAME_TEST_FINAL = [ '000003', '000011', '000015', '000016', '000018',
                           '000022', '000026', '000029', '000040', '000047',
                           '000050', '000060', '000065', '000069', '000076', 
                           '000079', '000083', '000090', '000094' ]

NEURON_NOT_IN_TRAIN_TEST = ['000008', '000009', '000011', '000012', '000015', '000018',
                            '000031', '000046', '000047', '000050', '000058', '000063']


FUNC = {
    'APP2':         f'{VAA3D_PATH} /x {VAA3D_PLUGINS}\\neuron_tracing\\Vaa3D_Neuron2\\vn2.dll /f app2',
    'MST_Tracing':  f'{VAA3D_PATH} /x {VAA3D_PLUGINS}\\neuron_tracing\\MST_tracing\\neurontracing_mst.dll /f trace_mst',
    'XY_3D_TreMap': f'{VAA3D_PATH} /x {VAA3D_PLUGINS}\\neuron_tracing\\TReMap\\neurontracing_mip.dll /f trace_mip',
    'snake':        f'{VAA3D_PATH} /x {VAA3D_PLUGINS}\\neuron_tracing\\Vaa3D-FarSight_snake_tracing\\snake_tracing.dll /f snake_trace',
}
# FUNC 中的命令参数需要根据个人电脑中 Vaa3D 的配置路径进行修改
# 由于我的办公电脑中只有以上几个能使用批量化操作，且运行速度较快，因此我们只使用以上几个追踪方法进行对比