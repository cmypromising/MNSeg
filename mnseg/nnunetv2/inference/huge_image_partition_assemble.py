"""
这个脚本对超大神经元图像进行划分、合并处理
处理的重点在于记录各个数据块的信息
"""

import os, cv2, sys
import numpy as np
from mnseg.nnunetv2.constant import ImageSize_NoMoreThan, ImageSize_PartitionTo, ImageSize_Overlap
from mnseg.nnunetv2.constant import Block_Size
from mnseg.nnunetv2.constant import ImageSuffixes
from mnseg.nnunetv2.constant import Image2DName_Length, Image3DName_Length
from datetime import datetime
from mnseg.nnunetv2.tools.printer import print_my

class NeuronImage_Huge():
    """
    这个类型定义一种超大的神经元图像，希望能在一般的配置条件下对其进行划分、分块分割、合并、组装等处理
    """
    def __init__(self, neuron_name, neuron_image_root):
        """
        这个超大的神经元图像由其根目录 root 唯一指代，
        root目录中直接保存文件夹 image (保存神经元二维图像序列)和信息文件 source.info (其中保存这个超大神经元图像的一些基本信息，如图像尺寸等)
        :param neuron_name: 这个超大尺寸神经元的名称
        :param root: 这个超大神经元图像的唯一标识
        """
        self.neuron_name = neuron_name
        self.neuron_image_root = neuron_image_root
        self.data_path = os.path.join(self.neuron_image_root, 'image')
        self.source_file = os.path.join(self.neuron_image_root, 'huge_image_source.info')    # 保存超大图像的尺寸信息，二维图像列表信息等
        self.partition_info_file = os.path.join(self.neuron_image_root, 'partition.info')    # 保存分割超大图像后各个子图像的名称、起始坐标、尺寸等信息
        self._check()
        self.partition_save_root = os.path.join(self.neuron_image_root, 'image_partition')
        self.assemble_save_root = os.path.join(self.neuron_image_root, 'image_assembled')
        self.subimage_size_list = list()     #存储划分出来的子图的大小
        self.subimage_location_list = list() #存储划分出来的子图的左上角坐标
        self.subimage_name_list = list()     #存储划分出来的子图的名称
        self.image3d_number_partitioned = 0     # 已经划分生成的三维子图像个数

    def _check(self):
        """
        这个函数检查当前神经元是否是超大型的
        :return:
        """
        if not os.path.isdir(self.data_path):
            raise EnvironmentError('给定的根目录 root ({}) 不存在或其中没有包含保存图像数据的 image 文件夹，请查证'.format(self.neuron_image_root))
        if not os.path.isfile(self.source_file):
            # 若 source_file 文件不存在，则生成该文件
            self._generate_source_file()
        self._read_basic_info()
        if (self.depth <= ImageSize_NoMoreThan[0]
                and self.height <= ImageSize_NoMoreThan[1]
                and self.width <= ImageSize_NoMoreThan[2]):
            self.is_huge = False
            print('当前神经元图像不是超大型的')
        else:
            self.is_huge = True

    def _read_basic_info(self):
        """
        从 source.info 中读取当前神经元图像的尺寸等基本信息
        :return:
        """
        source_file = open(self.source_file).readlines()
        assert 'neuron_image_root:' == source_file[0].split()[1]
        assert 'z:' == source_file[1].split()[1]
        assert 'y:' == source_file[2].split()[1]
        assert 'x:' == source_file[3].split()[1]
        assert self.neuron_image_root == source_file[0].split()[-1], \
            '当前神经元图像的唯一标识与其 source.info 文件中保存的不一致，请查证'
        self.depth  = int(source_file[1].split()[-1])
        self.height = int(source_file[2].split()[-1])
        self.width  = int(source_file[3].split()[-1])
        self.image2d_name_list = source_file[4:]
        self.image2d_name_list = [image2d_name.strip() for image2d_name in self.image2d_name_list]
        assert self.depth == len(self.image2d_name_list), '当前神经元图像的 source.info 文件中记录有错误，请查证'

    def _generate_source_file(self):
        """
        这个函数半自动生成保存当前神经元图像基本信息的 source.info 文件
        source.info 文件中保存当前神经元的尺寸信息和神经元图像数据的各个二维图像名称
        :return:
        """
        source_file = open(self.source_file, 'w')
        image_name_list = os.listdir(self.data_path)
        image_name_list.sort()      #神经元图像序列的名称是有序的，而且按照其在神经元图像中的实际先后顺序排列
                                    #这一点很重要，否则后续无法处理；若不满足这一点，请手动调整
        image_name_list = [image_name for image_name in image_name_list
                           if os.path.splitext(image_name)[-1] in ImageSuffixes]
        z_length = len(image_name_list)
        if z_length <= 0:
            raise EnvironmentError('image ({}) 文件夹中不美有包含神经元图像数据'.format(self.data_path))

        source_file.write('# neuron_image_root: {}\n'.format(self.neuron_image_root))
        source_file.write('# z: {}\n'.format(z_length))
        # 由于默认是超大尺寸的神经元图像，因此需要手动输入其宽、高
        print_my('请手动配置 source.info ({}) 中关于当前神经元图像尺寸的信息\n'.format(self.source_file))
        height = input('请输入当前神经元图像的高度： ')
        width = input('请输入当前神经元图像的宽度： ')
        source_file.write('# y: {}\n'.format(height))
        source_file.write('# x: {}\n'.format(width))

        self.image2d_suffix = ''
        for image_name in image_name_list:
            if self.image2d_suffix == '':
                self.image2d_suffix = os.path.splitext(image_name)[-1]
                print('当前神经元图像数据的二维图像序列的后缀名是: {}'.format(self.image2d_suffix))
            else:
                if self.image2d_suffix != os.path.splitext(image_name)[-1]:
                    print('当前神经元图像数据的二维图像序列的后缀名不一致，建议查证；若有必要，请删除 source 文件重新生成.')
            source_file.write(image_name + '\n')

        source_file.close()

    def partition(self, partition_save = None):
        """
        这个函数对超大尺寸的神经元图像进行划分处理，划分成一个个子图
        划分过程中，逐个读取当前神经元图像的二维图像，对每个二维图像序列按照 ImageSize_NoMoreThan 进行切分，切分后即进行保存，
        对当前二维图像处理完毕后，再读取处理下一张二维图像，
        直至所有二维图像被处理完毕
        :param partition_save: 子图的保存路径，默认是当前神经元图像 root 路径中的 image_partition 目录
        :return:
        """
        if not self.is_huge:
            print('当前神经元图像不是超大尺寸的，不需要被划分')
            return
        if os.path.isfile(self.partition_info_file):
            print_my('当前超大尺寸的神经元 {} 可能已经被切分了，请查证 ...'.format(self.neuron_name))
            self._load_partition_info()
            return
        self.partition_save_root = self.partition_save_root if partition_save is None else partition_save
        if not os.path.isdir(self.partition_save_root):
            os.mkdir(self.partition_save_root)

        self.image2d_number_partitioning = 0    # 已经被切分处理了的二维图像个数
        self.z_cutting = 0                      # 标定当前划分到的三维子图像的深度
        self.y_cutting = 0                      # 标定当前划分到的三维子图像的高度
        self.x_cutting = 0                      # 标定当前划分到的三维子图像的宽度
        while self.image2d_number_partitioning < self.depth:
            self._divide2d()
            time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
            sys.stdout.write('\r{}---- dividing {} / {} 2D image in huge neuron image {}'.
                      format(time, self.image2d_number_partitioning, self.depth, self.neuron_name))
            sys.stdout.flush()
        print_my('\n---- finish dividing ...\n')
        self._write_partition_info()

    def _write_partition_info(self):
        """
        将子图像信息写入硬盘，子图像信息包括子图像名称、子图像在原始超大图像中的起始坐标和尺寸
        :return:
        """
        file_info = open(self.partition_info_file, 'w')
        head_info = '# subimage_name'.rjust(40) \
                    + 'loc_y'.rjust(12) + 'loc_x'.rjust(12) \
                    + 'height'.rjust(12) + 'width'.rjust(12) + '\n'
        file_info.write(head_info)
        for index, subimage_name in enumerate(self.subimage_name_list):
            location = self.subimage_location_list[index]
            size = self.subimage_size_list[index]
            line_text = '{}{}{}{}{}\n'.format(subimage_name.rjust(40),
                                                  str(location[0]).rjust(12), str(location[1]).rjust(12),
                                                  str(size[0]).rjust(12), str(size[1]).rjust(12))
            file_info.write(line_text)
        file_info.close()

    def _divide2d(self):
        """
        切分单个二维图像
        :return:
        """
        image_name = os.path.join(self.data_path, self.image2d_name_list[self.image2d_number_partitioning])
        assert os.path.isfile(image_name), '文件 {} 不存在，请查证'.format(image_name)
        image2d = cv2.imread(image_name, -1)
        depth, height, width = ImageSize_PartitionTo
        depth = depth if self.depth > ImageSize_NoMoreThan[0] else self.depth
        height = height if self.height > ImageSize_NoMoreThan[1] else self.height
        width = width if self.width > ImageSize_NoMoreThan[2] else self.width       # 超大尺寸神经元划分后的实际目标大小
        if self.image2d_number_partitioning == 0:
            print_my('当前超大尺寸神经元图像尺寸为 ({},{},{})，'
                     '将要划分为目标大小为 ({},{},{}) 的神经元子图'.
                     format(self.depth, self.height, self.width, depth, height, width))
        image3d_number_partitioned = 0      # 当前二维图片切割出来的小图片个数，这些小图片应该属于某个三维图像块
        while True:
            image3d_current_number = self.image3d_number_partitioned + image3d_number_partitioned
            y_start = self.y_cutting if self.y_cutting + height < self.height \
                else (self.y_cutting if self.y_cutting + Block_Size[1] < self.height else self.height - Block_Size[1])
            y_stop  = self.y_cutting + height if self.y_cutting + height < self.height \
                else self.height
            x_start = self.x_cutting if self.x_cutting + width  < self.width  \
                else (self.x_cutting if self.x_cutting + Block_Size[2] < self.width  else self.width  - Block_Size[2])
            x_stop  = self.x_cutting + width  if self.x_cutting + width  < self.width  \
                else self.width
            subimage_name = '{}_part_{}'.format(self.neuron_name, str(image3d_current_number).zfill(Image3DName_Length))
            if self.z_cutting == 0:
                self.subimage_name_list.append(subimage_name)
                self.subimage_location_list.append((y_start, x_start))
                self.subimage_size_list.append((y_stop - y_start, x_stop - x_start))

            image2d_block = image2d[y_start:y_stop, x_start:x_stop]

            image3d_block_path = os.path.join(self.partition_save_root, subimage_name)
            if not os.path.isdir(image3d_block_path):
                os.mkdir(image3d_block_path)
            if not os.path.isdir(os.path.join(image3d_block_path, 'image')):
                os.mkdir(os.path.join(image3d_block_path, 'image'))
            image2d_name = os.path.join(image3d_block_path, 'image', '{}.tiff'.format(str(self.z_cutting).zfill(Image2DName_Length)))
            cv2.imwrite(image2d_name, image2d_block)

            image3d_number_partitioned += 1

            if self.x_cutting + width < self.width:
                self.x_cutting = self.x_cutting + width
            else:
                self.x_cutting = 0
                if self.y_cutting + height < self.height:
                    self.y_cutting = self.y_cutting + height
                else:
                    self.y_cutting = 0
                    break

        self.image2d_number_partitioning += 1
        self.z_cutting += 1                 #
        if self.z_cutting == depth or self.image2d_number_partitioning == self.depth:
            self.image3d_number_partitioned += image3d_number_partitioned       # 新的图像块
            self.z_cutting = 0                                                  # 每个图像块的二维图像从头开始编号

    def assemble(self, source_root = None, save_root = None, leaf_path = 'image'):
        """
        将划分开的神经元图像组装到一起
        :param source_root: 待组装的子图像所处的根目录，这个根目录中保存一个个子图像目录，每个子图像目录中保存名称为 leaf_path 的目录，其中保存一张张二维图片
        :param save_root: 整合组装后的超大二维图像的保存根目录，这个目录中保存名称为 leaf_path 的子目录，其中保存一张张组装得到的二维图片
        :param leaf_path: 子目录名称
        :return:
        """
        if (len(self.subimage_name_list) == 0 or
            len(self.subimage_location_list) == 0 or
            len(self.subimage_size_list) == 0 or
            len(self.subimage_name_list) != self.image3d_number_partitioned or
            len(self.subimage_location_list) != self.image3d_number_partitioned or
            len(self.subimage_size_list) != self.image3d_number_partitioned):
            #print('len(self.subimage_name_list) = {}, '
            #      'len(self.subimage_location_list) = {}, '
            #      'len(self.subimage_size_list) = {}, '
            #      'self.image3d_number_partitioned = {}'.format(len(self.subimage_name_list),
            #                                                    len(self.subimage_location_list),
            #                                                    len(self.subimage_size_list),
            #                                                    self.image3d_number_partitioned))
            self._load_partition_info()
        self.assemble_save_root = self.assemble_save_root if save_root is None else save_root
        source_root = self.partition_save_root if source_root is None else source_root
        if not os.path.isdir(self.assemble_save_root):
            os.makedirs(self.assemble_save_root)
        if not os.path.isdir(os.path.join(self.assemble_save_root, leaf_path)):
            os.mkdir(os.path.join(self.assemble_save_root, leaf_path))
        self.x_cutting = 0
        self.y_cutting = 0
        self.z_cutting = 0
        self.image2d_number_assembling = 0
        self.image3d_number_assembled = 0
        while self.image2d_number_assembling < self.depth:
            self._unite(source_root = source_root, leaf_path = leaf_path)
            time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
            sys.stdout.write('\r{}---- uniting {} / {} 2D image for huge neuron image {}'
                  .format(time, self.image2d_number_assembling, self.depth, self.neuron_name))
            sys.stdout.flush()
        print_my('\n---- finish assembling ...\n')

    def _load_partition_info(self):
        """
        加载子图像信息，子图像名称、在超大图像中的起始坐标、尺寸等
        :return:
        """
        self.subimage_name_list = list()
        self.subimage_location_list = list()
        self.subimage_size_list = list()
        self.image3d_number_partitioned = 0
        partition_info_list = open(self.partition_info_file, 'r').readlines()
        partition_info_list = [line.strip() for line in partition_info_list if not line.strip().startswith('#')]
        for index, partition_info in enumerate(partition_info_list):
            self.image3d_number_partitioned += 1
            elements = partition_info.split()
            subimage_name = elements[0]
            location_y = int(elements[1])
            location_x = int(elements[2])
            height = int(elements[3])
            width = int(elements[4])
            try:
                eles = subimage_name.split('_')
                index_ = int(eles[-1])
                if index_ != index:
                    raise AttributeError('出错了，子图像顺序不对；这会造成无法组装复原原始超大尺寸图像')
            except:
                raise AttributeError('出错了，子图像文件名格式不对，请查证')
            self.subimage_name_list.append(subimage_name)
            self.subimage_location_list.append((location_y, location_x))
            self.subimage_size_list.append((height, width))

    def _unite(self, source_root, leaf_path = 'image'):
        image_name = os.path.join(self.assemble_save_root, leaf_path, self.image2d_name_list[self.image2d_number_assembling])
        #print(image_name)
        image2d = None
        depth, height, width = ImageSize_PartitionTo
        depth = depth if self.depth > ImageSize_NoMoreThan[0] else self.depth
        height = height if self.height > ImageSize_NoMoreThan[1] else self.height
        width = width if self.width > ImageSize_NoMoreThan[2] else self.width       # 超大尺寸神经元划分后的实际目标大小
        image3d_number_assembled = 0  # 当前二维图片切割出来的小图片个数，这些小图片应该属于某个三维图像块
        while True:
            image2d_t = None
            while True:
                image3d_current_number = self.image3d_number_assembled + image3d_number_assembled
                y_start, x_start = self.subimage_location_list[image3d_current_number]
                height_block, width_block = self.subimage_size_list[image3d_current_number]
                y_stop = y_start + height_block
                x_stop = x_start + width_block
                subimage_name = self.subimage_name_list[image3d_current_number]
                image3d_block_path = os.path.join(source_root, subimage_name)
                image2d_name = os.path.join(image3d_block_path, leaf_path, '{}.tiff'.format(str(self.z_cutting).zfill(Image2DName_Length)))
                #print_my(image2d_name)
                image2d_block = cv2.imread(image2d_name, -1)
                image2d_t = image2d_block if image2d_t is None else np.hstack((image2d_t, image2d_block))
                image3d_number_assembled += 1
                if x_stop >= self.width:
                    break
            image2d = image2d_t if image2d is None else np.vstack((image2d, image2d_t))
            if y_stop >= self.height:
                break

        self.image2d_number_assembling += 1
        #print_my('image_name = {}, image2d.type is {}\n source_root = {}'.format(image_name, type(image2d), source_root))
        cv2.imwrite(image_name, image2d)
        self.z_cutting += 1  #
        if self.z_cutting == depth:
            self.image3d_number_assembled += image3d_number_assembled  # 新的图像块
            self.z_cutting = 0  # 每个图像块的二维图像从头开始编号
            # print('saving: part {}, slice {}'.format(image3d_number_partitioned, self.z_cutting))
            # input('请回车')
