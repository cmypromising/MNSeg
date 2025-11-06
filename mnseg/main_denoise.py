"""
对神经元图像进行分割处理
"""

from mnseg.nnunetv2.inference.segmentation_neuron_image import denoise_neuron_test
import argparse
from mnseg.nnunetv2.constant import NeuronNet, NEURON_NAME_TEST_FINAL

def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--net_name', type = str, default = 'final_test', choices = NeuronNet)
    parser.add_argument('--model_path', type=str, default='segmentation model path..')
    parser.add_argument('--parameter', type=str, default='MNSeg')

    parser.add_argument('--batchsize', type=int, default=0)

    args = parser.parse_args()

    denoise_neuron_test(net_name=args.net_name,
                        model_path=args.model_path,
                        parameter=args.parameter,
                        neuron_name_list=NEURON_NAME_TEST_FINAL,
                        batchsize=args.batchsize,
                        final_process=True)

if __name__ == '__main__':
    main()
