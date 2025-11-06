# MNSeg
MNSeg: Mamba-based 3D Neuron Segmentation Integrated with Bidirectional Attention Mechanism and Topological Loss.

- We propose the first Mamba-based deep network for 3D neuron segmentation.
- We introduce bidirectional attention and clDice loss to enhance the
neuron segmentation.
- Experiments demonstrate superior neuron segmentation and reconstruction performance.

# MNSeg
### MNSeg: Mamba-based 3D Neuron Segmentation Integrated with Bidirectional Attention Mechanism and Topological Loss.

- We propose the first Mamba-based deep network for 3D neuron segmentation.
- We introduce bidirectional attention and clDice loss to enhance the neuron segmentation.
- Experiments demonstrate superior neuron segmentation and reconstruction performance.

## Installation 

Requirements: `Ubuntu 20.04`, `CUDA 11.8`

1. Create a virtual environment: `conda create -n mnseg python=3.10 -y` and `conda activate mnseg `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d>=1.2.0` and `pip install mamba-ssm --no-cache-dir`
4. Download code: `git clone https://github.com/cmypromising/MNSeg.git`
5. `cd MNSeg/mnseg` and run `pip install -e .`


sanity test: Enter python command-line interface and run

```bash
import torch
import mamba_ssm
```

## Model Training
Dataset to be released. MNSeg is build on the popular [UMamba](https://github.com/bowang-lab/U-Mamba) framework. 

### Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Train MNSeg

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerMNSeg
```


## Inference
``` bash
python -m mnseg.main_denoise
```

## Remarks

1. Path settings

The default data directory for U-Mamba is preset to U-Mamba/data. Users with existing nnUNet setups who wish to use alternative directories for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` can easily adjust these paths in mnseg/nnunetv2/path.py to update your specific nnUNet data directory locations, as demonstrated below:

```python
# example
nnUNet_raw="/home/daixinle/NAS_DATA/datasets/nnunet/nnUNet_raw" 
nnUNet_preprocessed="/home/daixinle/NAS_DATA/nnunet/nnUNet_preprocessed" 
nnUNet_results="/home/daixinle/NAS_DATA/nnunet/nnUNet_results" 
```


## Acknowledgements

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Mamba](https://github.com/state-spaces/mamba) and [UMamba](https://github.com/bowang-lab/U-Mamba) for making their valuable code publicly available.

