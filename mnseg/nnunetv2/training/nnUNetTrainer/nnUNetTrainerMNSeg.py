from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.MNSeg import get_mnseg_test_3d_from_plans


class nnUNetTrainerMNSeg(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        print(f"plans_manager:{plans_manager}")
        print(f"configuration_manager:{configuration_manager}")
        configuration_manager.configuration['UNet_base_num_features'] = 32
        configuration_manager.configuration['unet_max_num_features'] = 256

        if len(configuration_manager.patch_size) == 3:
            model = get_mnseg_test_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 3D models are supported")

        print("MNSeg: {}".format(model))

        return model
