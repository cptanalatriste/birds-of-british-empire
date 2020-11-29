from typing import Tuple

from torch import nn, Tensor
from torch.autograd import Variable

from attnganw import config
from attnganw.random import get_single_normal_vector, get_zeroes


class ConditioningAugmentationWrapper:

    def __init__(self, conditioning_augmentation_net: nn.Module, gpu_id: int):
        self.conditioning_augmentation_net: nn.Module = conditioning_augmentation_net
        self.gpu_id: int = gpu_id

    def convert_to_conditioning_vector(self, sentence_vector: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        conditioning_vector, mean, diag_covariance_matrix = self.forward(sentence_vector=sentence_vector)
        return conditioning_vector, mean, diag_covariance_matrix

    def forward(self, sentence_vector: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, diag_covariance_matrix = self.conditioning_augmentation_net.encode(sentence_vector)

        if config.generation['do_conditioning_augmentation']:
            epsilon = get_single_normal_vector(shape=diag_covariance_matrix.size(), gpu_id=self.gpu_id)[0]
        else:
            epsilon = get_zeroes(shape=diag_covariance_matrix.size(), gpu_id=self.gpu_id)

        epsilon_variable = Variable(epsilon)

        conditioning_vector: Tensor = re_parametrise(mean=mean, diag_covariance_matrix=diag_covariance_matrix,
                                                     epsilon_variable=epsilon_variable)
        return conditioning_vector, mean, diag_covariance_matrix


def re_parametrise(mean: Tensor, diag_covariance_matrix: Tensor, epsilon_variable: Tensor) -> Tensor:
    standard_deviation = diag_covariance_matrix.mul(0.5).exp_()
    return epsilon_variable.mul(standard_deviation).add_(mean)
