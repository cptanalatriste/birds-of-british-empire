from typing import Tuple

import torch
from torch import nn, Tensor
from torch.autograd import Variable


class ConditioningAugmentationWrapper:

    def __init__(self, conditioning_augmentation_net: nn.Module, is_cuda: bool):
        self.conditioning_augmentation_net: nn.Module = conditioning_augmentation_net
        self.is_cuda: bool = is_cuda

    def convert_to_conditioning_vector(self, sentence_vector: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        conditioning_vector, mean, diag_covariance_matrix = self.forward(sentence_vector=sentence_vector)
        return conditioning_vector, mean, diag_covariance_matrix

    def forward(self, sentence_vector: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, diag_covariance_matrix = self.conditioning_augmentation_net.encode(sentence_vector)

        if self.is_cuda:
            epsilon = torch.cuda.FloatTensor(diag_covariance_matrix.size()).normal_()
        else:
            epsilon = torch.FloatTensor(diag_covariance_matrix.size()).normal_()
        epsilon = Variable(epsilon)

        conditioning_vector: Tensor = re_parametrise(mean=mean, diag_covariance_matrix=diag_covariance_matrix,
                                                     epsilon=epsilon)
        return conditioning_vector, mean, diag_covariance_matrix


def re_parametrise(mean: Tensor, diag_covariance_matrix: Tensor, epsilon: Tensor) -> Tensor:
    standard_deviation = diag_covariance_matrix.mul(0.5).exp_()
    return epsilon.mul(standard_deviation).add_(mean)
