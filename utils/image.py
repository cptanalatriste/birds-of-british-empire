from typing import List

from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt


def denormalize_and_show(images_tensor: Tensor, means: List[float], standard_devs: List[float],
                         filename: str, title: str = None) -> None:
    images_to_plot: np.ndarray = images_tensor.numpy().transpose((1, 2, 0))
    mean_vector: np.ndarray = np.array(means)
    stdev_vector: np.ndarray = np.array(standard_devs)

    images_to_plot = stdev_vector * images_to_plot + mean_vector
    if title is not None:
        plt.title(title)

    plt.imsave(filename)
