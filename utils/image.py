import logging
from typing import List, Dict

import torchvision
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt


def denormalize_and_show(images_tensor: Tensor, means: List[float], standard_devs: List[float],
                         title: str = None) -> None:
    images_to_plot: np.ndarray = images_tensor.numpy().transpose((1, 2, 0))
    mean_vector: np.ndarray = np.array(means)
    stdev_vector: np.ndarray = np.array(standard_devs)

    images_to_plot = stdev_vector * images_to_plot + mean_vector
    images_to_plot = np.clip(images_to_plot, a_min=0, a_max=1)
    if title is not None:
        plt.title(title)

    plt.imshow(images_to_plot)


def plot_grid(images: Tensor, classes: Tensor, class_names: Dict[int, str], means: List[float],
              standard_devs: List[float], file_name: str):
    image_grid: Tensor = torchvision.utils.make_grid(images, nrow=len(classes))
    labels: List[str] = [class_names[class_index] for class_index in classes]
    denormalize_and_show(images_tensor=image_grid, means=means,
                         standard_devs=standard_devs, title=str(labels))
    plt.savefig(fname=file_name, bbox_inches='tight')
    logging.info("Image save at {}".format(file_name))
