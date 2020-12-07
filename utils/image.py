import logging
from typing import List, Dict

import torchvision
from matplotlib.axes import Axes
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
              standard_devs: List[float], file_name: str) -> None:
    image_grid: Tensor = torchvision.utils.make_grid(images, nrow=len(classes))
    labels: List[str] = [class_names[class_index] for class_index in classes]
    denormalize_and_show(images_tensor=image_grid, means=means,
                         standard_devs=standard_devs, title=str(labels))
    plt.savefig(fname=file_name, bbox_inches='tight')
    logging.info("Image saved at {}".format(file_name))


def plot_images_with_labels(images: Tensor, classes: Tensor, class_names: Dict[int, str], file_name: str,
                            means: List[float], standard_devs: List[float]) -> None:
    plt.figure()

    total_images: int = images.size()[0]

    for image_index in range(total_images):
        axis: Axes = plt.subplot(total_images // 2, 2, image_index + 1)
        axis.axis('off')
        axis.set_title('class: {}'.format(class_names[classes[image_index]]))
        denormalize_and_show(images_tensor=images.data[image_index], means=means, standard_devs=standard_devs)

    plt.savefig(fname=file_name, bbox_inches='tight')
    logging.info("Image saved at {}".format(file_name))
