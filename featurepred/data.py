import logging
import shutil
import traceback
from typing import List

import pandas as pd
import numpy as np
from PIL import Image
from pandas import DataFrame, Series
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, Compose

ATTRIBUTE_ID_COLUMN: str = 'attribute_id'
CERTAINTY_ID_COLUMN: str = 'certainty_id'
IMAGE_ID_COLUMN: str = 'image_id'
IS_PRESENT_COLUMN: str = 'is_present'
IMAGE_NAME_COLUMN: str = 'image_name'
IS_TRAINING_IMAGE_COLUMN: str = 'is_training_image'

PRESENT_ATTRIBUTE_VALUE: int = 1
ABSENT_ATTRIBUTE_VALUE: int = 0

RESNET50_MEANS: List[float] = [0.485, 0.456, 0.406]
RESNET50_STD_DEVS: List[float] = [0.229, 0.224, 0.225]


class ResNet50DataLoaderBuilder:

    def __init__(self, image_folder: str, batch_size: int, input_resize: int, is_training: bool,
                 data_loader_workers: int):

        if is_training:
            self.image_transformations: Compose = ResNet50DataLoaderBuilder.get_training_transformation(
                input_resize=input_resize)
        else:
            self.image_transformations: Compose = ResNet50DataLoaderBuilder.get_validation_transformation(
                input_resize=input_resize)

        self.image_folder: ImageFolder = ImageFolder(root=image_folder, transform=self.image_transformations,
                                                     is_valid_file=can_open_image_file)
        self.class_names = self.image_folder.classes
        self.batch_size: int = batch_size
        self.data_loader_workers: int = data_loader_workers

    @staticmethod
    def get_validation_transformation(input_resize: int) -> Compose:
        return transforms.Compose([
            transforms.Resize(size=(input_resize, input_resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=RESNET50_MEANS,
                                 std=RESNET50_STD_DEVS)
        ])

    @staticmethod
    def get_training_transformation(input_resize: int) -> Compose:
        return transforms.Compose([
            transforms.Resize(size=(input_resize, input_resize)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RESNET50_MEANS,
                                 std=RESNET50_STD_DEVS)
        ])

    def build(self) -> DataLoader:
        sampler = get_weighted_sampler(self.image_folder)
        return DataLoader(dataset=self.image_folder, batch_size=self.batch_size, sampler=sampler,
                          num_workers=self.data_loader_workers)


def get_weighted_sampler(image_folder: ImageFolder) -> WeightedRandomSampler:
    class_per_image: List[int] = image_folder.targets
    num_samples: int = len(class_per_image)
    classes, counts = np.unique(class_per_image, return_counts=True)
    logging.info("classes: {} counts: {} num_samples: {}".format(classes, counts, num_samples))

    class_weights: List[float] = [sum(counts) / class_count for class_count in counts]
    image_weights: List[float] = [class_weights[image_class] for image_class in class_per_image]
    return WeightedRandomSampler(weights=image_weights, num_samples=num_samples)


def can_open_image_file(image_path: str):
    try:
        Image.open(image_path)
        return True
    except Exception:
        logging.error(traceback.format_exc())
        return False


class BirdDatasetRepository:

    def __init__(self, attributes_data_file: str, images_data_file: str, certainty_id: int,
                 split_data_file: str, is_training: bool):
        self.attributes_dataframe: DataFrame = pd.read_csv(attributes_data_file, sep='\s+', header=None,
                                                           error_bad_lines=False,
                                                           warn_bad_lines=False, usecols=[0, 1, 2, 3],
                                                           names=[IMAGE_ID_COLUMN, ATTRIBUTE_ID_COLUMN,
                                                                  IS_PRESENT_COLUMN,
                                                                  CERTAINTY_ID_COLUMN])

        self.split_dataframe: DataFrame = pd.read_csv(split_data_file, sep='\s+', header=None,
                                                      error_bad_lines=False,
                                                      warn_bad_lines=False,
                                                      index_col=0,
                                                      usecols=[0, 1],
                                                      names=[IMAGE_ID_COLUMN, IS_TRAINING_IMAGE_COLUMN])

        self.filter_by_train_test_split(is_training=is_training)
        self.filter_attributes_by_certainty_id(certainty_id=certainty_id)
        self.attributes_dataframe = self.attributes_dataframe.pivot(index=IMAGE_ID_COLUMN, columns=ATTRIBUTE_ID_COLUMN,
                                                                    values=IS_PRESENT_COLUMN)

        self.images_dataframe: DataFrame = pd.read_csv(images_data_file, sep='\s+', header=None,
                                                       error_bad_lines=False,
                                                       warn_bad_lines=False,
                                                       index_col=0,
                                                       usecols=[0, 1],
                                                       names=[IMAGE_ID_COLUMN, IMAGE_NAME_COLUMN])

    def filter_attributes_by_certainty_id(self, certainty_id: int) -> None:
        self.attributes_dataframe = self.attributes_dataframe.loc[
            self.attributes_dataframe[CERTAINTY_ID_COLUMN] == certainty_id]

    def filter_by_train_test_split(self, is_training: bool):

        flag_value: int = ABSENT_ATTRIBUTE_VALUE
        if is_training:
            flag_value = PRESENT_ATTRIBUTE_VALUE

        images_to_include: Series = self.split_dataframe[
            self.split_dataframe[IS_TRAINING_IMAGE_COLUMN] == flag_value].index

        self.attributes_dataframe = self.attributes_dataframe.loc[
            self.attributes_dataframe[IMAGE_ID_COLUMN].isin(images_to_include)]

    def get_images_by_attribute_value(self, attribute_id: int, attribute_value: int) -> DataFrame:
        matching_images_dataframe: DataFrame = self.attributes_dataframe.loc[
            self.attributes_dataframe[attribute_id] == attribute_value]

        logging.info("Images with value {} for attribute {}: {}".format(attribute_value, attribute_id,
                                                                        len(matching_images_dataframe.index)))

        return matching_images_dataframe

    def get_name_from_image_id(self, image_id) -> str:
        if image_id in self.images_dataframe.index:
            image_name: str = self.images_dataframe.loc[image_id][IMAGE_NAME_COLUMN]
            return image_name

        logging.error("Image Id {} not found".format(image_id))
        return ''


class ImageFolderBuilder:

    def __init__(self, attributes_data_file: str, images_data_file: str, image_directory: str, split_data_file: str):
        self.attributes_data_file = attributes_data_file
        self.image_directory = image_directory
        self.images_data_file = images_data_file
        self.split_data_file = split_data_file

    def build(self, attribute_id: int, certainty_id: int, positive_image_folder: str, negative_image_folder: str,
              is_training: bool):
        self.copy_by_attribute_value(attribute_id=attribute_id, attribute_value=PRESENT_ATTRIBUTE_VALUE,
                                     target_directory=positive_image_folder, certainty_id=certainty_id,
                                     is_training=is_training)
        self.copy_by_attribute_value(attribute_id=attribute_id, attribute_value=ABSENT_ATTRIBUTE_VALUE,
                                     target_directory=negative_image_folder, certainty_id=certainty_id,
                                     is_training=is_training)

    def copy_by_attribute_value(self, attribute_id: int, attribute_value: int, target_directory: str,
                                certainty_id: int, is_training: bool):
        bird_repository: BirdDatasetRepository = BirdDatasetRepository(attributes_data_file=self.attributes_data_file,
                                                                       images_data_file=self.images_data_file,
                                                                       certainty_id=certainty_id,
                                                                       split_data_file=self.split_data_file,
                                                                       is_training=is_training)

        images_dataframe: DataFrame = bird_repository.get_images_by_attribute_value(attribute_id=attribute_id,
                                                                                    attribute_value=attribute_value)

        for index, image_row in images_dataframe.iterrows():
            image_name: str = bird_repository.get_name_from_image_id(image_id=index)
            original_file: str = self.image_directory + image_name
            copy_file(original_file=original_file, destination_folder=target_directory)

        logging.info("{} images copied to {}".format(len(images_dataframe.index), target_directory))


def copy_file(original_file: str, destination_folder: str):
    shutil.copy2(src=original_file, dst=destination_folder)
