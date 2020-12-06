import glob
import logging
import os
import shutil

from PIL.Image import Image
from pandas import DataFrame
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, Compose
import pandas as pd

ATTRIBUTE_ID_COLUMN: str = 'attribute_id'
CERTAINTY_ID_COLUMN: str = 'certainty_id'
IMAGE_ID_COLUMN: str = 'image_id'
IS_PRESENT_COLUMN: str = 'is_present'
IMAGE_NAME_COLUMN: str = 'image_name'

PRESENT_ATTRIBUTE_VALUE: int = 1
ABSENT_ATTRIBUTE_VALUE: int = 0


class ResNet50DataLoaderBuilder:

    def __init__(self, image_folder: str, batch_size: int):
        self.image_transformations: Compose = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.image_folder: ImageFolder = ImageFolder(root=image_folder, transform=self.image_transformations,
                                                     is_valid_file=can_open_image_file)

        self.batch_size: int = batch_size

    def build(self) -> DataLoader:
        return DataLoader(dataset=self.image_folder, batch_size=self.batch_size, shuffle=True)


def can_open_image_file(image_path: str):
    try:
        Image.open(image_path)
        return True
    except:
        return False


class BirdDatasetRepository:

    def __init__(self, attributes_data_file: str, images_data_file: str, certainty_id: int):
        self.attributes_dataframe: DataFrame = pd.read_csv(attributes_data_file, sep='\s+', header=None,
                                                           error_bad_lines=False,
                                                           warn_bad_lines=False, usecols=[0, 1, 2, 3],
                                                           names=[IMAGE_ID_COLUMN, ATTRIBUTE_ID_COLUMN,
                                                                  IS_PRESENT_COLUMN,
                                                                  CERTAINTY_ID_COLUMN])

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

    def __init__(self, attributes_data_file: str, images_data_file: str, image_directory: str):
        self.attributes_data_file = attributes_data_file
        self.image_directory = image_directory
        self.images_data_file = images_data_file

    def build(self, attribute_id: int, certainty_id: int, positive_image_folder: str, negative_image_folder: str):
        self.copy_by_attribute_value(attribute_id=attribute_id, attribute_value=PRESENT_ATTRIBUTE_VALUE,
                                     target_directory=positive_image_folder, certainty_id=certainty_id)

        # self.copy_by_attribute_value(attribute_id=attribute_id, attribute_value=ABSENT_ATTRIBUTE_VALUE,
        #                              target_directory=negative_image_folder,
        #                              attributes_dataframe=attributes_dataframe,
        #                              images_dataframe=images_dataframe)

    def copy_by_attribute_value(self, attribute_id: int, attribute_value: int, target_directory: str,
                                certainty_id: int):
        bird_repository: BirdDatasetRepository = BirdDatasetRepository(attributes_data_file=self.attributes_data_file,
                                                                       images_data_file=self.images_data_file,
                                                                       certainty_id=certainty_id)

        images_dataframe: DataFrame = bird_repository.get_images_by_attribute_value(attribute_id=attribute_id,
                                                                                    attribute_value=attribute_value)

        for index, image_row in images_dataframe.iterrows():
            image_name: str = bird_repository.get_name_from_image_id(image_id=index)
            original_file: str = self.image_directory + image_name
            copy_file(original_file=original_file, destination_folder=target_directory)


def copy_file(original_file: str, destination_folder: str):
    shutil.copy2(src=original_file, dst=destination_folder)
