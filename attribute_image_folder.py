import logging

from featurepred.data import ImageFolderBuilder

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    image_directory: str = 'data/birds/CUB_200_2011/images/'
    attributes_data_file: str = 'data/birds/CUB_200_2011/attributes/image_attribute_labels.txt'
    images_data_file: str = 'data/birds/CUB_200_2011/images.txt'
    positive_image_folder: str = 'data/feature_data/train/feature_positive'
    negative_image_folder: str = 'data/feature_data/train/feature_negative'

    definitely_certainty_id: int = 4
    bill_needle_attribute_id: int = 4
    image_folder_builder: ImageFolderBuilder = ImageFolderBuilder(attributes_data_file=attributes_data_file,
                                                                  images_data_file=images_data_file,
                                                                  image_directory=image_directory)

    image_folder_builder.build(attribute_id=bill_needle_attribute_id, certainty_id=definitely_certainty_id,
                               positive_image_folder=positive_image_folder, negative_image_folder=negative_image_folder)
