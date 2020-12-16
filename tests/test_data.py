from unittest import TestCase

from pandas import DataFrame

from featurepred.data import BirdDatasetRepository


class TestDataUtils(TestCase):

    def test_bird_dataset_repository(self):
        attributes_data_file: str = 'image_attribute_labels_test.txt'
        certainty_id: int = 4
        images_data_file: str = 'images_test.txt'
        split_data_file: str = 'train_test_split_test.txt'
        bird_repository: BirdDatasetRepository = BirdDatasetRepository(attributes_data_file=attributes_data_file,
                                                                       images_data_file=images_data_file,
                                                                       minimum_certainty_id=certainty_id,
                                                                       split_data_file=split_data_file,
                                                                       is_training=True)

        rows_in_repository: int = len(bird_repository.attributes_dataframe.index)
        self.assertTrue(rows_in_repository == 2, msg="Rows in repository are {}".format(rows_in_repository))

        attribute_two_value_one: DataFrame = bird_repository.get_images_by_attribute_value(attribute_id=2,
                                                                                           attribute_value=1)
        self.assertTrue(len(attribute_two_value_one.index), 1)

        first_image_name: str = bird_repository.get_name_from_image_id(image_id=1)
        self.assertEqual(first_image_name, "001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg")

        second_image_name: str = bird_repository.get_name_from_image_id(image_id=2)
        self.assertEqual(second_image_name, "001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg")
