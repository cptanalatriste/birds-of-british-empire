from typing import Dict, List
from unittest import TestCase

from attnganw.text import TextProcessor
import numpy as np


class TestTextProcessor(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTextProcessor, self).__init__(*args, **kwargs)
        word_to_index: Dict[str, int] = {"this": 1,
                                         "bird": 2,
                                         "is": 3,
                                         "red": 4}
        index_to_word: Dict[int, str] = {1: "this",
                                         2: "bird",
                                         3: "is",
                                         4: "red"}
        self.text_processor: TextProcessor = TextProcessor(word_to_index=word_to_index,
                                                           index_to_word=index_to_word)

    def test_to_number_vector(self):
        output_vector: List[int] = self.text_processor.to_number_vector(text_to_encode="this bird is red")
        expected_output: List[int] = [1, 2, 3, 4]

        self.assertEqual(output_vector, expected_output)

    def test_to_word_vector(self):
        number_vector: np.ndarray = np.asarray([1, 2, 3, 4])
        output_vector: List[str] = self.text_processor.to_word_vector(number_vector=number_vector)
        expected_output: List[str] = ["this", "bird", "is", "red"]

        self.assertEqual(output_vector, expected_output)
