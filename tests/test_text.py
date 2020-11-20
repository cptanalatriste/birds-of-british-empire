from typing import Dict, List
from unittest import TestCase

from attnganw.text import TextProcessor


class TestTextProcessor(TestCase):

    def test_to_number_vector(self):
        word_to_index: Dict[str, int] = {"this": 0,
                                         "bird": 1,
                                         "is": 2,
                                         "red": 3}
        text_processor: TextProcessor = TextProcessor(word_to_index=word_to_index)
        output_vector: List[int] = text_processor.to_number_vector(text_to_encode="this bird is red")
        expected_output: List[int] = [0, 1, 2, 3]

        self.assertEqual(output_vector, expected_output)
