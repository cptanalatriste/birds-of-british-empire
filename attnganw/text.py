from typing import List, Dict

from nltk import RegexpTokenizer
import numpy as np
import logging


class TextProcessor:

    def __init__(self, word_to_index: Dict[str, int] = None, index_to_word: Dict[int, str] = None):
        self.tokenizer: RegexpTokenizer = RegexpTokenizer(r'\w+')
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word

    def to_number_vector(self, text_to_encode: str) -> List[int]:
        if len(text_to_encode) == 0:
            return []

        text_to_encode = text_to_encode.replace("\\ufffd\\ufffd", " ")
        text_tokens: List[str] = self.tokenizer.tokenize(text_to_encode.lower())

        if len(text_tokens) == 0:
            return []

        number_vector: List[int] = []
        for token in text_tokens:
            token = token.encode('ascii', 'ignore').decode('ascii')

            if len(token) > 0 and token in self.word_to_index:
                number_vector.append(self.word_to_index[token])

        logging.debug("Input sentence " + text_to_encode + " Number vector " + str(number_vector))
        return number_vector

    def to_word_vector(self, number_vector: np.ndarray) -> List[str]:
        word_vector: List[str] = []

        for word_index in range(len(number_vector)):

            word_as_number: int = number_vector[word_index]
            if word_as_number == 0:
                break

            word_as_string: str = self.index_to_word[word_as_number].encode('ascii', 'ignore').decode('ascii')
            word_vector.append(word_as_string)

        return word_vector
