import logging
from typing import List, Dict

import numpy as np
from nltk import RegexpTokenizer


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


def directory_to_trainer_input(file_names: List[str], text_processor: TextProcessor) -> Dict[str, List]:
    """generate images from example sentences"""
    captions_per_file: Dict[str, List] = {}
    for file_name in file_names:
        print('Load examples from:', file_name)
        sentences = get_lines_from_file(file_name)
        # a list of indices for a sentence
        captions: List[List[int]] = []
        caption_lengths: List[int] = []

        for sent in sentences:

            rev: List[int] = text_processor.to_number_vector(text_to_encode=sent)
            if len(rev) > 0:
                captions.append(rev)
                caption_lengths.append(len(rev))
        max_len = np.max(caption_lengths)

        sorted_indices = np.argsort(caption_lengths)[::-1]
        caption_lengths = np.asarray(caption_lengths)
        caption_lengths = caption_lengths[sorted_indices]
        cap_array = np.zeros((len(captions), max_len), dtype='int64')
        for i in range(len(captions)):
            idx = sorted_indices[i]
            cap = captions[idx]
            c_len = len(cap)
            cap_array[i, :c_len] = cap
        file_as_key = file_name[(file_name.rfind('/') + 1):]
        captions_per_file[file_as_key] = [cap_array, caption_lengths, sorted_indices]
    return captions_per_file


def get_lines_from_file(file_name: str) -> List[str]:
    with open(file_name, "r") as file:
        lines: List[str] = file.read().split('\n')
        return [line for line in lines if len(line) > 0]
