from typing import List, Dict

from nltk import RegexpTokenizer


class TextProcessor:

    def __init__(self, word_to_index: Dict[str, int]):
        self.tokenizer: RegexpTokenizer = RegexpTokenizer(r'\w+')
        self.word_to_index = word_to_index

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

        return number_vector
