from typing import Dict, List
import logging

from torch.utils.data import DataLoader
import numpy as np

from trainer import condGANTrainer
from datasets import TextDataset

from attnganw.text import TextProcessor


class GanTrainerWrapper:

    def __init__(self, output_directory: str, text_data_set: TextDataset, batch_size: int,
                 shuffle_data_loader: bool, data_loader_workers: int, split_directory: str):
        vocabulary_size: int = text_data_set.n_words
        index_to_word: Dict[int, str] = text_data_set.ixtoword

        data_loader: DataLoader = DataLoader(dataset=text_data_set,
                                             batch_size=batch_size,
                                             drop_last=True,
                                             shuffle=shuffle_data_loader,
                                             num_workers=data_loader_workers)

        self.data_split: str = split_directory
        self.word_to_index: Dict[str, int] = text_data_set.wordtoix
        self.gan_trainer: condGANTrainer = condGANTrainer(output_dir=output_directory,
                                                          data_loader=data_loader,
                                                          n_words=vocabulary_size,
                                                          ixtoword=index_to_word)

    def train(self):
        self.gan_trainer.train()

    def sample(self):
        self.gan_trainer.sampling(self.data_split)

    def generate_examples(self, data_directory: str):
        generate_examples(self.word_to_index, self.gan_trainer, data_directory)


def generate_examples(word_to_index: Dict[str, int], gan_trainer: condGANTrainer, data_directory: str):
    logging.basicConfig(level=logging.DEBUG)

    '''generate images from example sentences'''
    filepath = '%s/example_filenames.txt' % data_directory
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (data_directory, name)
            with open(filepath, "r") as f:
                print('Load examples from:', name)
                sentences = f.read().split('\n')
                # a list of indices for a sentence
                captions: List[List[int]] = []
                cap_lens: List[int] = []

                text_processor: TextProcessor = TextProcessor(word_to_index=word_to_index)
                for sent in sentences:

                    rev: List[int] = text_processor.to_number_vector(text_to_encode=sent)
                    if len(rev) > 0:
                        captions.append(rev)
                        cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    gan_trainer.generate_examples(data_dic)
