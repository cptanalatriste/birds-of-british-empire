from typing import Dict, List
import logging

import torch
from torch import Tensor
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
        logging.basicConfig(level=logging.DEBUG)
        text_processor: TextProcessor = TextProcessor(word_to_index=self.word_to_index)

        captions_per_file: Dict[str, List] = directory_to_trainer_input(data_directory=data_directory,
                                                                        text_processor=text_processor)

        # self.gan_trainer.generate_examples(captions_per_file=captions_per_file,
        #                                    noise_vector_generator=get_single_noise_vector)
        self.gan_trainer.generate_examples(captions_per_file=captions_per_file,
                                           noise_vector_generator=get_noise_interpolation)


def get_single_noise_vector(batch_size: int, noise_vector_size: int, gpu_id: int) -> List[Tensor]:
    noise_vector = torch.FloatTensor(batch_size, noise_vector_size)
    if gpu_id >= 0:
        noise_vector = noise_vector.cuda()
    noise_vector.data.normal_(mean=0, std=1)

    return [noise_vector]


def get_noise_interpolation(batch_size: int, noise_vector_size: int, gpu_id: int,
                            noise_vector_start: Tensor = None,
                            noise_vector_end: Tensor = None,
                            number_of_steps=4) -> List[Tensor]:
    if noise_vector_start is None:
        noise_vector_start: Tensor = torch.randn(batch_size, noise_vector_size, dtype=torch.float)

    if noise_vector_end is None:
        noise_vector_end: Tensor = torch.randn(batch_size, noise_vector_size, dtype=torch.float)

    noise_vectors: List[Tensor] = []
    for vector_index in range(number_of_steps + 1):
        ratio: float = vector_index / float(number_of_steps)
        print("ratio " + str(ratio))
        new_noise_vector: Tensor = noise_vector_start * (1 - ratio) + noise_vector_end * ratio
        if gpu_id >= 0:
            new_noise_vector = new_noise_vector.cuda()
        noise_vectors.append(new_noise_vector)

    return noise_vectors


def directory_to_trainer_input(data_directory: str, text_processor: TextProcessor) -> Dict[str, List]:
    """generate images from example sentences"""
    list_of_files_path = '%s/example_filenames.txt' % data_directory
    captions_per_file: Dict[str, List] = {}
    with open(list_of_files_path, "r") as list_file:
        filenames = list_file.read().split('\n')
        for file_name in filenames:
            if len(file_name) == 0:
                continue
            file_path = '%s/%s.txt' % (data_directory, file_name)
            with open(file_path, "r") as file:
                print('Load examples from:', file_name)
                sentences = file.read().split('\n')
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
