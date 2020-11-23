from typing import Dict, List
import logging

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from trainer import condGANTrainer
from datasets import TextDataset

from attnganw.text import TextProcessor, directory_to_trainer_input


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
        ratio = 0

        logging.debug("ratio " + str(ratio))
        new_noise_vector: Tensor = noise_vector_start * (1 - ratio) + noise_vector_end * ratio
        if gpu_id >= 0:
            new_noise_vector = new_noise_vector.cuda()
        noise_vectors.append(new_noise_vector)

    return noise_vectors
