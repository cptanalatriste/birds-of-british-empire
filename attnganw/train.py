from typing import Dict, List
import logging

from torch import Tensor
from torch.utils.data import DataLoader

from trainer import condGANTrainer
from datasets import TextDataset

from attnganw.random import get_vector_interpolation, get_single_normal_vector
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

        self.gan_trainer.generate_examples(captions_per_file=captions_per_file,
                                           noise_vector_generator=default_noise_vector_generator)
        # self.gan_trainer.generate_examples(captions_per_file=captions_per_file,
        #                                    noise_vector_generator=get_vector_interpolation)


def default_noise_vector_generator(batch_size: int, noise_vector_size: int, gpu_id: int) -> List[Tensor]:
    return get_single_normal_vector(shape=(batch_size, noise_vector_size), gpu_id=gpu_id)
