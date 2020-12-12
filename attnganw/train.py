import logging
from typing import Dict, List, NamedTuple

import numpy as np
from datasets import TextDataset
from torch import Tensor
from torch.utils.data import DataLoader
from trainer import condGANTrainer

from attnganw import config
from attnganw.randomutils import get_single_normal_vector, get_vector_interpolation
from attnganw.text import TextProcessor, directory_to_trainer_input, get_lines_from_file, caption_list_to_trainer_input


class BirdGenerationFromCaption(NamedTuple):
    file_as_key: str
    noise_vector: np.ndarray
    caption_index: int
    attention_map_0: str
    attention_map_1: str
    image_from_generator_0: str
    image_from_generator_1: str
    image_from_generator_2: str


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
        self.text_processor: TextProcessor = TextProcessor(word_to_index=self.word_to_index)

    def train(self):
        self.gan_trainer.train()

    def sample(self):
        self.gan_trainer.sampling(self.data_split)

    def generate_from_caption_list(self, identifier: str, caption_list: List[str]) -> List[BirdGenerationFromCaption]:

        trainer_input: List[np.ndarray] = caption_list_to_trainer_input(caption_list=caption_list,
                                                                        text_processor=self.text_processor)
        captions_per_file: Dict[str, List] = {identifier: trainer_input}
        generated_images_data = self.gan_trainer.generate_examples(captions_per_file=captions_per_file,
                                                                   noise_vector_generator=default_noise_vector_generator)

        logging.info("{} captions processed.".format(len(generated_images_data)))
        return [BirdGenerationFromCaption(**caption_metadata) for caption_metadata in generated_images_data]

    def generate_from_caption_files(self, data_directory: str) -> List[BirdGenerationFromCaption]:

        file_names: List[str]
        if config.generation['caption_file']:
            file_names = [config.generation['caption_file']]
        else:
            list_of_files_path: str = '%s/example_filenames.txt' % data_directory
            file_names: List[str] = get_lines_from_file(list_of_files_path)
            file_names = ['%s/%s.txt' % (data_directory, file_name) for file_name in file_names]

        captions_per_file: Dict[str, List] = directory_to_trainer_input(file_names=file_names,
                                                                        text_processor=self.text_processor)

        generated_images_data: List[Dict]
        if config.generation['do_noise_interpolation']:
            logging.info("Performing noise interpolation")
            generated_images_data = self.gan_trainer.generate_examples(captions_per_file=captions_per_file,
                                                                       noise_vector_generator=get_vector_interpolation)
        else:
            generated_images_data = self.gan_trainer.generate_examples(captions_per_file=captions_per_file,
                                                                       noise_vector_generator=default_noise_vector_generator)

        logging.info("{} captions processed.".format(len(generated_images_data)))
        return [BirdGenerationFromCaption(**caption_metadata) for caption_metadata in generated_images_data]


def default_noise_vector_generator(batch_size: int, noise_vector_size: int, gpu_id: int) -> List[Tensor]:
    return get_single_normal_vector(shape=(batch_size, noise_vector_size), gpu_id=gpu_id)
