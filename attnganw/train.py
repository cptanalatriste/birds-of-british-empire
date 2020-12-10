import logging
from typing import Dict, List, NamedTuple

from datasets import TextDataset
from torch import Tensor
from torch.utils.data import DataLoader
from trainer import condGANTrainer

from attnganw import config
from attnganw.random import get_single_normal_vector, get_vector_interpolation
from attnganw.text import TextProcessor, directory_to_trainer_input, get_lines_from_file


class BirdGenerationResult(NamedTuple):
    noise_vector: Tensor
    attention_map_0_file: str
    attention_map_1_file: str
    generator_output_0_file: str
    generator_output_2_file: str
    generator_output_3_file: str


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

        file_names: List[str]
        if config.generation['caption_file']:
            file_names = [config.generation['caption_file']]
        else:
            list_of_files_path: str = '%s/example_filenames.txt' % data_directory
            file_names: List[str] = get_lines_from_file(list_of_files_path)
            file_names = ['%s/%s.txt' % (data_directory, file_name) for file_name in file_names]

        captions_per_file: Dict[str, List] = directory_to_trainer_input(file_names=file_names,
                                                                        text_processor=text_processor)

        generated_images_data: List[Dict]
        if config.generation['do_noise_interpolation']:
            logging.info("Performing noise interpolation")
            generated_images_data = self.gan_trainer.generate_examples(captions_per_file=captions_per_file,
                                                                       noise_vector_generator=get_vector_interpolation)
        else:
            generated_images_data = self.gan_trainer.generate_examples(captions_per_file=captions_per_file,
                                                                       noise_vector_generator=default_noise_vector_generator)

        logging.info("{} files processed for image generation".format(len(generated_images_data)))


def default_noise_vector_generator(batch_size: int, noise_vector_size: int, gpu_id: int) -> List[Tensor]:
    return get_single_normal_vector(shape=(batch_size, noise_vector_size), gpu_id=gpu_id)
