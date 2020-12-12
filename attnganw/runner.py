import random
from datetime import datetime
from typing import List

import dateutil
import numpy as np
import torch
from datasets import TextDataset
from miscc.config import cfg_from_file, cfg
from torchvision.transforms import transforms

from attnganw.train import GanTrainerWrapper, BirdGenerationFromCaption


def set_random_seed(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if cfg.CUDA:
        torch.cuda.manual_seed_all(random_seed)


def get_text_dataset(tree_base_size: int, tree_branch_number: int, dataset_split: str,
                     data_directory: str) -> TextDataset:
    image_size = tree_base_size * (2 ** (tree_branch_number - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(image_size * 76 / 64)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(data_directory, dataset_split,
                          base_size=tree_base_size,
                          transform=image_transform)

    return dataset


def get_output_directory(dataset_name: str, config_name: str) -> str:
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_directory = '../output/%s_%s_%s' % \
                       (dataset_name, config_name, timestamp)

    return output_directory


def generate_images(config_file: str, gpu_id: int, random_seed: int, identifier: str, caption_list: List[str]) -> List[
    BirdGenerationFromCaption]:
    cfg_from_file(config_file)
    cfg.GPU_ID = gpu_id

    set_random_seed(random_seed)

    if cfg.CUDA:
        torch.cuda.manual_seed_all(random_seed)

    dataset_split: str = 'test'
    shuffle_data_loader: bool = True

    output_directory: str = get_output_directory(dataset_name=cfg.DATASET_NAME, config_name=cfg.CONFIG_NAME)
    text_dataset: TextDataset = get_text_dataset(tree_base_size=cfg.TREE.BASE_SIZE,
                                                 tree_branch_number=cfg.TREE.BRANCH_NUM,
                                                 dataset_split=dataset_split, data_directory=cfg.DATA_DIR)

    gan_trainer_wrapper: GanTrainerWrapper = GanTrainerWrapper(output_directory=output_directory,
                                                               text_data_set=text_dataset,
                                                               batch_size=cfg.TRAIN.BATCH_SIZE,
                                                               shuffle_data_loader=shuffle_data_loader,
                                                               data_loader_workers=int(cfg.WORKERS),
                                                               split_directory=dataset_split)

    return gan_trainer_wrapper.generate_from_caption_list(identifier=identifier, caption_list=caption_list)
