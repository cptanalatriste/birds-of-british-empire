import logging
from typing import List

from attnganw.randomutils import set_random_seed
from attnganw.runner import generate_images
from attnganw.train import BirdGenerationFromCaption
import pandas as pd
import numpy as np

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)

    run_config_file: str = 'cfg/eval_bird.yml'
    run_gpu_id: int = 0
    run_random_seed: int = 100
    metadata_file: str = 'metadata_file.csv'
    noise_vector_file: str = "noise_vectors_array.npy"
    num_batches: int = 15
    captions_per_batch: int = 40
    # num_batches: int = 3
    # captions_per_batch: int = 2

    caption: str = "bird"
    noise_vector_size: int = 100

    set_random_seed(random_seed=100)

    num_images: int = num_batches * captions_per_batch
    next_image_position: int = 0
    all_noise_array: np.ndarray = np.zeros(shape=(num_images, noise_vector_size))
    all_generated_files: List[BirdGenerationFromCaption] = []

    for current_batch in range(num_batches):
        identifier: str = "caption_list_{}".format(current_batch)

        caption_list: List[str] = [caption for _ in range(captions_per_batch)]
        batch_generated_files: List[BirdGenerationFromCaption] = generate_images(config_file=run_config_file,
                                                                                 gpu_id=run_gpu_id,
                                                                                 identifier=identifier,
                                                                                 caption_list=caption_list)
        batch_generated_files = sorted(batch_generated_files,
                                       key=lambda result: result.caption_index)
        for generation_result in batch_generated_files:
            all_noise_array[next_image_position] = generation_result.noise_vector
            next_image_position += 1

        all_generated_files += batch_generated_files

    logging.info("Resulting noise vector shape: {}".format(all_noise_array.shape))
    logging.debug("Resulting noise vector : {}".format(all_noise_array))

    np.save(file=noise_vector_file, arr=all_noise_array)
    logging.info("Noise vector stored at {}".format(noise_vector_file))

    pd.DataFrame(all_generated_files).to_csv(metadata_file)
    logging.info("Metadata file written as CSV at {} for {} images".format(metadata_file, len(all_generated_files)))
