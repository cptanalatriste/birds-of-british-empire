import logging
from typing import List

from attnganw.runner import generate_images
from attnganw.train import BirdGenerationFromCaption
import pandas as pd
import numpy as np

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    run_config_file: str = 'cfg/eval_bird.yml'
    run_gpu_id: int = 0
    run_random_seed: int = 100
    metadata_file: str = 'metadata_file.csv'
    noise_vector_file: str = "noise_vectors_array.npy"
    num_captions: int = 40
    caption: str = "bird"
    identifier: str = "caption_list"

    caption_list: List[str] = [caption for _ in range(num_captions)]
    # caption_list: List[str] = ["this bird is red with white and has a very short beak",
    #                            "the bird has a yellow crown and a black eyering that is round",
    #                            "this bird has a green crown black primaries and a white belly"]

    generated_files: List[BirdGenerationFromCaption] = generate_images(config_file=run_config_file, gpu_id=run_gpu_id,
                                                                       random_seed=run_random_seed,
                                                                       identifier=identifier,
                                                                       caption_list=caption_list)
    generated_files = sorted(generated_files, key=lambda generation_result: generation_result.caption_index)

    all_noise_vectors: List[np.ndarray] = [generation_result.noise_vector for generation_result in generated_files]
    noise_vectors_array: np.ndarray = np.stack(all_noise_vectors, axis=0)
    logging.info("Resulting noise vector shape: {}".format(noise_vectors_array.shape))
    logging.debug("Resulting noise vector : {}".format(noise_vectors_array))

    np.save(file=noise_vector_file, arr=noise_vectors_array)
    logging.info("Noise vector stored at {}".format(noise_vector_file))

    pd.DataFrame(generated_files).to_csv(metadata_file)
    logging.info("Metadata file written as CSV at {}".format(metadata_file))
