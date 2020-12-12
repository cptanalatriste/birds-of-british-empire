import logging
from typing import List

from attnganw.runner import generate_images
from attnganw.train import BirdGenerationFromCaption
import pandas as pd

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    run_config_file: str = 'cfg/eval_bird.yml'
    run_gpu_id: int = 0
    run_random_seed: int = 100
    generated_files: List[BirdGenerationFromCaption] = generate_images(config_file=run_config_file, gpu_id=run_gpu_id,
                                                                       random_seed=run_random_seed)

    metadata_file: str = 'metadata_file.csv'
    pd.DataFrame(generated_files).to_csv(metadata_file)
    logging.info("Metadata file written as CSV at {}".format(metadata_file))
