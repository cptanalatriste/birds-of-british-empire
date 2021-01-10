import logging
from typing import List
import imageio

from attnganw import config
from attnganw.randomutils import set_random_seed
from attnganw.runner import generate_images
from attnganw.train import BirdGenerationFromCaption

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    run_config_file: str = 'cfg/eval_bird.yml'
    run_gpu_id: int = 0
    current_caption: str = 'the bird has a yellow crown and a black eye ring that is round'
    identifier: str = 'large_size_interpolation'
    gif_filename: str = identifier + '.gif'
    frame_duration = 0.1

    config.generation['noise_interpolation_file'] = 'D:\git\AttnGAN\\artifacts\size_large_boundary.npy'
    config.generation['noise_interpolation_enabled'] = True
    config.generation['do_conditioning_augmentation'] = False
    config.generation['noise_interpolation_start'] = -10.0
    config.generation['noise_interpolation_end'] = 10.0
    config.generation['noise_interpolation_steps'] = 20

    set_random_seed(random_seed=100)

    batch_generated_files: List[BirdGenerationFromCaption] = generate_images(config_file=run_config_file,
                                                                             gpu_id=run_gpu_id,
                                                                             identifier=identifier,
                                                                             caption_list=[current_caption])

    filenames_for_gif: List[str] = [generated_file.image_from_generator_2
                                    for generated_file in batch_generated_files]

    images_for_gif = []
    for filename in filenames_for_gif:
        images_for_gif.append(imageio.imread(filename))

    imageio.mimsave(gif_filename, images_for_gif, duration=frame_duration)
    logging.info('Interpolation GIF saved at {}'.format(gif_filename))
