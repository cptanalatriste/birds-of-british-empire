import logging
from typing import Dict, List

from torch import Tensor
import numpy as np
from PIL import Image

from miscc.utils import decode_attention_maps

from attnganw import config


class ImageDecoder:

    def __init__(self):
        self.generated_format = '%s_generator_%d.png'
        self.attention_map_format = '%s_attn_model_stage_%d.png'

    def decode_generated_images(self, caption_index: int, file_prefix: str, generated_images: Tensor) -> List[str]:
        generated_image_files: List[str] = []
        for generator_index in range(len(generated_images)):
            image_as_array: np.ndarray = generated_images[generator_index][caption_index].data.cpu().numpy()
            image_as_array = (image_as_array + 1.0) * 127.5
            image_as_array = image_as_array.astype(np.uint8)
            image_as_array = np.transpose(image_as_array, (1, 2, 0))
            pil_image: Image = Image.fromarray(image_as_array)
            file_path = self.generated_format % (file_prefix, generator_index)
            pil_image.save(file_path)
            logging.info("Generated image saved at {}".format(file_path))
            generated_image_files.append(file_path)

        return generated_image_files

    def decode_attention_maps(self, batch_index: int, file_prefix: str, attention_maps: Tensor,
                              generated_images: Tensor, captions: Tensor, caption_lengths: Tensor,
                              index_to_word: Dict[int, str]) -> List[str]:

        generated_files: List[str] = []
        for attention_model_index in range(len(attention_maps)):
            if len(generated_images) > 1:
                im = generated_images[attention_model_index + 1].detach().cpu()
            else:
                im = generated_images[0].detach().cpu()
            current_attention_map = attention_maps[attention_model_index]
            attention_map_size = current_attention_map.size(2)
            img_set, sentences = \
                decode_attention_maps(batch_images=im[batch_index].unsqueeze(0),
                                      batch_captions=captions[batch_index].unsqueeze(0),
                                      caption_lengths=[caption_lengths[batch_index]],
                                      index_to_word=index_to_word,
                                      attention_maps=[current_attention_map[batch_index]],
                                      attention_map_size=attention_map_size,
                                      top_k_most_attended=5,
                                      font_location=config.generation['font_location'])
            if img_set is not None:
                im = Image.fromarray(img_set)
                file_path: str = self.attention_map_format % (file_prefix, attention_model_index)
                im.save(file_path)
                logging.info("Attention map saved at {}".format(file_path))
                generated_files.append(file_path)

        return generated_files


def extract_noise_vector(caption_index, file_noise_vector: Tensor) -> np.ndarray:
    noise_as_array: np.ndarray = file_noise_vector[caption_index].data.cpu().numpy()
    return noise_as_array
