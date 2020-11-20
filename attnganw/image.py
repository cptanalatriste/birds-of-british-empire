from torch import Tensor
import numpy as np
from PIL import Image


class ImageDecoder:

    def __init__(self):
        self.generated_format = '%s_generator_%d.png'

    def decode_generated_images(self, batch_index: int, file_prefix: str, generated_images: Tensor):
        for generator_index in range(len(generated_images)):
            image_as_array: np.ndarray = generated_images[generator_index][batch_index].data.cpu().numpy()
            image_as_array = (image_as_array + 1.0) * 127.5
            image_as_array = image_as_array.astype(np.uint8)
            image_as_array = np.transpose(image_as_array, (1, 2, 0))
            pil_image: Image = Image.fromarray(image_as_array)
            file_path = self.generated_format % (file_prefix, generator_index)
            pil_image.save(file_path)
            print("Generated image saved at ", file_path)
