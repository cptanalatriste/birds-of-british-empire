import logging

from PIL import Image
from torch import Tensor
from torchvision.transforms import Compose

from featurepred.model import FeaturePredictorModelWrapper
from featurepred.train import output_to_class_probabilities

POSITIVE_CLASS_INDEX = 1


def predict_feature(model_wrapper: FeaturePredictorModelWrapper, transform: Compose,
                    image_file: str) -> float:

    pil_image = Image.open(image_file)
    transformed_image: Tensor = transform(pil_image)

    input_as_batch: Tensor = transformed_image.unsqueeze(0)
    class_probabilities: Tensor = output_to_class_probabilities(model_output=model_wrapper.model(input_as_batch))
    class_probabilities: Tensor = class_probabilities.squeeze(0)

    logging.debug("class_probabilities: {} shape: {}".format(class_probabilities, class_probabilities.shape))
    return class_probabilities[POSITIVE_CLASS_INDEX]
