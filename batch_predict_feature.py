from typing import List

import pandas as pd
import numpy as np
import logging

from torchvision.transforms import Compose

from featurepred.data import ResNet50DataLoaderBuilder
from featurepred.model import FeaturePredictorModelWrapper
from featurepred.runner import predict_feature
from train_feature_predictor import INPUT_RESIZE

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    metadata_file: str = "artifacts/metadata_file.csv"
    noise_vector_file: str = "artifacts/noise_vectors_array.npy"
    prediction_array_file: str = 'predictions.npy'
    model_state_file: str = 'feature_predictor.pt'
    image_transform: Compose = ResNet50DataLoaderBuilder.get_validation_transformation(input_resize=INPUT_RESIZE)

    metadata_dataframe: pd.DataFrame = pd.read_csv(metadata_file)
    logging.info("Metadata data loaded from: {}".format(metadata_file))

    noise_vectors: np.ndarray = np.load(file=noise_vector_file)
    logging.info("Noise vectors loaded from: {}".format(noise_vector_file))

    model_wrapper: FeaturePredictorModelWrapper = FeaturePredictorModelWrapper(model_state_file=model_state_file,
                                                                               feature_extraction=False)
    model_wrapper.load_model_from_file(device="cpu")
    model_wrapper.model.eval()

    feature_predictions: List[float] = []
    for index, image_row in metadata_dataframe.iterrows():
        noise_vector_csv: np.ndarray = image_row['noise_vector']
        noise_vector_numpy: np.ndarray = noise_vectors[index, :]
        image_file: str = image_row['image_from_generator_2']

        logging.debug("Noise vector (np): {}".format(noise_vector_numpy))
        logging.debug("Noise vector (pd): {}".format(noise_vector_csv))
        logging.info("Image file: {}".format(image_file))

        probability_for_feature: float = predict_feature(model_wrapper=model_wrapper, transform=image_transform,
                                                         image_file=image_file[3:])
        logging.info("Probability from model: {}".format(probability_for_feature))
        feature_predictions.append(probability_for_feature)

    predictions_as_array: np.ndarray = np.array(feature_predictions)
    predictions_as_array = np.expand_dims(a=predictions_as_array, axis=1)
    logging.info("Predictions vector shape: {}".format(predictions_as_array.shape))
    np.save(file=prediction_array_file, arr=predictions_as_array)
    logging.info("Predictions saved at: {}".format(prediction_array_file))
