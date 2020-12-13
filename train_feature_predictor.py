import logging

import torch
from torch import optim, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from featurepred.FeaturePredictorModelWrapper import FeaturePredictorModelWrapper
from featurepred.data import ResNet50DataLoaderBuilder
from featurepred.train import FeaturePredictorTrainer, output_to_predictions
from utils.image import plot_images_with_labels


def start_training(predictor_trainer: FeaturePredictorTrainer, train_data_loader: DataLoader, validation_data_loader: DataLoader,
                   num_epochs: int, optimiser_learning_rate: float):
    model: Module = predictor_trainer.model_wrapper.model
    optimiser = optim.Adam(params=model.fc.parameters(), lr=optimiser_learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        logging.info("CUDA supported. Running on GPU")
        device = torch.device("cuda")
    predictor_trainer.train_predictor(epochs=num_epochs, train_loader=train_data_loader, validation_loader=validation_data_loader,
                                      optimiser=optimiser, loss_function=loss_function, device=device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)

    epochs: int = 20
    train_image_folder: str = 'data/feature_data/train'
    validation_image_folder: str = 'data/feature_data/val'
    batch_size: int = 32
    learning_rate: float = 0.001
    input_resize: int = 224
    model_state_file: str = 'feature_predictor.pt'

    model_wrapper: FeaturePredictorModelWrapper = FeaturePredictorModelWrapper(model_state_file=model_state_file,
                                                                               is_training=True)

    trainer: FeaturePredictorTrainer = FeaturePredictorTrainer(model_wrapper=model_wrapper)

    train_dataloader_builder: ResNet50DataLoaderBuilder = ResNet50DataLoaderBuilder(image_folder=train_image_folder,
                                                                                    batch_size=batch_size,
                                                                                    input_resize=input_resize,
                                                                                    is_training=True)
    train_loader: DataLoader = train_dataloader_builder.build()
    train_images, train_classes = next(iter(train_loader))
    plot_images_with_labels(images=train_images[:10], classes=train_classes[:10],
                            class_names=train_dataloader_builder.class_names,
                            means=train_dataloader_builder.means, standard_devs=train_dataloader_builder.std_devs,
                            file_name='training_sample.png')

    valid_data_loader_builder: ResNet50DataLoaderBuilder = ResNet50DataLoaderBuilder(
        image_folder=validation_image_folder,
        batch_size=batch_size,
        input_resize=input_resize,
        is_training=False)
    validation_loader: DataLoader = valid_data_loader_builder.build()
    start_training(predictor_trainer=trainer, train_data_loader=train_loader, validation_data_loader=validation_loader, num_epochs=epochs,
                   optimiser_learning_rate=learning_rate)

    model_wrapper.load_model_from_file(device="cpu")
    model_wrapper.model.eval()
    valid_images, valid_classes = next(iter(validation_loader))
    valid_images = valid_images[:10]
    predicted_classes: Tensor = output_to_predictions(model_output=model_wrapper.model(valid_images[:10]))

    plot_images_with_labels(images=valid_images, classes=predicted_classes,
                            class_names=valid_data_loader_builder.class_names,
                            means=valid_data_loader_builder.means, standard_devs=valid_data_loader_builder.std_devs,
                            file_name='prediction_sample.png')
