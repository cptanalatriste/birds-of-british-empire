import logging

import torch
from torch import optim, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import models

from featurepred.data import ResNet50DataLoaderBuilder
from featurepred.train import FeaturePredictorTrainer, output_to_predictions
from utils.image import plot_images_with_labels


def start_training(trainer: FeaturePredictorTrainer, train_loader: DataLoader, validation_loader: DataLoader,
                   epochs: int):
    optimiser = optim.Adam(params=resnet50_model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        logging.info("CUDA supported. Running on GPU")
        device = torch.device("cuda")
    trainer.train_predictor(epochs=epochs, train_loader=train_loader, validation_loader=validation_loader,
                            optimiser=optimiser, loss_function=loss_function, device=device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)

    resnet50_model: Module = models.resnet50(pretrained=True)
    linear_out_features: int = 500
    trainer: FeaturePredictorTrainer = FeaturePredictorTrainer(base_model=resnet50_model,
                                                               linear_out_features=linear_out_features,
                                                               model_state_file='feature_predictor.pt')

    epochs: int = 5
    train_image_folder: str = 'data/feature_data/train'
    validation_image_folder: str = 'data/feature_data/val'
    batch_size: int = 64
    learning_rate: float = 0.001
    train_dataloader_builder: ResNet50DataLoaderBuilder = ResNet50DataLoaderBuilder(image_folder=train_image_folder,
                                                                                    batch_size=batch_size)
    train_loader: DataLoader = train_dataloader_builder.build()
    train_images, train_classes = next(iter(train_loader))
    plot_images_with_labels(images=train_images[:6], classes=train_classes[:6],
                            class_names=train_dataloader_builder.class_names,
                            means=train_dataloader_builder.means, standard_devs=train_dataloader_builder.std_devs,
                            file_name='training_sample.png')

    valid_data_loader_builder: ResNet50DataLoaderBuilder = ResNet50DataLoaderBuilder(
        image_folder=validation_image_folder,
        batch_size=batch_size)
    validation_loader: DataLoader = valid_data_loader_builder.build()
    start_training(trainer=trainer, train_loader=train_loader, validation_loader=validation_loader, epochs=epochs)

    trainer.load_model_from_file()
    trainer.model.eval()
    valid_images, valid_classes = next(iter(validation_loader))
    valid_images = valid_images[:6]
    predicted_classes: Tensor = output_to_predictions(model_output=trainer.model(valid_images[:6]))

    plot_images_with_labels(images=valid_images, classes=predicted_classes,
                            class_names=valid_data_loader_builder.class_names,
                            means=valid_data_loader_builder.means, standard_devs=valid_data_loader_builder.std_devs,
                            file_name='prediction_sample.png')
