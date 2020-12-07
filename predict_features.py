import logging

import torch
from torch import optim
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import models

from featurepred.data import ResNet50DataLoaderBuilder
from featurepred.train import FeaturePredictorTrainer
from utils.image import plot_grid

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)

    resnet50_model: Module = models.resnet50(pretrained=True)
    linear_out_features: int = 500
    trainer: FeaturePredictorTrainer = FeaturePredictorTrainer(model=resnet50_model,
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
    sample_images, sample_classes = next(iter(train_loader))
    plot_grid(images=sample_images[:5], classes=sample_classes[:5], class_names=train_dataloader_builder.class_names,
              means=train_dataloader_builder.means, standard_devs=train_dataloader_builder.std_devs,
              file_name='training_sample.png')

    validation_loader: DataLoader = ResNet50DataLoaderBuilder(image_folder=validation_image_folder,
                                                              batch_size=batch_size).build()
    optimiser = optim.Adam(params=resnet50_model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        logging.info("CUDA supported. Running on GPU")
        device = torch.device("cuda")

    trainer.train_predictor(epochs=epochs, train_loader=train_loader, validation_loader=validation_loader,
                            optimiser=optimiser, loss_function=loss_function, device=device)
