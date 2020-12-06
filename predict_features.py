import logging

import torch
from torch import optim
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import models

from featurepred.data import ResNet50DataLoaderBuilder
from featurepred.train import FeaturePredictorTrainer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    resnet50_model: Module = models.resnet50(pretrained=True)
    linear_out_features: int = 500
    trainer: FeaturePredictorTrainer = FeaturePredictorTrainer(model=resnet50_model,
                                                               linear_out_features=linear_out_features)

    epochs: int = 5
    train_image_folder: str = 'data/feature_data/train'
    validation_image_folder: str = 'data/feature_data/val'
    batch_size: int = 64
    learning_rate: float = 0.001
    train_loader: DataLoader = ResNet50DataLoaderBuilder(image_folder=train_image_folder, batch_size=batch_size).build()
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
