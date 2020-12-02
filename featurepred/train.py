import logging

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class FeaturePredictorTrainer:

    def __init__(self, model: Module, linear_out_features: int):
        self.model: Module = model

        self.freeze_layers()

        classifier_block_features: int = self.model.fc.in_features
        linear_out_features: int = linear_out_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=classifier_block_features, out_features=linear_out_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=linear_out_features, out_features=2)
        )

    def freeze_layers(self):
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = False

    def train_predictor(self, epochs: int, train_loader: DataLoader, validation_loader: DataLoader,
                        optimiser: Optimizer,
                        loss_function, device):
        for epoch in range(1, epochs + 1):
            avg_training_loss: float = self.do_train(train_loader=train_loader, optimiser=optimiser,
                                                     loss_function=loss_function, device=device)
            avg_validation_loss: float
            validation_accuracy: float
            avg_validation_loss, validation_accuracy = self.evaluate(validation_loader=validation_loader,
                                                                     loss_function=loss_function, device=device)

            logging.info("Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy: {:.2f}".format(epoch,
                                                                                                              avg_training_loss,
                                                                                                              avg_validation_loss,
                                                                                                              validation_accuracy))

    def do_train(self, train_loader: DataLoader, optimiser: Optimizer, loss_function, device) -> float:

        self.model.train()

        avg_training_loss: float = 0.0

        for training_batch in train_loader:
            optimiser.zero_grad()
            images: Tensor
            classes: Tensor

            images, classes_in_batch = training_batch
            images.to(device)
            classes_in_batch.to(device)

            model_output = self.model(images)
            training_loss = loss_function(model_output, classes_in_batch)
            training_loss.backward()
            optimiser.step()

            avg_training_loss += training_loss.data.item() * images.size(0)

        avg_training_loss = avg_training_loss / train_loader.batch_size
        return avg_training_loss

    def evaluate(self, validation_loader: DataLoader, loss_function, device):

        self.model.eval()

        correct_predictions: int = 0
        evaluations: int = 0
        avg_validation_loss: float = 0.0

        for validation_batch in validation_loader:
            images: Tensor
            classes: Tensor

            images, classes_in_batch = validation_batch
            images.to(device)
            classes_in_batch.to(device)

            model_output = self.model(images)
            validation_loss = loss_function(model_output, classes_in_batch)
            avg_validation_loss += validation_loss.data.item() * images.size(0)

            class_by_model = torch.amax(F.softmax(model_output), dim=1)[1]
            model_matches = torch.eq(class_by_model, classes_in_batch).view(-1)

            correct_predictions += torch.sum(model_matches).item()
            evaluations += model_matches.shape[0]

        avg_validation_loss = avg_validation_loss / validation_loader.batch_size
        validation_accuracy = correct_predictions / evaluations

        return avg_validation_loss, validation_accuracy
