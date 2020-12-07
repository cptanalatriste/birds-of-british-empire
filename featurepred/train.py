import logging
import time
from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class FeaturePredictorTrainer:

    def __init__(self, model: Module, linear_out_features: int, model_state_file: str):
        self.model: Module = model
        self.model_state_file = model_state_file

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

    def save_model_state(self):
        torch.save(self.model.state_dict(), self.model_state_file)
        logging.info("Model state saved at {}".format(self.model_state_file))

    def train_predictor(self, epochs: int, train_loader: DataLoader, validation_loader: DataLoader,
                        optimiser: Optimizer, loss_function, device):
        train_start: float = time.time()
        best_accuracy: float = 0.0
        self.save_model_state()

        for epoch in range(1, epochs + 1):
            training_loss: float = do_train(model=self.model, train_loader=train_loader, optimiser=optimiser,
                                            loss_function=loss_function, device=device)
            validation_loss: float
            validation_accuracy: float
            validation_loss, validation_accuracy = evaluate(model=self.model, validation_loader=validation_loader,
                                                            loss_function=loss_function, device=device)

            logging.info("Epoch: {}, training Loss: {:.2f}, validation Loss: {:.2f}, accuracy: {:.2f}".format(epoch,
                                                                                                              training_loss,
                                                                                                              validation_loss,
                                                                                                              validation_accuracy))

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                self.save_model_state()

        training_time: float = time.time() - train_start
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(training_time // 60, training_time % 60))
        logging.info('Best accuracy: {}'.format(best_accuracy))


def do_train(model, train_loader: DataLoader, optimiser: Optimizer, loss_function, device) -> float:
    model.train()

    loss_per_batch: List[Tuple[float, int]] = []

    for training_batch in train_loader:
        optimiser.zero_grad()
        images: Tensor
        classes: Tensor

        images, classes_in_batch = training_batch
        images.to(device)
        classes_in_batch.to(device)

        model_output: Tensor = model(images)
        training_loss: Tensor = loss_function(model_output, classes_in_batch)

        training_loss.backward()
        optimiser.step()

        logging.debug(
            "training_loss.data.item() {} images.size(0) {}".format(training_loss.data.item(), images.size(0)))
        loss_per_batch.append((training_loss.data.item(), images.size(0)))

    total_loss: float = calculate_epoch_metric(metrics_per_batch=loss_per_batch)
    return total_loss


def evaluate(model, validation_loader: DataLoader, loss_function, device) -> Tuple[float, float]:
    model.eval()

    loss_per_batch: List[Tuple[float, int]] = []
    accuracy_per_batch: List[Tuple[float, int]] = []

    for validation_batch in validation_loader:
        images: Tensor
        classes: Tensor

        images, classes_in_batch = validation_batch
        images.to(device)
        classes_in_batch.to(device)

        model_output: Tensor = model(images)
        validation_loss: Tensor = loss_function(model_output, classes_in_batch)

        loss_per_batch.append((validation_loss.data.item(), images.size(0)))
        batch_matches, batch_evaluation = predict_and_evaluate(model_output=model_output, real_labels=classes_in_batch)
        accuracy_per_batch.append((batch_matches / batch_evaluation, 1))

    total_loss: float = calculate_epoch_metric(metrics_per_batch=loss_per_batch)
    total_accuracy: float = calculate_epoch_metric(metrics_per_batch=accuracy_per_batch)

    return total_loss, total_accuracy


def predict_and_evaluate(model_output: Tensor, real_labels: Tensor) -> Tuple[float, float]:
    class_by_model = torch.max(F.softmax(model_output), dim=1)[1]
    model_matches = torch.eq(class_by_model, real_labels).view(-1)

    correct_predictions = torch.sum(model_matches).item()
    evaluations = model_matches.shape[0]

    logging.debug("correct_predictions {} evaluations {} ".format(correct_predictions, evaluations))
    return correct_predictions, evaluations


def calculate_epoch_metric(metrics_per_batch: List[Tuple[float, int]]) -> float:
    total_sum: float = sum([batch_metric * images for batch_metric, images in metrics_per_batch])
    total_images: int = sum([images for _, images in metrics_per_batch])

    return total_sum / total_images
