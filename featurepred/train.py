import logging
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from featurepred import model


class FeaturePredictorTrainer:

    def __init__(self, model_wrapper: model):
        self.model_wrapper: Module = model_wrapper

    def train_predictor(self, epochs: int, train_loader: DataLoader, validation_loader: DataLoader,
                        optimiser: Optimizer, loss_function, device):
        model: Module = self.model_wrapper.model.to(device)

        train_start: float = time.time()
        best_accuracy: float = 0.0
        self.model_wrapper.save_model_state()

        for epoch in range(1, epochs + 1):
            training_loss: float = do_train(model_to_train=model, train_loader=train_loader, optimiser=optimiser,
                                            loss_function=loss_function, device=device)
            validation_loss: float
            validation_accuracy: float
            validation_loss, validation_accuracy = evaluate(model_to_validate=model,
                                                            validation_loader=validation_loader,
                                                            loss_function=loss_function, device=device)

            logging.info("Epoch: {}, training Loss: {:.2f}, validation Loss: {:.2f}, accuracy: {:.2f}".format(epoch,
                                                                                                              training_loss,
                                                                                                              validation_loss,
                                                                                                              validation_accuracy))

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                self.model_wrapper.save_model_state()

        training_time: float = time.time() - train_start
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(training_time // 60, training_time % 60))
        logging.info('Best accuracy: {}'.format(best_accuracy))


def do_train(model_to_train: Module, train_loader: DataLoader, optimiser: Optimizer, loss_function, device) -> float:
    model_to_train.train()

    total_images: int = len(train_loader.dataset)
    running_loss: float = 0.0
    for training_batch in train_loader:
        optimiser.zero_grad()
        images: Tensor
        classes: Tensor

        images, classes_in_batch = training_batch
        images = images.to(device)
        classes_in_batch = classes_in_batch.to(device)

        model_output: Tensor = model_to_train(images)
        training_loss: Tensor = loss_function(model_output, classes_in_batch)

        optimiser.zero_grad()
        training_loss.backward()
        optimiser.step()

        images_in_batch: int = images.size(0)
        running_loss += training_loss.data.item() * images_in_batch

    total_loss: float = running_loss / total_images
    return total_loss


def evaluate(model_to_validate: Module, validation_loader: DataLoader, loss_function, device) -> Tuple[float, float]:
    model_to_validate.eval()

    total_images: int = len(validation_loader.dataset)
    running_loss: float = 0.0
    running_matches: float = 0.0

    for validation_batch in validation_loader:
        images: Tensor
        classes: Tensor

        images, classes_in_batch = validation_batch
        images = images.to(device)
        classes_in_batch = classes_in_batch.to(device)

        model_output: Tensor = model_to_validate(images)
        validation_loss: Tensor = loss_function(model_output, classes_in_batch)
        batch_matches, _ = predict_and_evaluate(model_output=model_output, real_labels=classes_in_batch)

        images_in_batch: int = images.size(0)
        running_loss += validation_loss.data.item() * images_in_batch
        running_matches += batch_matches

    total_loss: float = running_loss / total_images
    total_accuracy: float = running_matches / total_images

    return total_loss, total_accuracy


def predict_and_evaluate(model_output: Tensor, real_labels: Tensor) -> Tuple[float, float]:
    class_by_model: Tensor = output_to_target_class(model_output=model_output)
    model_matches = torch.eq(class_by_model, real_labels).view(-1)

    correct_predictions = torch.sum(model_matches).item()
    evaluations = model_matches.shape[0]

    logging.debug("correct_predictions {} evaluations {} ".format(correct_predictions, evaluations))
    return correct_predictions, evaluations


def output_to_target_class(model_output: Tensor) -> Tensor:
    class_by_model = torch.max(F.softmax(model_output), dim=1)[1]
    return class_by_model


def output_to_class_probabilities(model_output: Tensor) -> Tensor:
    class_probabilities = F.softmax(model_output)
    return class_probabilities
