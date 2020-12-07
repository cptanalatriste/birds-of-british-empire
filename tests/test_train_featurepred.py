from typing import List, Tuple
from unittest import TestCase

import torch
from torch import Tensor

from featurepred.train import predict_and_evaluate, calculate_epoch_metric


class TestFeaturePredictionTrain(TestCase):

    def test_predict_and_evaluate(self):
        model_output: Tensor = torch.tensor([[.8, .2],
                                             [.7, .3],
                                             [.6, .4]])
        real_labels: Tensor = torch.tensor([0, 0, 1])

        correct_predictions, evaluations = predict_and_evaluate(model_output=model_output, real_labels=real_labels)

        self.assertEqual(correct_predictions, 2)
        self.assertEqual(evaluations, 3)

    def test_calculate_batch_loss(self):
        loss_per_batch: List[Tuple[float, int]] = [(.2, 2), (.4, 2), (.5, 1)]
        batch_loss: float = calculate_epoch_metric(metrics_per_batch=loss_per_batch)

        self.assertEqual(batch_loss, 0.34)
