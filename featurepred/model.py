import logging

import torch
from torch import nn
from torch.nn import Module
from torchvision import models


class FeaturePredictorModelWrapper:

    def __init__(self, model_state_file: str, feature_extraction: bool):
        self.model_state_file = model_state_file

        self.model: Module = models.resnet50(pretrained=True)
        classifier_block_features: int = self.model.fc.in_features
        linear_out_features: int = 128
        self.feature_extraction = feature_extraction

        if self.feature_extraction:
            self.freeze_layers()

        self.model.fc = nn.Sequential(
            nn.Linear(in_features=classifier_block_features, out_features=linear_out_features),
            nn.ReLU(),
            nn.Linear(in_features=linear_out_features, out_features=2)
        )

    def freeze_layers(self):
        logging.info("Freezing base model parameters for feature extraction")
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = False

    def load_model_from_file(self, device: str) -> None:
        self.model.load_state_dict(torch.load(self.model_state_file))
        self.model = self.model.to(device)

        logging.info("Model state loaded from {} to device {}".format(self.model_state_file, device))

    def save_model_state(self):
        torch.save(self.model.state_dict(), self.model_state_file)
        logging.info("Model state saved at {}".format(self.model_state_file))
