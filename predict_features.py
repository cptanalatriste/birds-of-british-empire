from torch.nn import Module
from torchvision import models

from featurepred.train import FeaturePredictorTrainer

if __name__ == "__main__":
    resnet50_model: Module = models.resnet50(pretrained=True)
    linear_out_features: int = 500
    trainer: FeaturePredictorTrainer = FeaturePredictorTrainer(model=resnet50_model,
                                                               linear_out_features=linear_out_features)
