import logging

import torch
from torch import optim, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from attnganw.randomutils import set_random_seed
from featurepred.model import FeaturePredictorModelWrapper
from featurepred.data import ResNet50DataLoaderBuilder, RESNET50_MEANS, RESNET50_STD_DEVS
from featurepred.train import FeaturePredictorTrainer, output_to_target_class
from utils.image import plot_images_with_labels

INPUT_RESIZE: int = 224


def start_training(predictor_trainer: FeaturePredictorTrainer, train_data_loader: DataLoader,
                   validation_data_loader: DataLoader,
                   num_epochs: int, optimiser_learning_rate: float):
    model: Module = predictor_trainer.model_wrapper.model

    if predictor_trainer.model_wrapper.feature_extraction:
        logging.info("Feature extraction: Only optimizing last layer's parameters")
        optimiser = optim.Adam(params=model.fc.parameters(), lr=optimiser_learning_rate)
    else:
        optimiser = optim.Adam(params=model.parameters(), lr=optimiser_learning_rate)

    loss_function = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        logging.info("CUDA supported. Running on GPU")
        device = torch.device("cuda")
    predictor_trainer.train_predictor(epochs=num_epochs, train_loader=train_data_loader,
                                      validation_loader=validation_data_loader,
                                      optimiser=optimiser, loss_function=loss_function, device=device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)

    epochs: int = 3
    train_image_folder: str = 'data/feature_data/train'
    validation_image_folder: str = 'data/feature_data/val'
    batch_size: int = 32
    learning_rate: float = 0.001
    input_resize: int = INPUT_RESIZE
    model_state_file: str = 'feature_predictor.pt'
    random_seed: int = 100
    data_loader_workers: int = 4

    set_random_seed(random_seed=random_seed)
    model_wrapper: FeaturePredictorModelWrapper = FeaturePredictorModelWrapper(model_state_file=model_state_file,
                                                                               feature_extraction=True)

    trainer: FeaturePredictorTrainer = FeaturePredictorTrainer(model_wrapper=model_wrapper)

    train_dataloader_builder: ResNet50DataLoaderBuilder = ResNet50DataLoaderBuilder(image_folder=train_image_folder,
                                                                                    batch_size=batch_size,
                                                                                    input_resize=input_resize,
                                                                                    is_training=True,
                                                                                    data_loader_workers=data_loader_workers)
    train_loader: DataLoader = train_dataloader_builder.build()
    train_images, train_classes = next(iter(train_loader))
    plot_images_with_labels(images=train_images[:10], classes=train_classes[:10],
                            class_names=train_dataloader_builder.class_names,
                            means=RESNET50_MEANS, standard_devs=RESNET50_STD_DEVS,
                            file_name='training_sample.png')

    valid_data_loader_builder: ResNet50DataLoaderBuilder = ResNet50DataLoaderBuilder(
        image_folder=validation_image_folder,
        batch_size=batch_size,
        input_resize=input_resize,
        is_training=False,
        data_loader_workers=data_loader_workers)
    validation_loader: DataLoader = valid_data_loader_builder.build()
    start_training(predictor_trainer=trainer, train_data_loader=train_loader, validation_data_loader=validation_loader,
                   num_epochs=epochs,
                   optimiser_learning_rate=learning_rate)

    model_wrapper.load_model_from_file(device="cpu")
    model_wrapper.model.eval()
    valid_images, valid_classes = next(iter(validation_loader))
    valid_images = valid_images[:10]
    predicted_classes: Tensor = output_to_target_class(model_output=model_wrapper.model(valid_images[:10]))

    plot_images_with_labels(images=valid_images, classes=predicted_classes,
                            class_names=valid_data_loader_builder.class_names,
                            means=RESNET50_MEANS, standard_devs=RESNET50_STD_DEVS,
                            file_name='prediction_sample.png')
