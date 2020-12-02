from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, Compose


class ResNet50DataLoaderBuilder:

    def __init__(self, image_folder: str, batch_size: int):
        self.image_transformations: Compose = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.image_folder: ImageFolder = ImageFolder(root=image_folder, transform=self.image_transformations,
                                                     is_valid_file=can_open_image_file)

        self.batch_size: int = batch_size

    def build(self) -> DataLoader:
        return DataLoader(dataset=self.image_folder, batch_size=self.batch_size, shuffle=True)


def can_open_image_file(image_path: str):
    try:
        Image.open(image_path)
        return True
    except:
        return False
