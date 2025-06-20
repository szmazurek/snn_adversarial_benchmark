from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10


class MNISTRepeated(MNIST):
    def __init__(self, *args, repeat=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeat = repeat

        self.transform_pipeline = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        img_tensor = self.transform_pipeline(img)

        img_tensor = img_tensor.squeeze(0)

        img_tensor = img_tensor.repeat(self.repeat, 1, 1).unsqueeze(1)

        return img_tensor, target


class CIFAR10Repeated(CIFAR10):

    def __init__(self, *args, repeat=1, **kwargs):

        super().__init__(*args, transform=None, **kwargs)

        self.repeat = repeat
        self.transform_pipeline = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        img_tensor = self.transform_pipeline(img).unsqueeze(0)
        repeated_img_tensor = img_tensor.repeat(self.repeat, 1, 1, 1)

        return repeated_img_tensor, target


class DatasetFactory:
    @staticmethod
    def create_dataset(name, *args, **kwargs):
        if name == "MNIST":
            return MNISTRepeated(*args, **kwargs)
        elif name == "CIFAR10":
            return CIFAR10Repeated(*args, **kwargs)
        else:
            raise ValueError(f"Dataset {name} not recognized.")
