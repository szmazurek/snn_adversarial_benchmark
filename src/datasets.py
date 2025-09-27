import os
import shutil
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST, CIFAR10
from spikingjelly.datasets.n_mnist import NMNIST

SINGLE_CHANNEL_DATASETS = ["MNIST", "FashionMNIST", "KMNIST"]

DATASET_CHANNEL_MAP = {
    "MNIST": 1,
    "CIFAR10": 3,
    "EventMNIST": 2,
}


class EventMNISTToTensorTransform(torch.nn.Module):
    def forward(self, x):
        # x is a np array of shape (T, C, H, W)
        return torch.from_numpy(x)


class MNISTRepeated(MNIST):
    def __init__(
        self, *args, repeat: int = 1, normalize: bool = True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.repeat = repeat
        self.normalize = normalize
        self.transform_pipeline = (
            Compose(
                [
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,)),
                ]
            )
            if normalize
            else ToTensor()
        )
        if not self.normalize:
            print(
                "Z-score standarization is disabled, will use unchanged pixel values."
            )

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        img_tensor = self.transform_pipeline(img).unsqueeze(0)
        img_tensor = img_tensor.repeat(self.repeat, 1, 1, 1)

        return img_tensor, target


class CIFAR10Repeated(CIFAR10):

    def __init__(
        self, *args, repeat: int = 1, normalize: bool = True, **kwargs
    ):

        super().__init__(*args, transform=None, **kwargs)

        self.repeat = repeat
        self.normalize = normalize
        self.transform_pipeline = (
            Compose(
                [
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                    ),
                ]
            )
            if normalize
            else ToTensor()
        )

        if not self.normalize:
            print(
                "Z-score standarization is disabled, will use unchanged pixel values."
            )

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        img_tensor = self.transform_pipeline(img).unsqueeze(0)

        repeated_img_tensor = img_tensor.repeat(self.repeat, 1, 1, 1)

        return repeated_img_tensor, target


class EventMNISTDataset(NMNIST):
    PREEXTRACTED_DATA_PATH = "/net/tscratch/people/plgmazurekagh/snn_adversarial_benchmark/data/nmnist_compressed_10_frames.tar.gz"
    DATA_ROOT_DIR_FOLDER_NAME = "nmnist"

    def __init__(
        self, *args, repeat: int = 1, normalize: bool = True, **kwargs
    ):
        self.memfs_path = os.environ.get("MEMFS")
        assert os.path.exists(self.PREEXTRACTED_DATA_PATH), (
            f"Pre-extracted data path {self.PREEXTRACTED_DATA_PATH} does not exist. "
            "You either do not have the file or you are not on the cluster."
            "This path is HARDCODED in the class of this dataset, please change it if needed."
        )
        assert (
            os.environ.get("MEMFS") is not None
        ), "MEMFS environment variable must be set, or you kill the cluster FS with many files"
        # untar the dataset into memfs
        self.data_root = os.path.join(
            self.memfs_path, self.DATA_ROOT_DIR_FOLDER_NAME
        )
        if os.path.exists(self.data_root):
            print(
                f"Data root {self.data_root} already exists, skipping extraction."
            )
        else:
            print(
                f"Extracting dataset to MEMFS path {self.memfs_path}, this may take a while..."
            )
            shutil.unpack_archive(self.PREEXTRACTED_DATA_PATH, self.memfs_path)

        assert os.path.exists(self.data_root), (
            f"Extracted data path {self.data_root} does not exist. "
            "Something went wrong with the extraction."
            f"The dataset should be extracted into MEMFS path as '{self.DATA_ROOT_DIR_FOLDER_NAME}' folder."
            "The folder name is HARDCODED in the class of this dataset, please change it if needed."
        )
        transform_pipeline = (
            Compose(
                [
                    EventMNISTToTensorTransform(),
                    Normalize(
                        (
                            0.180647,
                            0.180247,
                        ),
                        (0.753486, 0.696370),
                    ),
                ]
            )
            if normalize
            else EventMNISTToTensorTransform()
        )
        # for kwargs skip the root and transform, we set them ourselves
        kwargs.pop("root", None)
        kwargs.pop("transform", None)
        kwargs.pop("download", None)  # we do not want to download

        super().__init__(
            *args,
            root=self.data_root,
            transform=transform_pipeline,
            frames_number=repeat,
            split_by="time",
            data_type="frame",
            **kwargs,
        )


class DatasetFactory:
    @staticmethod
    def create_dataset(name, *args, **kwargs):
        if name == "MNIST":
            return MNISTRepeated(*args, **kwargs)
        elif name == "CIFAR10":
            return CIFAR10Repeated(*args, **kwargs)
        elif name == "EventMNIST":
            return EventMNISTDataset(*args, **kwargs)
        else:
            raise ValueError(f"Dataset {name} not recognized.")

    @staticmethod
    def available_datasets():
        return ["MNIST", "CIFAR10", "EventMNIST"]

    @staticmethod
    def is_native_dvs(name):
        return name == "EventMNIST"
