import os
import glob
import shutil

import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST, CIFAR10, UCF101
from spikingjelly.datasets.n_mnist import NMNIST
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import VisionDataset
from torchvision.io import read_video


SINGLE_CHANNEL_DATASETS = ["MNIST", "FashionMNIST", "KMNIST"]

DATASET_CHANNEL_MAP = {
    "MNIST": 1,
    "CIFAR10": 3,
    "EventMNIST": 2,
    "UCF101": 3,
    "UCF11": 3,
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


class VideoTransform:
    """
    Applies transforms to a video tensor of shape (T, C, H, W).
    """

    def __init__(self, size=(128, 128), normalize=True, mean=None, std=None):
        self.normalize = normalize
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.size = size

    def __call__(self, video):

        video = video.float() / 255.0

        video = torch.nn.functional.interpolate(
            video, size=self.size, mode="bilinear", align_corners=False
        )

        if self.normalize:

            mean = torch.tensor(self.mean, device=video.device).view(
                1, 3, 1, 1
            )
            std = torch.tensor(self.std, device=video.device).view(1, 3, 1, 1)
            video = (video - mean) / std

        return video


class UCF101ClipDataset(UCF101):

    def __init__(
        self,
        root=None,  # unused
        annotation_path=None,
        repeat: int = 10,
        step_between_clips: int = 10,
        train: bool = True,
        normalize: bool = True,
        resize_shape: tuple = (128, 128),
        download: bool = False,  # unused, just for compatibility
        **kwargs,
    ):
        if root is None:
            raise ValueError(
                "Root directory not specified for UCF101 dataset."
            )

        if annotation_path is None:
            # Assume annotations are in "annotations" subdirectory of root's parent or similar,
            # for now, let's just default to root/annotations if not provided,
            # OR raise error. Making it required is safer if structure is not guaranteed.
            # But based on typical use, let's try to be helpful.
            annotation_path = os.path.join(root, "annotations")
            if not os.path.exists(annotation_path):
                print(
                    f"Warning: Annotation path not found at {annotation_path}. Please specify annotation_path explicitly if it is different."
                )

        if not os.path.exists(root) or not os.path.exists(annotation_path):
            print(
                f"Warning: UCF101 paths not found.\nRoot: {root}\nAnno: {annotation_path}"
            )

        self.transform_pipeline = VideoTransform(
            size=resize_shape, normalize=normalize
        )

        if not normalize:
            print("Z-score standardization is disabled for UCF101.")

        super().__init__(
            root=root,
            annotation_path=annotation_path,
            frames_per_clip=repeat,
            step_between_clips=step_between_clips,
            train=train,
            transform=None,
            output_format="TCHW",
            **kwargs,
        )

    def __getitem__(self, index):
        video, audio, label = super().__getitem__(index)
        video = self.transform_pipeline(video)
        return video, label


class UCF11ClipDataset(VisionDataset):
    """
    UCF11 (YouTube Action) Dataset.
    Source Data Structure expected:
        root/basketball/v_shooting_01/video.mpg
        root/biking/v_biking_01/video.mpg
    """

    CLASSES = [
        "basketball",
        "biking",
        "diving",
        "golf_swing",
        "horse_riding",
        "soccer_juggling",
        "swing",
        "tennis_swing",
        "trampoline_jumping",
        "volleyball_spiking",
        "walking",
    ]

    def __init__(
        self,
        root=None,
        train: bool = True,
        repeat: int = 16,  # Maps to frames_per_clip
        normalize: bool = True,
        resize_shape: tuple = (128, 128),
        **kwargs,
    ):
        # root = self.DEFAULT_ROOT
        if root is None:
            raise ValueError("Root directory not specified for UCF11 dataset.")
        super().__init__(root)

        self.train = train
        self.frames_per_clip = repeat
        self.transform_pipeline = VideoTransform(
            size=resize_shape, normalize=normalize
        )
        self.samples = []
        self.class_to_idx = {cls: i for i, cls in enumerate(self.CLASSES)}

        if not os.path.exists(root):
            print(f"Warning: UCF11 root path not found: {root}")

        if not normalize:
            print("Z-score standardization is disabled for UCF11.")

        self._make_dataset()

    def _make_dataset(self):
        """
        Scans the directory.
        Splitting Protocol: UCF11 has ~25 groups per class.
        We use groups 01-19 for TRAIN and 20-25 for TEST (approx 80/20 split).
        """
        if not os.path.exists(self.root):
            return

        for class_name in self.CLASSES:
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Recursive search for video files (mpg or avi)
            # UCF11 structure: class_name/group_name/video_file
            extensions = ["*.mpg", "*.avi"]
            video_files = []
            for ext in extensions:
                video_files.extend(
                    glob.glob(
                        os.path.join(class_dir, "**", ext), recursive=True
                    )
                )

            for file_path in video_files:
                # Get parent folder name to determine the Group (e.g., "v_shooting_01")
                parent_dir = os.path.basename(os.path.dirname(file_path))

                # Heuristic: Parse group number "01" from "v_shooting_01"
                try:
                    # Usually the last part of the string after '_'
                    group_id = int(parent_dir.split("_")[-1])
                except ValueError:
                    # Fallback if naming is weird
                    group_id = hash(parent_dir) % 25

                # Groups 1-19 -> Train, Groups 20+ -> Test
                is_train_group = group_id < 20

                if self.train == is_train_group:
                    self.samples.append(
                        (file_path, self.class_to_idx[class_name])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]

        # 1. Load Video
        # read_video returns (T, H, W, C) in [0, 255] uint8
        # We assume pts_unit='sec' is robust for mpg files
        vframes, _, _ = read_video(path, pts_unit="sec")

        # 2. Temporal Slicing (Get exactly N frames)
        total_frames = vframes.shape[0]

        if total_frames > self.frames_per_clip:
            if self.train:
                # Random crop in time
                start = torch.randint(
                    0, total_frames - self.frames_per_clip, (1,)
                ).item()
            else:
                # Center crop in time
                start = (total_frames - self.frames_per_clip) // 2
            vframes = vframes[start : start + self.frames_per_clip]
        else:
            # Loop padding if video is too short
            missing = self.frames_per_clip - total_frames
            if missing > 0:
                vframes = torch.cat([vframes, vframes[:missing]], dim=0)
                # Check again in case it was extremely short
                if vframes.shape[0] < self.frames_per_clip:
                    # Resize temporal dimension (last resort)
                    vframes = vframes.permute(3, 0, 1, 2).unsqueeze(
                        0
                    )  # (1, C, T, H, W)
                    vframes = torch.nn.functional.interpolate(
                        vframes,
                        size=(
                            self.frames_per_clip,
                            vframes.shape[3],
                            vframes.shape[4],
                        ),
                    )
                    vframes = vframes.squeeze(0).permute(
                        1, 2, 3, 0
                    )  # Back to (T, H, W, C)

        # 3. Permute to (T, C, H, W) for VideoTransform
        vframes = vframes.permute(0, 3, 1, 2)

        # 4. Apply Transform (Float conversion, Resize, Normalize)
        vframes = self.transform_pipeline(vframes)

        return vframes, target


class EventMNISTDataset(NMNIST):
    DATA_ROOT_DIR_FOLDER_NAME = "nmnist"

    def __init__(
        self, root: str, repeat: int = 1, normalize: bool = True, **kwargs
    ):
        if root is None:
            raise ValueError(
                "Root directory not specified for EventMNIST dataset."
            )

        # If user provides a root, we check if it is the dataset folder itself or contains it
        # NMNIST class usually expects root to contain the "n_mnist" folder structure or similar.
        # But here we want to be flexible.

        # We will assume `root` passed is where the data is supposed to be.
        # Check if root is a directory
        self.data_root = root

        # Check if compressed file exists if folder doesn't
        if not os.path.isdir(self.data_root):
            # Check if it might be a tar.gz file
            if self.data_root.endswith("tar.gz") and os.path.isfile(
                self.data_root
            ):
                # If it is a tar file, we might want to extract it?
                # ideally we extract to the same dir.
                extract_path = os.path.dirname(self.data_root)
                print(f"Extracting {self.data_root} to {extract_path}...")
                shutil.unpack_archive(self.data_root, extract_path)
                # Update data_root to the extracted folder.
                # Assuming extracting "nmnist.tar.gz" gives "nmnist" folder?
                # This logic is a bit specific to how the tar was made.
                # Let's try to find the folder.
                # For now, let's keep it simple: WE EXPECT EXTRACTED FOLDER unless explicit.
                pass

        if not os.path.exists(self.data_root):
            print(
                f"Warning: EventMNIST data root {self.data_root} does not exist."
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
        elif name == "UCF101":
            return UCF101ClipDataset(*args, **kwargs)
        elif name == "UCF11":
            # UCF11 uses specific kwargs, we pass them through
            return UCF11ClipDataset(*args, **kwargs)
        else:
            raise ValueError(f"Dataset {name} not recognized.")

    @staticmethod
    def num_classes(name):
        if name in ["MNIST", "CIFAR10", "EventMNIST"]:
            return 10
        elif name == "UCF101":
            return 101
        elif name == "UCF11":
            return 11
        else:
            raise ValueError(f"Dataset {name} not recognized.")

    @staticmethod
    def available_datasets():
        return ["MNIST", "CIFAR10", "EventMNIST", "UCF101", "UCF11"]

    @staticmethod
    def is_native_dvs(name):
        return name == "EventMNIST"


if __name__ == "__main__":
    dataset = DatasetFactory.create_dataset(
        "UCF11", train=True, repeat=10, normalize=True, step_between_clips=10
    )

    breakpoint()
