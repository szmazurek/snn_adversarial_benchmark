import torch
import numpy as np
from torchvision.datasets import MNIST


class MNISTRepeated(MNIST):
    def __init__(self, *args, repeat=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeat = repeat

    @staticmethod
    def _min_max_scale(img):
        return (img - img.min()) / (img.max() - img.min())

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32)
        img_tensor = self._min_max_scale(img_tensor)
        img_tensor = img_tensor.repeat(self.repeat, 1, 1).unsqueeze(1)
        return img_tensor, target
