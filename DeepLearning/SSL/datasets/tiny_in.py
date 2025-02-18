from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from .transforms import MultiSample, aug_transform
from .base import BaseDataset


def base_transform():
    return T.Compose(
        [T.ToTensor(), T.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))]
    )


class TinyImageNet(BaseDataset):
    def ds_train(self):
        t = MultiSample(
            aug_transform(64, base_transform, self.aug_cfg), n=self.aug_cfg.num_samples
        )
        return ImageFolder(root="data/tiny-imagenet-200/train", transform=t)

    def ds_clf(self):
        t = base_transform()
        return ImageFolder(root="data/tiny-imagenet-200/train", transform=t)

    def ds_test(self):
        t = base_transform()
        return ImageFolder(root="data/tiny-imagenet-200/test", transform=t)
