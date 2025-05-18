from torchvision.datasets import CIFAR10 as C10
import torchvision.transforms as T
from .transforms import MultiSample, aug_transform
from .base import BaseDataset


def base_transform():
    return T.Compose(
        [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )


class CIFAR10(BaseDataset):
    def ds_train(self):
        t = MultiSample(
            aug_transform(32, base_transform, self.aug_cfg), n=self.aug_cfg.num_samples
        )
        return C10(root="./data", train=True, download=True, transform=t)

    def ds_clf(self):
        t = base_transform()
        return C10(root="./data", train=True, download=True, transform=t)

    def ds_test(self):
        t = base_transform()
        return C10(root="./data", train=False, download=True, transform=t)
