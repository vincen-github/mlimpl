from torchvision.datasets import CIFAR100 as C100
import torchvision.transforms as T
from .transforms import MultiSample, aug_transform
from .base import BaseDataset


def base_transform():
    return T.Compose(
        [T.ToTensor(), T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
    )


class CIFAR100(BaseDataset):
    def ds_train(self):
        t = MultiSample(
            aug_transform(32, base_transform, self.aug_cfg), n=self.aug_cfg.num_samples
        )
        return C100(root="./data", train=True, download=True, transform=t,)

    def ds_clf(self):
        t = base_transform()
        return C100(root="./data", train=True, download=True, transform=t)

    def ds_test(self):
        t = base_transform()
        return C100(root="./data", train=False, download=True, transform=t)
