from torchvision.datasets import STL10 as S10
import torchvision.transforms as T
from .transforms import MultiSample, aug_transform
from .base import BaseDataset


def base_transform():
    return T.Compose(
        [T.ToTensor(), T.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))]
    )


def test_transform():
    return T.Compose(
        [T.Resize(70, interpolation=3), T.CenterCrop(64), base_transform()]
    )


class STL10(BaseDataset):
    def ds_train(self):
        t = MultiSample(
            aug_transform(64, base_transform, self.aug_cfg), n=self.aug_cfg.num_samples
        )
        return S10(root="./data", split="train+unlabeled", download=True, transform=t)

    def ds_clf(self):
        t = test_transform()
        return S10(root="./data", split="train", download=True, transform=t)

    def ds_test(self):
        t = test_transform()
        return S10(root="./data", split="test", download=True, transform=t)
