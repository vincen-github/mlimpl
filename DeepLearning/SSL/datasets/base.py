from abc import ABCMeta, abstractmethod
from functools import lru_cache
from torch.utils.data import DataLoader


class BaseDataset(metaclass=ABCMeta):
    """
        base class for datasets, it includes 3 types:
            - for self-supervised training,
            - for classifier training for evaluation,
            - for testing
    """

    def __init__(
        self, bs_train, aug_cfg, num_workers, bs_clf=1000, bs_test=1000,
    ):
        self.aug_cfg = aug_cfg
        self.bs_train, self.bs_clf, self.bs_test = bs_train, bs_clf, bs_test
        self.num_workers = num_workers

    @abstractmethod
    def ds_train(self):
        raise NotImplementedError

    @abstractmethod
    def ds_clf(self):
        raise NotImplementedError

    @abstractmethod
    def ds_test(self):
        raise NotImplementedError

    @property
    @lru_cache()
    def train(self):
        return DataLoader(
            dataset=self.ds_train(),
            batch_size=self.bs_train,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    @property
    @lru_cache()
    def clf(self):
        return DataLoader(
            dataset=self.ds_clf(),
            batch_size=self.bs_clf,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    @property
    @lru_cache()
    def test(self):
        return DataLoader(
            dataset=self.ds_test(),
            batch_size=self.bs_test,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
