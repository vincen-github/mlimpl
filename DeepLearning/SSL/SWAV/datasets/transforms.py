import torchvision.transforms as T
#from .gaussian_blur import GaussianBlur

def aug_transform(crop, base_transform, cfg, extra_t=[]):
    """ augmentation transform generated from config """
    return T.Compose(
        [
            T.RandomApply(
                [T.ColorJitter(cfg.cj_bright, cfg.cj_contrast, cfg.cj_sat, cfg.cj_hue)], p=cfg.cj_prob
            ),
            T.RandomGrayscale(p=cfg.gs_prob),
            T.RandomResizedCrop(
                crop,
                scale=(cfg.crop_s0, cfg.crop_s1),
                ratio=(cfg.crop_r0, cfg.crop_r1),
                interpolation=3,
            ),
            T.RandomHorizontalFlip(p=cfg.hf_prob),
#            GaussianBlur(kernel_size=cfg.kernel_size * crop, prob=cfg.blur_prob),
            *extra_t,
            base_transform(),
        ]
    )


class MultiSample:
    """ generates n samples with augmentation """

    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))
