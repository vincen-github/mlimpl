from torch.utils.data import DataLoader
from torchvision import disable_beta_transforms_warning
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from params import val_path

disable_beta_transforms_warning()

transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset = ImageFolder(val_path, transform=transform)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)