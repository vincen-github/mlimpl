from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from params import val_path, num_gpus, val_batch_size, hdf5_names

transform = Compose([
	ToTensor(),
	Resize(256),
	CenterCrop(224),
	Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset = ImageFolder(val_path, transform=transform)

# Because in Linux, The label constructed by the element order in 
# hdf5_names=[hdf5_name for hdf5_name in listdir (hdf5_path) if hdf5name.endswith ('. h5')] does not match 
# the file name order in ImageFolder, so we need a class_ To_ IDX is used to create tags in the training set.
class_to_idx = val_dataset.class_to_idx

val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=12,
                        pin_memory=True,
                        prefetch_factor=10 * num_gpus)

