from os import chdir

from PIL.Image import fromarray
from h5py import File
from numpy import stack
from torch.utils.data import Dataset, DataLoader
from torchvision import disable_beta_transforms_warning
from torchvision.transforms import ToTensor, Compose, Normalize, RandomResizedCrop, RandomHorizontalFlip

from params import batch_size, hdf5_path, hdf5_names, label_to_int

disable_beta_transforms_warning()

# enter the folder where the data is stored.
chdir(hdf5_path)


class imagenet1k_h5(Dataset):
    def __init__(self):
        # self.file_paths is aim to store the path of every img.
        self.img_belonged_hdf5_and_key = []
        self.labels = []
        for hdf5_name in hdf5_names:
            with File(hdf5_name, 'r') as file:
                for key in file.keys():
                    # record the hdf5 file name and corresponding key of each image
                    self.img_belonged_hdf5_and_key.append((hdf5_name, key))
                    # As I named hdf5 file to be the label of image when i build it.
                    # thus the label of each image can be got from the name of hdf5 file.
                    self.labels.append(label_to_int[hdf5_name])

        # you need to deal the size of image to be same that make there is no error when build dataloader
        # here we use the same transform stated in kaiming's paper.
        self.transform = Compose([
            ToTensor(),
            RandomResizedCrop(224, antialias=True),
            RandomHorizontalFlip(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __getitem__(self, index):
        # Note that the index passed above is the position w.r.t the whole image_net set
        img_belonged_hdf5, img_key = self.img_belonged_hdf5_and_key[index]
        label = self.labels[index]
        with File(img_belonged_hdf5, 'r') as file:
            img = file[img_key][:]
            # If img only has single channel, copy it three times and stack them to triple its channel.
            if len(img.shape) != 3:
                img = fromarray(stack((img,) * 3, axis=-1))
        return self.transform(img), label

    def __len__(self):
        return len(self.img_belonged_hdf5_and_key)


imagenet1k = imagenet1k_h5()
train_loader = DataLoader(imagenet1k, batch_size=batch_size, shuffle=True)
