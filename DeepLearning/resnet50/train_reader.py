from os import chdir

from numpy import stack
from h5py import File
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, RandomHorizontalFlip, ToTensor

from params import train_batch_size, hdf5_path, hdf5_names, num_gpus
from val_reader import class_to_idx

# enter the folder where the data is stored.
chdir(hdf5_path)


class imagenet1k_h5(Dataset):
    def __init__(self):
        self.hdf5name_and_imgname = []
        self.files = dict()
        for hdf5_name in hdf5_names:
            file = File(hdf5_name, 'r')
            for img_name in file.keys():
                # record img_name and the corresponding hdf5_name of each image.
                self.hdf5name_and_imgname.append((hdf5_name, img_name))
                # Note that you can't use order of hdf5_names to get label of img, as it is not coincide with val data.
            # store file into files to avoid Repeatedly reading files when invoke __getitem__.
            self.files[hdf5_name] = file

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
        # get image name and hdf5 name it belongs to through index
        hdf5_name, imgname = self.hdf5name_and_imgname[index]
        label = class_to_idx[hdf5_name.split('.')[0]]

        file = self.files[hdf5_name]

        img = file[imgname][:]

        # If img only has single channel, copy it three times and stack them to triple its channel.
        if len(img.shape) != 3:
            img = stack((img,) * 3, axis=-1)
        # In the process of running code, I encountered the error that
        # 'The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0'
        # This is because there is an image in the training set that is a PNG image, and the suffix name is
        # forcibly changed to. JPEG, but the PNG image still retains its 4 channels
        # (in addition to the RGB three channels, there is also an Alpha channel for the image's transparency)
        # instance n02105855_2933.JPEG
        # reference: https://xungejiang.com/2019/10/20/imagenet-train-set/

        if img.shape[-1] == 4:
            img = img[:, :, :3]

        return self.transform(img), label

    def __len__(self):
        return len(self.hdf5name_and_imgname)


imagenet1k = imagenet1k_h5()
train_loader = DataLoader(imagenet1k, batch_size=train_batch_size, shuffle=True, num_workers=4 * num_gpus,
                          pin_memory=True, prefetch_factor=10 * num_gpus)

if __name__ == "__main__":
    for iters, (imgs, labels) in enumerate(train_loader):
        print(imgs.shape, labels)

