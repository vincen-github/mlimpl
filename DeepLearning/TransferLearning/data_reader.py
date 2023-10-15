from os import listdir, chdir

from PIL import Image
from pandas import read_csv
from torch.utils.data import Dataset, DataLoader, random_split
from DeepLearning.TransferLearning.params import train_imgs_path, train_labels_path, transform, train_batch_size, \
    val_proportion, val_batch_size

chdir(train_imgs_path)


class aptos2019_blindness_detection(Dataset):
    def __init__(self):
        self.img_names = listdir(train_imgs_path)
        self.transform = transform['train']
        self.labels = read_csv(train_labels_path, header=0, index_col=0)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        label = self.labels.loc[img_name.split('.')[0], 'diagnosis']
        imgs = Image.open(img_name)

        return self.transform(imgs), label

    def __len__(self):
        return len(self.img_names)


dataset = aptos2019_blindness_detection()
val_size = int(len(dataset) * val_proportion)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# As the transform applied to training data and validation data are different, altering it manually is compulsory.
val_dataset.transform = transform['val']

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

if __name__ == '__main__':
    for iters, (imgs, labels) in enumerate(train_loader):
        print(imgs.shape, labels)
