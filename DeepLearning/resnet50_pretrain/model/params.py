from os import listdir

import torch

# train_path
hdf5_path = r"C:\Users\WenSen Ma\OneDrive - whu.edu.cn\桌面\ILSVRC2012_img_val"
val_path = r"C:\Users\WenSen Ma\OneDrive - whu.edu.cn\桌面\val"

batch_size = 32
num_epochs = 120
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.1

hdf5_names = [hdf5_name for hdf5_name in listdir(hdf5_path) if hdf5_name.endswith('.h5')]
# convert the image class to int
label_to_int = {hdf5_name: i for i, hdf5_name in enumerate(hdf5_names)}
