from os import listdir

import torch

# train_path
hdf5_path = r"/home/mawensen/scratch/image-net-1k/ILSVRC2012_img_train/"
val_path = r"/home/mawensen/scratch/image-net-1k/ILSVRC2012_img_val/"
save_model_path = r"/home/mawensen/scratch/code/res/model.pt"
save_val_err_path = r"/home/mawensen/scratch/code/res/val_err.dat"
save_train_err_path = r"/home/mawensen/scratch/code/res/train_err.dat"

num_gpus = torch.cuda.device_count()
train_batch_size = 128 * num_gpus
val_batch_size = 128 * num_gpus
num_epochs = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 1e-2

hdf5_names = [hdf5_name for hdf5_name in listdir(hdf5_path) if hdf5_name.endswith('.h5')]
