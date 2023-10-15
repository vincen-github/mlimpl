import torch
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, RandomHorizontalFlip, Resize, CenterCrop, \
    Normalize

train_imgs_path = r"C:\Users\WenSen Ma\OneDrive - whu.edu.cn\桌面\aptos2019-blindness-detection\train_images"
train_labels_path = r"C:\Users\WenSen Ma\OneDrive - whu.edu.cn\桌面\aptos2019-blindness-detection\train.csv"
model_params_path = r'C:\Users\WenSen Ma\OneDrive - whu.edu.cn\桌面\mlimpl\DeepLearning\TransferLearning\resnet50_pretraining.pth'

num_gpus = torch.cuda.device_count()
train_batch_size = 64 * num_gpus
val_proportion = 0.3
val_batch_size = 64 * num_gpus
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 1e-4

transform = {
    'train': Compose([ToTensor(),
                      RandomResizedCrop(224, antialias=True),
                      RandomHorizontalFlip(),
                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'val': Compose([
        ToTensor(),
        Resize(256),
        CenterCrop(224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
}
