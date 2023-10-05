from datetime import datetime

from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam

from val_reader import val_loader
from validation import validate
from train_reader import train_loader

from model import ResNet50
from params import num_epochs, device, lr

resnet50 = ResNet50().to(device)
cross_entropy = CrossEntropyLoss()
optimizer = Adam(resnet50.parameters(), lr=lr)

for epoch in range(num_epochs):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for iters, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outs = resnet50(imgs)
        train_loss = cross_entropy(outs, labels)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        accuracy = validate(resnet50, val_loader)
        currentTime = datetime.now().strftime("%y-%m-%d %H:%M:%S")
        pbar.set_description(
            f'{currentTime}, Epoch [{epoch + 1}/{num_epochs}], Step [{iters + 1}/{len(train_loader)}], val_accuracy: {accuracy}')
