import logging

from torch import load, softmax, max
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam

from DeepLearning.TransferLearning.data_reader import train_loader, val_loader
from DeepLearning.TransferLearning.params import device, num_epochs, lr, model_params_path
from DeepLearning.TransferLearning.validation import validate
from DeepLearning.resnet50.model import ResNet50

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

resnet50 = ResNet50()
resnet50.to(device)

resnet50.load_state_dict(load(model_params_path))
in_channel = resnet50.fc.in_features
resnet50.fc = Linear(in_channel, 5).to(device)

cross_entropy = CrossEntropyLoss()
optimizer = Adam(resnet50.parameters(), lr=lr)

total = 0
train_correct = 0

for epoch in range(num_epochs):
    for iters, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outs = resnet50(imgs)
        probs = softmax(outs.data, dim=1)
        _, predicted = max(probs, 1)
        total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_loss = cross_entropy(outs, labels)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # if iters % 10 == 0:
        val_correct = validate(resnet50, val_loader)
        logger.info(
            f'Epoch [{epoch + 1}/{num_epochs}], Step [{iters + 1}/{len(train_loader)}], training loss: {train_loss}'
            f', training accuracy: {train_correct / total}, validation accuracy: {val_correct}')
        total = 0
        train_correct = 0
