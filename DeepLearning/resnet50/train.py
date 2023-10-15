import logging

from torch.cuda import device_count
from torch.nn import DataParallel
from torch.nn.functional import softmax
from torch import save, max
from torch.optim import lr_scheduler

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from val_reader import val_loader
from validation import validate
from train_reader import train_loader

from model import ResNet50
from params import num_epochs, device, lr, save_val_err_path, save_model_path, save_train_err_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

resnet50 = ResNet50()

if device_count() > 1:
    logger.info("{} GPUs will be used for training...".format(device_count()))
    resnet50 = DataParallel(resnet50)
resnet50 = resnet50.to(device)

cross_entropy = CrossEntropyLoss()
optimizer = Adam(resnet50.parameters(), lr=lr)
# The learning rate starts from 0.1 and is divided by 10 when the error plateaus in original paper, we use following
# simpler policy to replace it.
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 50], gamma=0.1)

train_accs = []
val_accs = []
total = 0
train_correct = 0

for epoch in range(num_epochs):
    if epoch != 0:
        accuracy = validate(resnet50, val_loader)
        logger.info(
            f'Epoch [{epoch + 1}/{num_epochs}], accuracy: {accuracy}')
        val_accs.append(accuracy)

    for iters, (imgs, labels) in enumerate(train_loader):

        imgs = imgs.to(device)
        labels = labels.to(device)

        outs = resnet50(imgs)
        # In original code of CrossEntropy, The `input` is expected to contain the unnormalized logits for each class
        # (which do `not` need to be positive or sum to 1, in general). So we used two different variables to store
        # outs and probs, one for calculating accuracy and the other for calculating loss.
        probs = softmax(outs.data, dim=1)
        _, predicted = max(probs, 1)
        total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_loss = cross_entropy(outs, labels)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if iters % 100 == 0:
            train_accs.append(train_correct / total)
            logger.info(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{iters + 1}/{len(train_loader)}], training loss: {train_loss}'
                f', training accuracy: {train_correct / total}')
            total = 0
            train_correct = 0

    scheduler.step()

# save model
save(resnet50.state_dict(), save_model_path)

# write val errs to a file for easy plot.
with open(save_val_err_path, 'w') as file:
    file.writelines(str(val_accs))
with open(save_train_err_path, 'w') as file:
    file.writelines(str(train_accs))

