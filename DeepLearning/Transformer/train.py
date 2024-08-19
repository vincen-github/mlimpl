import logging
from torch import LongTensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler

from Utils import indices2sentence
from DeepLearning.Transformer.model.transformer import Transformer
from DeepLearning.Transformer.params import d_model, N, d_k, d_v, h, d_ff, dropout, num_epochs, lr, device
from DeepLearning.Transformer.dataloader import translation2019zh_train, dataloader_train, dataloader_valid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model = Transformer(chinese_vocab_len=translation2019zh_train.chinese_vocab_len,
                    english_vocab_len=translation2019zh_train.english_vocab_len,
                    d_model=d_model,
                    N=N,
                    d_k=d_k,
                    d_v=d_v,
                    h=h,
                    d_ff=d_ff,
                    dropout=dropout)

model = model.to(device)

criterion = CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=lr)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12, 15, 20, 25, 30], gamma=0.5)

valid_loss = []
for epoch_train in range(num_epochs):
    for iter, (chinese_train, english_train) in enumerate(dataloader_train):
        chinese_train = chinese_train.to(device)
        english_train = english_train.to(device)

        if (epoch_train - 1) % 10 == 0:
            model.eval()
            for valid_iter, (chinese_valid, english_valid) in enumerate(dataloader_valid):
                out = model(chinese_valid.to(device), english_valid.to(device))
                logger.info("valid iter: {} \n chinese: {} \n english: {} \n translated english: {}".format(
                    valid_iter,
                    indices2sentence(chinese_valid[0], translation2019zh_train.chinese_vocab, False),
                    indices2sentence(english_valid[0], translation2019zh_train.english_vocab, True),
                    indices2sentence(out[0].argmax(dim=1), translation2019zh_train.english_vocab, True)
                ))
                current_batch_size = out.size(0)
                seq_length = out.size(1)
                loss_valid = criterion(out.view(current_batch_size * seq_length, -1),
                                       english_valid.view(-1).type(LongTensor).to(device))
                valid_loss.append(loss_valid.item())
            model.train()

        out = model(chinese_train, english_train)
        current_batch_size = out.size(0)
        seq_length = out.size(1)
        loss_train = criterion(out.view(current_batch_size * seq_length, -1),
                               english_train.view(-1).type(LongTensor).to(device))

        logger.info(
            f'Epoch [{epoch_train + 1}/{num_epochs}], Step [{iter + 1}/{len(dataloader_train)}], training loss: {loss_train.item():.4f}')

        logger.info("train iter: {} \n chinese: {} \n english: {} \n translated english: {}".format(
            iter,
            indices2sentence(chinese_train[0], translation2019zh_train.chinese_vocab, False),
            indices2sentence(english_train[0], translation2019zh_train.english_vocab, True),
            indices2sentence(out[0].argmax(dim=1), translation2019zh_train.english_vocab, True)
        ))

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    scheduler.step()
