import logging
from json import loads

from torch import IntTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from DeepLearning.Transformer.Utils import is_contain_zh
from DeepLearning.Transformer.params import train_path, valid_path, valid_batch_size, pad_id, train_batch_size

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class translation2019zh2en(Dataset):
    def __init__(self, train=True):
        # tokenizer is used to cut a sentence into a list of words.
        self.en_tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
        self.zh_tokenizer = get_tokenizer(tokenizer='spacy', language='zh_core_web_sm')
        # Following character is special token in vocabulary.
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        self.train = train
        self.zh, self.en = [], []
        path = train_path if self.train else valid_path
        with open(path, mode='r', encoding='utf-8') as file:
            sentence_pairs = file.readlines()
            for sentence_pair in sentence_pairs:
                sentence_pair = loads(sentence_pair)
                # There are some data with opposite Chinese and English positions, we need to make corrections.
                if is_contain_zh(sentence_pair['english']):
                    # If the sentence contains Chinese characters, it means that the Chinese and English sentences are reversed.
                    # Note that using isascii() to judge whether opposite phenomenon happens is not working.
                    # As there exists some characters like ’, ——, “ which is not ascii character in english sentences.
                    self.zh.append(self.zh_tokenizer(sentence_pair['english']))
                    self.en.append(self.en_tokenizer(sentence_pair['chinese']))
                else:
                    self.zh.append(self.zh_tokenizer(sentence_pair['chinese']))
                    self.en.append(self.en_tokenizer(sentence_pair['english']))

        # <pad> is used to pad a sentence to the same length.
        # <unk> is used to represent a word that is not in the vocabulary.
        # <bos> is used to represent the beginning of a sentence.
        # <eos> is used to represent the end of a sentence.
        self.english_vocab = build_vocab_from_iterator(self.english,
                                                       specials=["<unk>", "<pad>", "<bos>", "<eos>"],
                                                       special_first=True,
                                                       min_freq=10)
        self.english_vocab.set_default_index(self.english_vocab["<unk>"])
        # english_vocab_len is the number of words in the vocabulary of english.
        self.english_vocab_len = len(self.english_vocab)

        self.chinese_vocab = build_vocab_from_iterator(self.chinese,
                                                       specials=["<unk>", "<pad>"],
                                                       special_first=True,
                                                       min_freq=10)
        self.chinese_vocab.set_default_index(self.chinese_vocab["<unk>"])
        # chinese_vocab_len is the number of words in the vocabulary of chinese.
        self.chinese_vocab_len = len(self.chinese_vocab)

        if self.train:
            logger.info("English vocabulary has been built, the size of it is {}.".format(self.english_vocab_len))
            logger.info("Chinese vocabulary has been built, the size of it is {}.".format(self.chinese_vocab_len))

        # Add <bos> and <eos> to the beginning and end of the sentences.
        for sentence in self.english:
            sentence.insert(0, '<bos>')
            sentence.append('<eos>')
        # Use the vocabulary to convert the cut sentence to a list of onehot embedding.
        self.english = [self.english_vocab(sentence) for sentence in self.english]
        self.chinese = [self.chinese_vocab(sentence) for sentence in self.chinese]

    def __getitem__(self, index):
        return IntTensor(self.chinese[index]), IntTensor(self.english[index])

    def __len__(self):
        # len(translation2019zh) will return the number of sentences.
        return len(self.english)

    def collate_fn(self, batch):
        x, y = zip(*batch)
        # In our setting, the padding token is <pad>, thus pad_index = 1.
        x_pad = pad_sequence(x, batch_first=True, padding_value=pad_id)
        y_pad = pad_sequence(y, batch_first=True, padding_value=pad_id)
        return x_pad, y_pad


translation2019zh_train = translation2019zh2en(train=False)
#
# dataloader_train = DataLoader(translation2019zh_train, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
# # num_workers=4 * num_gpus, pin_memory=True, prefetch_factor=10 * num_gpus)
# dataloader_valid = DataLoader(translation2019zh_valid, batch_size=valid_batch_size, shuffle=True, collate_fn=collate_fn)
# # num_workers=4 * num_gpus, pin_memory=True, prefetch_factor=10 * num_gpus)
