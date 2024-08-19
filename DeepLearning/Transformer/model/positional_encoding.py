from numpy import sin, cos
from torch.nn import Module


class PositionalEncoding(Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        # x is a sentence with shape (batch_size, seq_len, d_model)
        # where pos is the position of the word in the sentence and 2i, 2i+1 is the dimension of word's representation.
        for pos in range(x.shape[1]):
            for i in range(self.d_model):
                if i % 2 == 0:
                    x[:, pos, i] += sin(pos / (1e4 ** (i / self.d_model)))
                else:
                    x[:, pos, i] += cos(pos / (1e4 ** (2 * (i // 2) / self.d_model)))
        return x
