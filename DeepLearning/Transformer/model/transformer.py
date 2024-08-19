from torch import softmax
from torch.nn import Module, Linear

from DeepLearning.Transformer.Utils import get_pad_tensor, get_masked_tensor
from DeepLearning.Transformer.params import pad_id
from DeepLearning.Transformer.model.decoder import Decoder
from DeepLearning.Transformer.model.encoder import Encoder


class Transformer(Module):
    """
    parameters:
        1. chinese_vocab_len: the number of words in the vocabulary of chinese.
        2. english_vocab_len: the number of words in the vocabulary of english.
        3. pad_id: the index of <pad> in the vocabulary of chinese.
        4. d_model: The output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by
        the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers,
        produce outputs of dimension dmodel = 512.
        5. N: the encoder is composed of a stack of N = 6 identical layers.
        6. d_k: the dimension of the key and query
        7. d_v: the dimension of the value
        8. h: the number of mult-head
        9. d_ff: output dimension of the feedforward network
    """

    def __init__(self, chinese_vocab_len, english_vocab_len, d_model, N, d_k, d_v, h, d_ff,
                 dropout=0.1):
        super(Transformer, self).__init__()
        self.chinese_vocab_len = chinese_vocab_len
        self.english_vocab_len = english_vocab_len

        # In this work we employ h = 8 parallel attention layers, or heads.For each of these we use dk = dv = dmodel / h = 64.
        assert d_model % h == 0, "d_model % h != 0"
        assert d_model / d_k == h, "d_model % d_k != h"
        assert d_model / d_v == h, "d_model % d_v != h"

        self.d_model = d_model
        self.N = N
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.dropout = dropout

        self.encoder = Encoder(self.chinese_vocab_len, self.d_model, self.N, self.d_k, self.d_v, self.h, self.d_ff,
                               self.dropout)
        self.decoder = Decoder(self.english_vocab_len, self.d_model, self.N, self.d_k, self.d_v, self.h, self.d_ff,
                               self.dropout)
        self.projection = Linear(self.d_model, self.english_vocab_len, bias=False)

    def forward(self, encoder_input, decoder_input):
        # get encoder padding tensor in eliminate the influence of <pad> when computing attention score.
        self.encoder.pad_tensor = get_pad_tensor(encoder_input, pad_id)
        # the shape of encoder_output is same as encoder_input.
        encoder_output = self.encoder(encoder_input)

        self.decoder.masked_tensor = get_masked_tensor(decoder_input, pad_id)
        self.decoder.encoder_decoder_pad_tensor = get_pad_tensor(encoder_input, pad_id, decoder_input)
        # get the output of decoder, the shape of it is [batch_size, seq_len, d_model]
        decoder_output = self.decoder(decoder_input, encoder_output)
        # project the output of decoder to the shape of [batch_size, seq_len, english_vocab_len]
        logit = softmax(self.projection(decoder_output), dim=2)

        return logit
