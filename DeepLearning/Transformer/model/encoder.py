from torch import cat
from torch.nn import Embedding, ModuleList, Module, Linear, Sequential, ReLU, LayerNorm, Dropout

from DeepLearning.Transformer.model.attention import Attention
from DeepLearning.Transformer.model.positional_encoding import PositionalEncoding


class EncoderLayer(Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        # pad_tensor is determined by the length of sequence in a single batch. therefore we can't initialize it in __init__ method.
        # it will be passed from Encoder class, this parameter of Encoder class is passed from Transformer class.
        self.pad_tensor = None

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.dropout = dropout

        # MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        # where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
        self.multi_heads = [Attention(self.d_model, self.d_k, self.d_v) for _ in range(self.h)]
        self.ffc = Sequential(Linear(in_features=self.d_model, out_features=self.d_ff),
                              ReLU(),
                              Linear(in_features=self.d_ff, out_features=self.d_model))
        self.layer_norm1 = LayerNorm(self.d_model)
        self.layer_norm2 = LayerNorm(self.d_model)
        # We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
        # self.dropout1 = Dropout(self.dropout)
        # self.dropout2 = Dropout(self.dropout)

    def forward(self, x):
        for head in self.multi_heads:
            head.masked_tensor = self.pad_tensor
        multi_heads = cat([head(x, x, x) for head in self.multi_heads], dim=2)
        # dropout_multi_heads = self.dropout1(multi_heads)
        x = self.layer_norm1(x + multi_heads)
        ffc = self.ffc(x)
        # dropout_ffc = self.dropout2(ffc)
        # The output of each sub-layer is LayerNorm(x + Sublayer(x))
        x = self.layer_norm2(x + ffc)
        return x


class Encoder(Module):
    def __init__(self, chinese_vocab_len, d_model, N, d_k, d_v, h, d_ff, dropout):
        super(Encoder, self).__init__()
        self.chinese_vocab_len = chinese_vocab_len
        self.d_model = d_model
        self.N = N
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff

        # pad_tensor is determined by the length of sequence in a single batch. therefore we can't initialize it in __init__ method.
        self.pad_tensor = None

        # shape of self.embedding is (chinese_vocab_len, d_model)
        # As the inputting data is of the form [word1_index, word2_index, ..., wordn_index], we use Embedding instead of nn.Linear.
        self.embedding = Embedding(num_embeddings=self.chinese_vocab_len, embedding_dim=self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.layers = ModuleList(
            [EncoderLayer(self.d_model, self.d_k, self.d_v, self.h, self.d_ff, dropout) for _ in range(self.N)])
        # We apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.
        # self.dropout = Dropout(dropout)

    def forward(self, encoder_input):
        # convert onehot embedding to word embedding.
        encoder_output = self.embedding(encoder_input)
        # add positional encoding to word embedding.
        encoder_output = self.positional_encoding(encoder_output)
        # We apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.
        # encoder_output = self.dropout(encoder_output)
        # go through N identical transformer blocks. each block has two sub-layers. the first is a multi-head self-attention mechanism,
        # and the second is a simple, position-wise fully connected feed-forward network.
        for layer in self.layers:
            layer.pad_tensor = self.pad_tensor
            encoder_output = layer(encoder_output)
        return encoder_output
