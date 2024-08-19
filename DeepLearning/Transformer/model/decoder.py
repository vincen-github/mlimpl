from torch import cat
from torch.nn import Module, Linear, Sequential, ReLU, LayerNorm, Dropout, Embedding, ModuleList

from DeepLearning.Transformer.model.attention import Attention
from DeepLearning.Transformer.model.positional_encoding import PositionalEncoding


class DecoderLayer(Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        # To prevent model get the information from the latter words of current word, we need to mask the latter words of current word.
        self.masked_tensor = None
        # encoder_decoder_pod_tensor will be used in encoder-decoder attention, it is a matrix with shape (batch_size, seq_len1, seq_len2)
        # seq_len1 is the length of sequence of decoder input, seq_len2 is the length of sequence of encoder output.
        self.encoder_decoder_pad_tensor = None

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.dropout = dropout

        # masked self attention, the input only contains the words before current word in decoder input.
        self.masked_multi_heads = [Attention(self.d_model, self.d_k, self.d_v) for _ in range(self.h)]
        # non-self attention, the input contains both encoder output and decoder input. among them, amin them, Q comes from decoder input, K and V come from encoder output.
        self.multi_heads = [Attention(self.d_model, self.d_k, self.d_v) for _ in range(self.h)]
        self.ffc = Sequential(Linear(in_features=self.d_model, out_features=self.d_ff),
                              ReLU(),
                              Linear(in_features=self.d_ff, out_features=self.d_model))
        self.layer_norm1 = LayerNorm(self.d_model)
        self.layer_norm2 = LayerNorm(self.d_model)
        self.layer_norm3 = LayerNorm(self.d_model)
        # self.dropout1 = Dropout(self.dropout)
        # self.dropout2 = Dropout(self.dropout)
        # self.dropout3 = Dropout(self.dropout)

    def forward(self, decoder_embedding, encoder_output):
        for head in self.masked_multi_heads:
            head.masked_tensor = self.masked_tensor
        masked_multi_heads = cat(
            [head(decoder_embedding, decoder_embedding, decoder_embedding) for head in self.masked_multi_heads],
            dim=2)
        # dropout_masked_multi_heads = self.dropout1(masked_multi_heads)
        decoder_output = self.layer_norm1(decoder_embedding + masked_multi_heads)

        for head in self.multi_heads:
            head.masked_tensor = self.encoder_decoder_pad_tensor
        multi_heads = cat([head(encoder_output, decoder_output, encoder_output) for head in self.multi_heads], dim=2)
        # dropout_multi_heads = self.dropout2(multi_heads)
        decoder_output = self.layer_norm2(decoder_output + multi_heads)

        ffc = self.ffc(decoder_output)
        # dropout_ffc = self.dropout2(ffc)
        decoder_output = self.layer_norm2(decoder_output + ffc)

        return decoder_output


class Decoder(Module):
    def __init__(self, english_vocab_len, d_model, N, d_k, d_v, h, d_ff, dropout):
        super(Decoder, self).__init__()

        self.english_vocab_len = english_vocab_len

        self.d_model = d_model
        self.N = N
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.d_ff = d_ff
        self.dropout = dropout

        self.decoder_input_masked_tensor = None
        self.encoder_decoder_pad_tensor = None

        # transform
        self.embedding = Embedding(num_embeddings=self.english_vocab_len, embedding_dim=self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.layers = ModuleList(
            [DecoderLayer(self.d_model, self.d_k, self.d_v, self.h, self.d_ff, self.dropout) for _ in range(self.N)])
        # self.dropout1 = Dropout(self.dropout)

    def forward(self, decoder_input, encoder_output):
        decoder_output = self.embedding(decoder_input)
        decoder_output = self.positional_encoding(decoder_output)
        # decoder_output = self.dropout1(decoder_output)

        for layer in self.layers:
            layer.masked_tensor = self.masked_tensor
            layer.encoder_decoder_pad_tensor = self.encoder_decoder_pad_tensor
            decoder_output = layer(decoder_output, encoder_output)

        return decoder_output
