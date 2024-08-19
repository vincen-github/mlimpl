from numpy import sqrt
from torch import softmax
from torch.nn import Module, Linear

from DeepLearning.TransferLearning.params import device


class Attention(Module):
    def __init__(self, d_model, d_k, d_v):
        super(Attention, self).__init__()

        # masked_tensor indicates which position should be masked, the shape of masked_tensor is (batch_size, seq_len1, seq_len1)
        # seq_len1 is same as seq_len2 when we implement self attention. It is different from seq_len2 when we implement encoder-decoder attention.
        # As seq_len1, seq_len2 is altering with different batch(seq_len is the maximal length of sentence in a single batch), so that we can't initialize masked_tensor in __init__ method.
        self.masked_tensor = None
        # masked_tensor is equal to pad_tensor in self attention to eliminate the influence of <pad> in attention score.
        # In contrast, masked_tensor is the combination of pad_tensor and masked_tensor in encoder-decoder attention.
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.K = Linear(in_features=self.d_model, out_features=self.d_k, bias=False, device=device)
        self.Q = Linear(in_features=self.d_model, out_features=self.d_k, bias=False, device=device)
        self.V = Linear(in_features=self.d_model, out_features=self.d_v, bias=False, device=device)

    def forward(self, k, q, v):
        # k,q,v are of shape (batch_size, seq_len, d_model), the reason distinguishing them is that we will re-use the same class to implement decoder, whose input is different with each other.
        # K is the key matrix with shape (batch_size, seq_len, d_k)
        K = self.K(k)
        # Q is the query matrix with shape (batch_size, seq_len, d_k)
        Q = self.Q(q)
        # V is the value matrix with shape (batch_size, seq_len, d_v)
        V = self.V(v)

        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        # where softmax(QK^T / sqrt(d_k)) is the attention weights with shape (batch_size, seq_len, seq_len)
        # and softmax(QK^T / sqrt(d_k))V is the output of attention mechanism with shape (batch_size, seq_len, d_v)
        attention_scores = Q @ K.transpose(dim0=1, dim1=2) / sqrt(self.d_k)
        # Note that we added <pad> to complete the sentence to the same length, so we need to mask the attention scores which was padded to make it doesn't influence attention score.
        masked_attention_scores = attention_scores.masked_fill(self.masked_tensor, -1e9)
        scaled_attention_scores = softmax(masked_attention_scores, dim=2)
        attention_output = scaled_attention_scores @ V

        return attention_output
