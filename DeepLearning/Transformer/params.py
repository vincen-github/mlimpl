import torch

train_path = "./translation2019zh/translation2019zh_train.json"
valid_path = "./translation2019zh/translation2019zh_valid.json"

lr = 1e-3
num_epochs = 32

train_batch_size = 128
valid_batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# The output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by
# the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the
# embedding layers, produce outputs of dimension dmodel = 512.
d_model = 512
# the encoder is composed of a stack of N = 6 identical layers.
N = 6
# the number of mult-head
h = 8
# d_k is the dimension of the key and query
d_k = int(d_model / h)
# d_v is the dimension of the value
d_v = d_k
# output dimension of the feedforward network
d_ff = 2048
# dropout is the dropout rate
dropout = 0.1

# special token
unk_id = 0
pad_id = 1
bos_id = 2
eos_id = 3
