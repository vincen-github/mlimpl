import torch

num_episodes = 200

vnet_hidden_dim = 128
anet_hidden_dim = 128
lr = 1e-2
gamma = 0.98
eps = 0.1
target_update_frequency = 50
buffer_capacity = 5000
batch_size = 128
minimal_size = 1000

update_mode = "mean"
share = True

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
