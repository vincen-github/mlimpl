import torch

lr = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

