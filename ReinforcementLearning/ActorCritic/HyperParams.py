import torch

critic_lr = 5e-3
actor_lr = 2e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.98
target_update_frequency = 100
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

