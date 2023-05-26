import torch

lr = 3e-3
num_episodes = 300
hidden_dim = 128
gamma = 0.97
eps = 0.01
target_update_frequency = 10
buffer_capacity = 1000
batch_size = 64
# minimal size determines when to start updating the parameters of the neural network using data from the experience
# replay pool
minimal_size = 500

# print(torch.cuda.device_count())
# 1
# print(torch.cuda.get_device_name(0))
# NVIDIA GeForce RTX 3050 Laptop GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")