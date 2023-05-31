import torch

lr = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.98
eps = 0.1
target_update_frequency = 100
buffer_capacity = 5000
batch_size = 128
# minimal size determines when to start updating the parameters of the neural network using data from the experience
# replay pool
minimal_size = 1000

# print(torch.cuda.device_count())
# 1
# print(torch.cuda.get_device_name(0))
# NVIDIA GeForce RTX 3050 Laptop GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")