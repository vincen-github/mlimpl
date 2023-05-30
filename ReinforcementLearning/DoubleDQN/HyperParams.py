import torch

lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.97
eps = 0.01
target_update_frequency = 50
buffer_capacity = 10000
batch_size = 64
# minimal size determines when to start updating the parameters of the neural network using data from the experience
# replay pool
minimal_size = 1000

# print(torch.cuda.device_count())
# 1
# print(torch.cuda.get_device_name(0))
# NVIDIA GeForce RTX 3050 Laptop GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")