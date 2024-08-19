from networks import Critic, Generator
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import ones, rand, randn, mean, pow, norm, no_grad
from torch.cuda import is_available
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter


# Hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 256
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
# latent_channels is used to indicate the channels of latent representation 
LATENT_CHANNELS = 128
Z_DIM = 100
CRITIC_ITERS = 5
LAMBDA = 10
LEARNING_RATE = 5e-4

device = "cuda" if is_available() else "cpu"

# data_loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
anime_faces = ImageFolder(r'data/', transform=transform)
dataloader = DataLoader(anime_faces, BATCH_SIZE, shuffle=True)

# models
generator = Generator(Z_DIM, LATENT_CHANNELS, IMAGE_CHANNELS).to(device)
critic = Critic(IMAGE_CHANNELS, LATENT_CHANNELS).to(device)

# optimizers
generator_opt = Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0, 0.9))
critic_opt = Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0, 0.9))

writer_real = SummaryWriter(f"logs/real_GP")
writer_fake = SummaryWriter(f"logs/fake_GP")
step = 0

generator.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        current_batch_size = real.shape[0]
        real = real.to(device)
        # Train critic
        for _ in range(CRITIC_ITERS):
            noise = randn(current_batch_size, Z_DIM, 1, 1).to(device)
            fake = generator(noise)
            # Uniformly sample along straight lines between pairs of points sampled from the data distribution Pr and the generator distribution Pg
            eps = rand((current_batch_size, 1, 1, 1)).repeat(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            # hatx's shape: (BATCH_SIZE, IMG_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            hatx = eps * real + (1 - eps) * fake
            # hatx_score's shape : BATCH_SIZE
#            hatx_score = critic(hatx)
#            gradients = grad(
#                outputs=hatx_score,
#                inputs=hatx,
                # torch doesn't support to calculate derivative of tensors w.r.t tensors, 
                # thus it is necessary to set the augment `grad_outputs` to be a tensor possessing same shape as y.
                # torch will compute the derivative of y_i w.r.t x and then weighted sum them according to passed `grad_outputs`.
                # For example, suppose current_batch_size = 4, thus the shape of hatx is [4, IMG_CHANNELS, IMAGE_SIZE, IMAGE_SIZE]
                # the mathematic formulation of hatx_score = [f(hatx_1), f(hatx_2), f(hatx_3), f(hatx_4)], but torch views it as 
                #                                   hatx_score = [f1(hatx), f2(hatx), f3(hatx), f4(hatx)]
                # for any i = 1, 2, 3, 4, the shape of ∂(fi) / ∂(hatx) still remains [4, IMG_CHANNELS, IMAGE_SIZE, IMAGE_SIZE]
                # but all elements of ∂(f1) / ∂(hatx) are vanish except for the block ∂(f1) / ∂(hatx)[0, :] as f1 only depends on hatx_1.
                # therefore the summation operation in here is make sense.
 #               grad_outputs=ones_like(hatx_score),
 #               create_graph=True,
 #               retain_graph=True
 #           )[0]
            hatx_score = critic(hatx).reshape(-1)
            gradients = grad(
                outputs=hatx_score,
                inputs=hatx,
                # torch doesn't support to calculate derivative of tensors w.r.t tensors, 
                # thus it is necessary to set the augment `grad_outputs` to be a tensor possessing same shape as y.
                # torch will compute the derivative of y_i w.r.t x and then weighted sum them according to passed `grad_outputs`.
                # For example, suppose current_batch_size = 4, thus the shape of hatx is [4, IMG_CHANNELS, IMAGE_SIZE, IMAGE_SIZE]
                # the mathematic formulation of hatx_score = [f(hatx_1), f(hatx_2), f(hatx_3), f(hatx_4)], but torch views it as 
                #                                   hatx_score = [f1(hatx), f2(hatx), f3(hatx), f4(hatx)]
                # for any i = 1, 2, 3, 4, the shape of ∂(fi) / ∂(hatx) still remains [4, IMG_CHANNELS, IMAGE_SIZE, IMAGE_SIZE]
                # but all elements of ∂(f1) / ∂(hatx) are vanish except for the block ∂(f1) / ∂(hatx)[0, :] as f1 only depends on hatx_1.
                # therefore the summation operation in here is make sense.
                grad_outputs=ones(current_batch_size).to(device),
                create_graph=True,
                retain_graph=True
            )[0]
            # BATCH_SIZE x 1 x 1 x 1 -> BATCH_SIZE x 1
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            # Wasserstein loss
            critic_loss = mean(critic_fake) - mean(critic_real) + LAMBDA * mean(pow(gradients.view(current_batch_size, -1).norm(2, dim=1) - 1, 2))
            critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_opt.step()
        
        # Train generator
        generator_loss = -mean(critic(fake).reshape(-1)) 
        generator.zero_grad()
        generator_loss.backward()
        generator_opt.step()
        
        if batch_idx % 10 == 0:
            generator.eval()
            critic.eval()
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} Loss D: {critic_loss:.4f}, loss G: {generator_loss:.4f}"
            )
            
            with no_grad():
                fake = generator(noise)
                real_grid = make_grid(
                    real[:32], normalize=True
                )
                fake_grid = make_grid(
                    fake[:32], normalize=True
                )
                
                writer_real.add_image("Real", real_grid, global_step=step)
                writer_fake.add_image("Fake", fake_grid, global_step=step)
                
            step += 1
            generator.train()
            critic.train()
                

