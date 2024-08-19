from networks import Critic, Generator
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import randn, mean, no_grad
from torch.cuda import is_available
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
WEIGHT_CLIP = 0.01
LEARNING_RATE = 5e-4

device = "cuda" if is_available() else "cpu"

# data_loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
anime_faces = ImageFolder(r'./data/', transform=transform)
dataloader = DataLoader(anime_faces, BATCH_SIZE, shuffle=True)

# models
generator = Generator(Z_DIM, LATENT_CHANNELS, IMAGE_CHANNELS).to(device)
critic = Critic(IMAGE_CHANNELS, LATENT_CHANNELS).to(device)

# optimizers
generator_opt = Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
critic_opt = Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

generator.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        
        # Train critic
        for _ in range(CRITIC_ITERS):
            noise = randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
            fake = generator(noise)
            # BATCH_SIZE x 1 x 1 x 1 -> BATCH_SIZE x 1
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            # Wasserstein loss
            critic_loss = mean(critic_fake) - mean(critic_real)
            critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_opt.step()
        
            # weight clip
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
        
        # Train generator
        generator_loss = -mean(critic(fake).reshape(-1)) 
        generator.zero_grad()
        generator_loss.backward()
        generator_opt.step()
        
        if batch_idx % 100 == 0:
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
                
