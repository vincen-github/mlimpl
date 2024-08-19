import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, img_channels, latent_channels):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            # input_size: BATCH_SIZE x img_channels x 64 x 64
            nn.Conv2d(
                img_channels, latent_channels, kernel_size=4, stride=2, padding=1
            ),
            # BATCH_SIZE x latent_channels x 32 x 32
            nn.LeakyReLU(0.2),
            self._block(latent_channels, latent_channels * 2, kernel_size=4, stride=2, padding=1),
            # BATCH_SIZE x latent_channels x 16 x 16
            self._block(latent_channels * 2, latent_channels * 4, kernel_size=4, stride=2, padding=1),
            # BATCH_SIZE x latent_channels x 8 x 8
            self._block(latent_channels * 4, latent_channels * 8, kernel_size=4, stride=2, padding=1),
            # BATCH_SIZE x latent_channels x 4 x 4
            nn.Conv2d(latent_channels * 8, 1, kernel_size=4, stride=2, padding=0),
            # BATCH_SIZE x 1 x 1 x 1
            nn.Sigmoid(),
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias = False,
            ),
#            nn.BatchNorm2d(out_channels),
            # wgan_gp adopt LayerNorm to replace BatchNorm2d of wgan_clip
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        return self.critic(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, latent_channels, img_channels):
        # latent_channels: 128
        # img_channels: 64
        # ConvTranspose2d: out_size = (in_size - 1) * stride - 2 * padding + kernel_size
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            # input: BATCH_SIZE x z_dim x 1 x 1
            self._block(z_dim, latent_channels * 8, kernel_size= 4, stride=1, padding=0),
            # BATCH_SIZE x 1024 x 4 x 4
            self._block(latent_channels * 8, latent_channels * 4, kernel_size=4, stride=2, padding=1),
            # BATCH_SIZE x 512 x 8 x 8
            self._block(latent_channels * 4, latent_channels * 2, kernel_size=4, stride=2, padding=1),
            # BATCH_SIZE x 256 x 16 x 16
            self._block(latent_channels * 2, latent_channels, kernel_size=4, stride=2, padding=1),
            # BATCH_SIZE x 128 x 32 x 32
            nn.ConvTranspose2d(
                latent_channels,
                img_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            # BATCH_SIZE x 3 x 64 x 64
            nn.Tanh()
            # map to [-1, 1]
            # Use ReLU activation in generator for all layers except for the output, which uses Tanh
        )
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            # ConvTranspose2d: out_size = (in_size - 1) * stride - 2 * padding + kernel_size
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            # Use ReLU activation in generator for all layers except for the output, which uses Tanh
            nn.ReLU()
        )
            
    def forward(self, x):
        return self.generator(x)
