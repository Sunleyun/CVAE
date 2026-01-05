import torch
import torch.nn as nn
from einops import rearrange

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512 * 16 * 16, z_dim)
        self.fc_logvar = nn.Linear(512 * 16 * 16, z_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc = nn.Linear(z_dim + 3 * 256 * 256, 512 * 16 * 16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, img_gt, z):
        b = img_gt.size(0)
        img_flat = img_gt.view(b, -1)
        x = torch.cat([img_flat, z], dim=1)
        h = self.fc(x)
        h = h.view(b, 512, 16, 16)
        return self.deconv(h)


class SpecularCVAE(nn.Module):
    def __init__(self, z_int=16, z_shape=16, z_tex=16):
        super().__init__()
        self.z_dim = z_int + z_shape + z_tex
        self.encoder = Encoder(self.z_dim)
        self.decoder = Decoder(self.z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img_spec, img_gt, mask):
        x = torch.cat([img_spec, mask], dim=1)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(img_gt, z)
        return recon, mu, logvar
