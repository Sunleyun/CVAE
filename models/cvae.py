# model_highlight_cvae.py
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        nn.ReLU(inplace=True),
    )


class MaskUNet(nn.Module):
    """
    Predict highlight mask LOGITS from D (no-highlight image).
    Input:  (B, 3, 200, 200)
    Output: (B, 1, 200, 200) logits (NO sigmoid here)
    """
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)                   # 200
        self.down1 = nn.Conv2d(base, base, 4, 2, 1)           # 100

        self.enc2 = conv_block(base, base * 2)                # 100
        self.down2 = nn.Conv2d(base * 2, base * 2, 4, 2, 1)   # 50

        self.enc3 = conv_block(base * 2, base * 4)            # 50
        self.down3 = nn.Conv2d(base * 4, base * 4, 4, 2, 1)   # 25

        self.bottleneck = conv_block(base * 4, base * 4)      # 25

        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1)  # 50
        self.dec3 = conv_block(base * 8, base * 2)

        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1)  # 100
        self.dec2 = conv_block(base * 4, base)

        self.up1 = nn.ConvTranspose2d(base, base, 4, 2, 1)          # 200
        self.dec1 = conv_block(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1, 1, 0)

    def forward(self, x):
        e1 = self.enc1(x)
        d1 = torch.relu(self.down1(e1))

        e2 = self.enc2(d1)
        d2 = torch.relu(self.down2(e2))

        e3 = self.enc3(d2)
        d3 = torch.relu(self.down3(e3))

        b = self.bottleneck(d3)

        u3 = self.up3(b)
        o3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(o3)
        o2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(o2)
        o1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out(o1)  # logits


class ConvEncoder(nn.Module):
    """
    q(z | cond, target), GAP -> no fixed spatial size.
    Downsample 3 times: 200->100->50->25
    """
    def __init__(self, in_ch: int, z_dim: int, base_ch: int = 64):
        super().__init__()
        ch = base_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(ch * 4, z_dim)
        self.fc_logvar = nn.Linear(ch * 4, z_dim)

    def forward(self, x):
        h = self.pool(self.conv(x)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class CondFeature(nn.Module):
    """Encode condition to /8 feature map: 200->100->50->25"""
    def __init__(self, in_ch: int, base_ch: int = 64):
        super().__init__()
        ch = base_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.out_ch = ch * 4

    def forward(self, cond):
        return self.net(cond)


class ConvDecoder(nn.Module):
    """
    Predict residual/specular layer R_hat in [0,1].
    Up 3 times: 25->50->100->200
    """
    def __init__(self, cond_feat_ch: int, z_dim: int, base_ch: int = 64, out_ch: int = 3):
        super().__init__()
        self.z_proj = nn.Linear(z_dim, cond_feat_ch)

        ch = base_ch
        fused_ch = cond_feat_ch * 2
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(fused_ch, ch * 2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch, out_ch, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, cond_feat, z):
        b, c, hf, wf = cond_feat.shape
        zc = self.z_proj(z).view(b, c, 1, 1).expand(b, c, hf, wf)
        h = torch.cat([cond_feat, zc], dim=1)
        return self.deconv(h)  # R_hat


class HighlightCVAE(nn.Module):
    """
    - MaskUNet predicts M_logits from D (no-highlight)
    - CVAE predicts residual R_hat from (D, M_used, z)
    - Compose A_hat = clamp(D + R_hat * M_used, 0,1)
    - z splits into [z_intensity, z_shape, z_texture] for auxiliary heads

    Training: encoder q(z|cond,target=A)
    Inference: sample z; M from mask net
    """
    def __init__(
        self,
        z_dim=32,
        z_intensity_dim=None,
        z_shape_dim=None,
        z_texture_dim=None,
        base_cvae=64,
        base_mask=32,
        spec_map_hw=25,
    ):
        super().__init__()
        if z_intensity_dim is None and z_shape_dim is None and z_texture_dim is None:
            base = z_dim // 3
            rem = z_dim - base * 3
            z_intensity_dim = base + (1 if rem > 0 else 0)
            z_shape_dim = base + (1 if rem > 1 else 0)
            z_texture_dim = base
        elif z_intensity_dim is None or z_shape_dim is None or z_texture_dim is None:
            raise ValueError("z_intensity_dim, z_shape_dim, z_texture_dim must be all set or all None")
        else:
            z_dim = int(z_intensity_dim) + int(z_shape_dim) + int(z_texture_dim)

        if z_intensity_dim <= 0 or z_shape_dim <= 0 or z_texture_dim <= 0:
            raise ValueError("z_intensity_dim, z_shape_dim, z_texture_dim must be > 0")

        self.z_dim = z_dim
        self.z_intensity_dim = int(z_intensity_dim)
        self.z_shape_dim = int(z_shape_dim)
        self.z_texture_dim = int(z_texture_dim)
        self.spec_map_hw = int(spec_map_hw)
        self.mask_net = MaskUNet(in_ch=3, base=base_mask)

        cond_ch = 3 + 1  # D + mask
        target_ch = 3    # A

        self.encoder = ConvEncoder(in_ch=cond_ch + target_ch, z_dim=z_dim, base_ch=base_cvae)
        self.cond_enc = CondFeature(in_ch=cond_ch, base_ch=base_cvae)
        self.decoder = ConvDecoder(cond_feat_ch=self.cond_enc.out_ch, z_dim=z_dim, base_ch=base_cvae, out_ch=target_ch)
        self.intensity_head = nn.Sequential(
            nn.Linear(self.z_intensity_dim, base_cvae),
            nn.ReLU(inplace=True),
            nn.Linear(base_cvae, 2),
        )
        self.shape_head = nn.Linear(self.z_shape_dim, self.spec_map_hw * self.spec_map_hw)
        self.texture_head = nn.Linear(self.z_texture_dim, self.spec_map_hw * self.spec_map_hw)

    @staticmethod
    def reparameterize(mu, logvar):
        # clamp logvar to avoid exp overflow under AMP
        logvar = torch.clamp(logvar, -20.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, logvar

    def forward(self, D, A_gt=None, z=None, M_used=None):
        """
        D: (B,3,H,W)
        Optional:
          - A_gt: for posterior
          - z: provide z for sampling
          - M_used: externally provided mask (e.g., teacher forcing mix); if None use sigmoid(mask logits)
        Returns:
          R_hat, M_logits, mu, logvar, spec_pred
        """
        M_logits = self.mask_net(D)  # (B,1,H,W) logits
        if M_used is None:
            M_used = torch.sigmoid(M_logits)

        cond = torch.cat([D, M_used], dim=1)
        cond_feat = self.cond_enc(cond)

        if A_gt is not None:
            enc_in = torch.cat([cond, A_gt], dim=1)
            mu, logvar = self.encoder(enc_in)
            z, logvar = self.reparameterize(mu, logvar)
        else:
            mu = torch.zeros(D.size(0), self.z_dim, device=D.device)
            logvar = torch.zeros_like(mu)
            if z is None:
                z = torch.randn_like(mu)

        R_hat = self.decoder(cond_feat, z)
        z_int, z_shape, z_tex = torch.split(
            z, [self.z_intensity_dim, self.z_shape_dim, self.z_texture_dim], dim=1
        )
        intensity_stats = self.intensity_head(z_int)
        shape_logits = self.shape_head(z_shape).view(-1, 1, self.spec_map_hw, self.spec_map_hw)
        texture_map = self.texture_head(z_tex).view(-1, 1, self.spec_map_hw, self.spec_map_hw)
        spec_pred = {
            "intensity": intensity_stats,
            "shape_logits": shape_logits,
            "texture_map": texture_map,
        }
        return R_hat, M_logits, mu, logvar, spec_pred
