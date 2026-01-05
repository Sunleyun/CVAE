import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.shiq_dataset import SHIQDataset
from models.cvae import SpecularCVAE


# =========================
# Config
# =========================
DATA_ROOT = "D:/0_SLYgra/project/data/SHIQ_data_10825"
SAVE_DIR = "./checkpoints"
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Z_INT = 16
Z_SHAPE = 16
Z_TEX = 16

LAMBDA_KL = 1e-3
LAMBDA_INT = 1.0   # intensity supervision weight


os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# Loss functions
# =========================
def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def intensity_loss(z_int, mask_S):
    """
    z_int: [B, z_int_dim]
    mask_S: [B, 1, H, W]
    """
    mask_mean = mask_S.mean(dim=[1, 2, 3])
    z_mean = z_int.mean(dim=1)
    return F.l1_loss(z_mean, mask_mean)


# =========================
# Train
# =========================
def main():
    dataset = SHIQDataset(DATA_ROOT, split="train", size=IMG_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = SpecularCVAE(
        z_int=Z_INT,
        z_shape=Z_SHAPE,
        z_tex=Z_TEX
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {len(dataset)} samples")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for img_A, img_D, mask_T, mask_S in pbar:
            img_A = img_A.to(DEVICE)
            img_D = img_D.to(DEVICE)
            mask_T = mask_T.to(DEVICE)
            mask_S = mask_S.to(DEVICE)

            recon, mu, logvar = model(
                img_spec=img_A,
                img_gt=img_D,
                mask=mask_T
            )

            # =========================
            # Split latent
            # =========================
            z = model.reparameterize(mu, logvar)
            z_int = z[:, :Z_INT]

            # =========================
            # Losses
            # =========================
            recon_loss = F.mse_loss(recon, img_A)
            kl = kl_loss(mu, logvar)
            lint = intensity_loss(z_int, mask_S)

            loss = (
                recon_loss
                + LAMBDA_KL * kl
                + LAMBDA_INT * lint
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{kl.item():.4f}",
                "int": f"{lint.item():.4f}",
            })

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"cvae_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
