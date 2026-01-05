import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class SHIQDataset(Dataset):
    def __init__(self, root, split='train', size=256):
        self.root = os.path.join(root, split)
        self.size = size

        self.ids = sorted(list(set(
            f.split('_')[0] for f in os.listdir(self.root) if f.endswith('_A.png')
        )))

    def __len__(self):
        return len(self.ids)

    def read_img(self, path, gray=False):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if not gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        return img

    def __getitem__(self, idx):
        obj_id = self.ids[idx]

        img_A = self.read_img(os.path.join(self.root, f"{obj_id}_A.png"))
        img_D = self.read_img(os.path.join(self.root, f"{obj_id}_D.png"))
        mask_T = self.read_img(os.path.join(self.root, f"{obj_id}_T.png"), gray=True)
        mask_S = self.read_img(os.path.join(self.root, f"{obj_id}_S.png"), gray=True)

        # [H, W] → [1, H, W]
        mask_T = np.expand_dims(mask_T, axis=0)
        mask_S = np.expand_dims(mask_S, axis=0)

        return (
            torch.from_numpy(img_A).permute(2, 0, 1),
            torch.from_numpy(img_D).permute(2, 0, 1),
            torch.from_numpy(mask_T),
            torch.from_numpy(mask_S),
        )
