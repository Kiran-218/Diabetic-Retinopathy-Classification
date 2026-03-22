import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class APTOSDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['id_code']

        # Check for .png (APTOS) first, then .jpg (IDRiD)
        img_path = os.path.join(self.img_dir, f"{img_id}.png")

        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        # If still not found, check if the ID already has an extension
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, img_id)

        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find image {img_id} as .png or .jpg in {self.img_dir}"
            )

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.df.iloc[idx]['thresholds'], dtype=torch.float)

        return img, label
