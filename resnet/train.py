import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.functional as TF

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import APTOSDataset

# ─────────────────────────────────────────────
# PATHS — update these if your layout changes
# ─────────────────────────────────────────────
APTOS_CSV        = "/home/s2759545/datasets/aptos/train.csv"
APTOS_IMAGE_DIR  = "/home/s2759545/datasets/aptos"

CHECKPOINT_PATH  = "/home/s2759545/checkpoints/dr_resnet50_checkpoint.pth"
BEST_MODEL_PATH  = "/home/s2759545/checkpoints/dr_resnet50_best_model.pth"

os.makedirs("/home/s2759545/checkpoints", exist_ok=True)

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────
NUM_EPOCHS  = 10
BATCH_SIZE  = 32
LR          = 1e-4
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 78

# ─────────────────────────────────────────────
# SQUARE PAD (PIL-compatible)
# ─────────────────────────────────────────────
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = [hp, vp, max_wh - w - hp, max_wh - h - vp]
        return TF.pad(image, padding, fill=0, padding_mode='constant')

# ─────────────────────────────────────────────
# ORDINAL LABEL HELPERS
# ─────────────────────────────────────────────
def make_thresholds(label):
    return [1 if label >= i else 0 for i in range(1, 5)]

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
transform_train = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load & filter APTOS CSV ──
    df = pd.read_csv(APTOS_CSV)

    # Filter out any rows where the image file is missing
    df = df[df['id_code'].apply(
        lambda x: os.path.exists(os.path.join(APTOS_IMAGE_DIR, x + ".png"))
    )].reset_index(drop=True)

    print(f"Dataset size after filtering: {df.shape[0]}")

    # Add ordinal thresholds
    df['thresholds'] = df['diagnosis'].apply(make_thresholds)

    # ── Train / Val split ──
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['diagnosis'],
        random_state=RANDOM_SEED
    )

    print(f"Train: {train_df.shape[0]}  Val: {val_df.shape[0]}")

    # ── Datasets & Loaders ──
    train_dataset = APTOSDataset(train_df, APTOS_IMAGE_DIR, transform_train)
    val_dataset   = APTOSDataset(val_df,   APTOS_IMAGE_DIR, transform_val)

    # Weighted sampler to handle class imbalance
    class_counts  = train_df['diagnosis'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = torch.tensor(
        train_df['diagnosis'].map(
            {i: class_weights[i] for i in range(len(class_weights))}
        ).values,
        dtype=torch.float
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ──
    weights = ResNet50_Weights.DEFAULT
    model   = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model   = model.to(device)

    # ── Loss: stage-weighted BCEWithLogitsLoss ──
    stage_counts  = train_df['diagnosis'].value_counts().sort_index()
    stage_weights = 1.0 / stage_counts
    stage_weights = stage_weights / stage_weights.sum()

    threshold_weights = torch.tensor([
        stage_weights[1:].sum(),
        stage_weights[2:].sum(),
        stage_weights[3:].sum(),
        stage_weights[4]
    ]).float().to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=threshold_weights)

    # ── Optimizer & Scheduler ──
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ── Resume from checkpoint if it exists ──
    start_epoch   = 0
    best_val_loss = float("inf")
    train_losses  = []
    val_losses    = []

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch   = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        train_losses  = checkpoint.get('train_losses', [])
        val_losses    = checkpoint.get('val_losses', [])
        print(f"Resuming from epoch {start_epoch}")

    # ── Training Loop ──
    for epoch in range(start_epoch, NUM_EPOCHS):

        # --- Train ---
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images = images.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # --- Validate ---
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"\nEpoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        scheduler.step(epoch_val_loss)

        # --- Save checkpoint ---
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss':           epoch_train_loss,
            'val_loss':             epoch_val_loss,
            'best_val_loss':        best_val_loss,
            'train_losses':         train_losses,
            'val_losses':           val_losses,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved → {CHECKPOINT_PATH}")

        # --- Save best model ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ Best model updated → {BEST_MODEL_PATH}")

    # ── Loss Plot ──
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Val Loss',   marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/s2759545/checkpoints/loss_plot.png", dpi=150)
    print("Loss plot saved → /home/s2759545/checkpoints/loss_plot.png")


if __name__ == "__main__":
    main()
