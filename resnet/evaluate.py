import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.functional as TF
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, cohen_kappa_score
)
from tqdm import tqdm

from dataset import APTOSDataset

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
APTOS_CSV       = "/home/s2759545/datasets/aptos/train.csv"
APTOS_IMAGE_DIR = "/home/s2759545/datasets/aptos"

IDRID_TRAIN_CSV    = "/disk/scratch/s2759545/datasets/idrid_dataset/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
IDRID_TEST_CSV     = "/disk/scratch/s2759545/datasets/idrid_dataset/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
IDRID_TRAIN_IMAGES = "/disk/scratch/s2759545/datasets/idrid_dataset/1. Original Images/a. Training Set"
IDRID_TEST_IMAGES  = "/disk/scratch/s2759545/datasets/idrid_dataset/1. Original Images/b. Testing Set"

BEST_MODEL_PATH = "/home/s2759545/checkpoints/dr_resnet50_best_model.pth"
OUTPUT_DIR      = "/home/s2759545/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE  = 32
RANDOM_SEED = 78

# ─────────────────────────────────────────────
# SQUARE PAD
# ─────────────────────────────────────────────
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = [hp, vp, max_wh - w - hp, max_wh - h - vp]
        return TF.pad(image, padding, fill=0, padding_mode='constant')

transform_val = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def make_thresholds(label):
    return [1 if label >= i else 0 for i in range(1, 5)]

def decode_ordinal_predictions(logits):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()
    return preds.sum(dim=1)

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
def generate_gradcam(model, input_batch, target_layer):
    model.eval()
    gradients  = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_b = target_layer.register_full_backward_hook(backward_hook)
    handle_f = target_layer.register_forward_hook(forward_hook)

    logits = model(input_batch)
    score  = logits[:, logits.argmax(dim=1)].sum()

    model.zero_grad()
    score.backward()

    grads  = gradients[0].cpu().data.numpy()
    f_maps = activations[0].cpu().data.numpy()
    weights = np.mean(grads, axis=(2, 3))[0]

    cam = np.zeros(f_maps.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * f_maps[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_b.remove()
    handle_f.remove()

    return cam

def plot_gradcam_results(model, dataset, indices, device, save_path):
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 5 * len(indices)))

    for i, idx in enumerate(indices):
        img_tensor, label_vec = dataset[idx]
        input_batch = img_tensor.unsqueeze(0).to(device)

        heatmap = generate_gradcam(model, input_batch, model.layer4)

        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"True Stage: {int(label_vec.sum())}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img)
        axes[i, 1].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[i, 1].set_title("Grad-CAM (Attention)")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Grad-CAM plot saved → {save_path}")

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def get_full_metrics(labels, preds):
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4])
    stats = []
    for stage in range(5):
        tp             = cm[stage, stage]
        actual_total   = cm[stage, :].sum()
        predicted_total = cm[:, stage].sum()
        acc  = (tp / actual_total)   if actual_total   > 0 else 0
        prec = (tp / predicted_total) if predicted_total > 0 else 0
        stats.append({"Acc": acc, "Sens": acc, "Prec": prec})
    return stats

def print_metrics(all_labels, all_preds, dataset_name):
    acc       = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    qwk       = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    print(f"\n{'='*50}")
    print(f"  {dataset_name} — Overall Metrics")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  QWK      : {qwk:.4f}")
    print(f"{'='*50}\n")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ──
    model    = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model    = model.to(device)
    model.eval()
    print(f"Model loaded from {BEST_MODEL_PATH}")

    # ────────────────────────────────
    # APTOS Validation Set Evaluation
    # ────────────────────────────────
    df = pd.read_csv(APTOS_CSV)
    df = df[df['id_code'].apply(
        lambda x: os.path.exists(os.path.join(APTOS_IMAGE_DIR, x + ".png"))
    )].reset_index(drop=True)
    df['thresholds'] = df['diagnosis'].apply(make_thresholds)

    _, val_df = train_test_split(
        df, test_size=0.2, stratify=df['diagnosis'], random_state=RANDOM_SEED
    )

    val_dataset = APTOSDataset(val_df, APTOS_IMAGE_DIR, transform_val)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="APTOS Validation"):
            images = images.to(device)
            logits = model(images)
            preds  = decode_ordinal_predictions(logits).cpu().numpy()
            true   = labels.numpy().sum(axis=1)
            all_preds.extend(preds)
            all_labels.extend(true)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print_metrics(all_labels, all_preds, "APTOS (In-Domain Validation)")

    # ── APTOS Confusion Matrix ──
    cm_aptos      = confusion_matrix(all_labels, all_preds)
    cm_aptos_norm = cm_aptos.astype('float') / cm_aptos.sum(axis=1)[:, np.newaxis]

    # ────────────────────────────────
    # IDRiD Cross-Dataset Evaluation
    # ────────────────────────────────
    idrid_test_df = pd.read_csv(IDRID_TEST_CSV)
    idrid_test_df.columns = idrid_test_df.columns.str.strip()
    idrid_test_df = idrid_test_df[['Image name', 'Retinopathy grade']]
    idrid_test_df = idrid_test_df.rename(columns={
        'Image name': 'id_code', 'Retinopathy grade': 'diagnosis'
    })
    idrid_test_df['id_code']    = idrid_test_df['id_code'].astype(str).str.strip()
    idrid_test_df['thresholds'] = idrid_test_df['diagnosis'].apply(make_thresholds)

    idrid_test_dataset = APTOSDataset(idrid_test_df, IDRID_TEST_IMAGES, transform_val)
    idrid_loader       = DataLoader(idrid_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"IDRiD test set: {len(idrid_test_dataset)} images")

    idrid_preds, idrid_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(idrid_loader, desc="IDRiD Testing"):
            images = images.to(device)
            logits = model(images)
            preds  = decode_ordinal_predictions(logits).cpu().numpy()
            true   = labels.numpy().sum(axis=1)
            idrid_preds.extend(preds)
            idrid_labels.extend(true)

    idrid_preds  = np.array(idrid_preds)
    idrid_labels = np.array(idrid_labels)

    print_metrics(idrid_labels, idrid_preds, "IDRiD (Cross-Domain)")

    # ── Comparison Table ──
    aptos_m  = get_full_metrics(all_labels, all_preds)
    idrid_m  = get_full_metrics(idrid_labels, idrid_preds)
    stages   = ['0 (Normal)', '1 (Mild)', '2 (Moderate)', '3 (Severe)', '4 (Prolif)']

    rows = []
    for i in range(5):
        rows.append({
            'DR Stage':    stages[i],
            'APTOS Acc':   aptos_m[i]['Acc'],
            'APTOS Sens':  aptos_m[i]['Sens'],
            'APTOS Prec':  aptos_m[i]['Prec'],
            'IDRiD Acc':   idrid_m[i]['Acc'],
            'IDRiD Sens':  idrid_m[i]['Sens'],
            'IDRiD Prec':  idrid_m[i]['Prec'],
        })

    master_df = pd.DataFrame(rows)
    aptos_qwk = cohen_kappa_score(all_labels,   all_preds,   weights='quadratic')
    idrid_qwk = cohen_kappa_score(idrid_labels, idrid_preds, weights='quadratic')

    print("=" * 115)
    print(f"{'':15} | {'--- APTOS (In-Domain) ---':^30} | {'--- IDRiD (Cross-Domain) ---':^30}")
    print("=" * 115)
    print(master_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
    print("-" * 115)
    print(f"OVERALL QWK | APTOS: {aptos_qwk:.4f} | IDRiD: {idrid_qwk:.4f} | Retention: {(idrid_qwk/aptos_qwk)*100:.2f}%")
    print("=" * 115)

    master_df.to_csv(os.path.join(OUTPUT_DIR, "comparison_table.csv"), index=False)
    print(f"Comparison table saved → {OUTPUT_DIR}/comparison_table.csv")

    # ── Dual Confusion Matrix Plot ──
    cm_idrid      = confusion_matrix(idrid_labels, idrid_preds)
    cm_idrid_norm = cm_idrid.astype('float') / cm_idrid.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm_aptos_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0],
                xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    axes[0].set_title('APTOS Validation (In-Domain)\nNormalized Confusion Matrix')
    axes[0].set_xlabel('Predicted Stage')
    axes[0].set_ylabel('True Stage')

    sns.heatmap(cm_idrid_norm, annot=True, fmt='.2f', cmap='Reds', ax=axes[1],
                xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    axes[1].set_title('IDRiD Evaluation (Cross-Domain)\nNormalized Confusion Matrix')
    axes[1].set_xlabel('Predicted Stage')
    axes[1].set_ylabel('True Stage')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150)
    print(f"Confusion matrix plot saved → {OUTPUT_DIR}/confusion_matrices.png")

    # ── Grad-CAM on IDRiD samples ──
    sample_indices = np.random.choice(len(idrid_test_dataset), 3, replace=False)
    plot_gradcam_results(
        model, idrid_test_dataset, sample_indices, device,
        save_path=os.path.join(OUTPUT_DIR, "gradcam_idrid.png")
    )


if __name__ == "__main__":
    main()
