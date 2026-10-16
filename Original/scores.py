import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import CLIPVisionModel
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# =========================
# PATHS  (update these to match your environment)
# =========================
VIDEO_ROOT = r"C:\Users\navne\Pictures\Kathakali\Videos_Cropped"
EXCEL_PATH = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Automatic keyframe extraction\Manually_Extracted_keyframes.xlsx"
OUTPUT_DIR = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Automatic keyframe extraction\Baseline\baseline_best_models"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "rl_dsn_best.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 9
TARGET_FPS = 5

# =========================
# KEYFRAME PARSER
# =========================
def parse_keyframes_fixed3(row):
    keyframes = []
    for col in ['Keyframes_1', 'Keyframes_2', 'Keyframes_3']:
        if col in row and pd.notna(row[col]):
            val = str(row[col])
            if '-' in val:
                try:
                    s, e = map(int, val.split('-'))
                    keyframes.extend(np.linspace(s + 1, e - 1, 3, dtype=int))
                except:
                    pass
    return sorted(keyframes)

# =========================
# DATA SPLIT (must match training script exactly so test_list is identical)
# =========================
def prepare_data(video_root):
    train_list, val_list, test_list = [], [], []

    for word in sorted(os.listdir(video_root)):
        word_path = os.path.join(video_root, word)
        if not os.path.isdir(word_path):
            continue

        videos = sorted([v for v in os.listdir(word_path) if v.endswith(".mp4")])

        for i, vid in enumerate(videos):
            full_path = os.path.join(word_path, vid)

            if i < 7:
                train_list.append(full_path)
            elif i == 7:
                val_list.append(full_path)
            else:
                test_list.append(full_path)

    return train_list, val_list, test_list

def build_label_dict(excel_path):
    df = pd.read_excel(excel_path)
    return {str(r['Video']).strip(): r for _, r in df.iterrows()}

# =========================
# DATASET (returns video name alongside frames/labels)
# =========================
class DatasetSLNamed(Dataset):
    def __init__(self, video_paths):
        self.video_paths = video_paths

        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        name = os.path.basename(path).replace(".mp4", "")

        # Keep only test videos
        if not (name.endswith("8") or name.endswith("9")):
            return None

        FEATURE_DIR = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Automatic keyframe extraction\Baseline\.cache_C_Users_navne_Pictures_Kathakali_Videos_Cropped"

        feature_path = os.path.join(FEATURE_DIR, name + ".npz")
        data = np.load(feature_path)

        features = data["resnet"]
        labels = data["labels"]

        return features, torch.from_numpy(labels).float(), name

def collate_fn_named(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    max_t = max(v.shape[0] for v, _, _ in batch)

    feats, lbls, masks, names = [], [], [], []

    for v, l, n in batch:
        v = torch.from_numpy(v).float()

        pad = max_t - v.shape[0]

        feats.append(F.pad(v, (0,0,0,pad)))
        lbls.append(F.pad(l, (0, pad)))
        masks.append(torch.cat([torch.ones(v.shape[0]), torch.zeros(pad)]))
        names.append(n)

    return torch.stack(feats), torch.stack(lbls), torch.stack(masks), names

# =========================
# MODEL (must match training architecture exactly)
# =========================
class DSN(nn.Module):
    def __init__(self, in_dim=2048, hidden=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, 512)
        self.lstm = nn.LSTM(
            512,
            hidden,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        h = torch.relu(self.proj(x))
        h, _ = self.lstm(h)
        return torch.sigmoid(self.fc(h)).squeeze(-1)

# =========================
# METRIC HELPERS
# =========================
def precision_recall_at_k(pred_idx, gt_idx, k):
    pred = set(pred_idx[:k])
    gt = set(gt_idx)

    correct = len(pred & gt)

    precision = correct / k if k else 0
    recall = correct / len(gt) if len(gt) else 0

    return precision, recall

def compute_mate(pred_idx, gt_idx):
    if len(pred_idx) == 0 or len(gt_idx) == 0:
        return np.nan

    errors = []
    for p in pred_idx:
        nearest = np.min(np.abs(np.array(gt_idx) - p))
        errors.append(nearest)

    return np.mean(errors)

def compute_dtw(pred_idx, gt_idx):
    if len(pred_idx) == 0 or len(gt_idx) == 0:
        return np.nan

    pred = np.sort(pred_idx)
    gt = np.sort(gt_idx)

    distance, _ = fastdtw(
        pred.reshape(-1, 1),
        gt.reshape(-1, 1),
        dist=euclidean
    )

    return distance

# =========================
# PER-VIDEO EVALUATION
# =========================
def evaluate_per_video(model, loader):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            v, l, m, names = batch
            v, l, m = v.to(DEVICE), l.to(DEVICE), m.to(DEVICE)

            probs = model(v).cpu().numpy()

            for i in range(v.size(0)):
                n = int(m[i].sum().item())

                pb = probs[i][:n]
                lb = l[i].cpu().numpy()[:n]

                pred_idx = np.argsort(pb)[-TOP_K:]
                gt_idx = np.where(lb == 1)[0]

                p9, r9 = precision_recall_at_k(pred_idx, gt_idx, TOP_K)
                mate = compute_mate(pred_idx, gt_idx)
                dtw = compute_dtw(pred_idx, gt_idx)

                pred = np.zeros_like(pb)
                pred[pred_idx] = 1

                tp = np.sum((pred == 1) & (lb == 1))
                fp = np.sum((pred == 1) & (lb == 0))
                fn = np.sum((pred == 0) & (lb == 1))

                f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

                rows.append({
                    "video": names[i],
                    "f1": f1,
                    "precision9": p9,
                    "recall9": r9,
                    "mate": mate,
                    "dtw": dtw
                })

    return pd.DataFrame(rows)

# ---- 7.9 CNN (per-frame convolutional keyframe classifier) ----
class FrameCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.body = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128),
                                  nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.body(x).flatten(1)).squeeze(-1)


# =========================
# RUN
# =========================
if __name__ == "__main__":

    # 1. Rebuild the exact same split used during training
    train_list, val_list, test_list = prepare_data(VIDEO_ROOT)
    #label_dict = build_label_dict(EXCEL_PATH)

    print(f"Test set size: {len(test_list)} videos")

    # 2. Build test loader with video names attached
    test_loader_named = DataLoader(
        DatasetSLNamed(test_list),
        batch_size=1,
        collate_fn=collate_fn_named
    )

    # 3. Load model architecture + trained weights
    model = DSN().to(DEVICE)

    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    print(f"Loaded model weights from {BEST_MODEL_PATH}")

    # 4. Run per-video evaluation on test set
    df_test_scores = evaluate_per_video(model, test_loader_named)

    # 5. Save results
    save_path = os.path.join(OUTPUT_DIR, "rl_dsn_test_per_video_scores.csv")
    df_test_scores.to_csv(save_path, index=False)

    print("\n===== Per-video score summary (test set) =====")
    print(df_test_scores.describe())
    print(f"\nSaved per-video scores to: {save_path}")

    print("\n===== Mean metrics (test set) =====")
    print(f"F1          : {df_test_scores['f1'].mean():.4f}")
    print(f"Precision@9 : {df_test_scores['precision9'].mean():.4f}")
    print(f"Recall@9    : {df_test_scores['recall9'].mean():.4f}")
    print(f"MATE        : {df_test_scores['mate'].mean():.4f}")
    print(f"DTW         : {df_test_scores['dtw'].mean():.4f}")