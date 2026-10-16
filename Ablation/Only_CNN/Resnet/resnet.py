import os
import random
import datetime
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import CLIPVisionModel
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, average_precision_score,
)

# =========================
# PATHS
# =========================
VIDEO_ROOT = "/content/drive/MyDrive/Videos_Cropped"
EXCEL_PATH = "/content/drive/MyDrive/Manually_Extracted_keyframes.xlsx"
OUTPUT_DIR = "/content/drive/MyDrive/Only_CNN/Resnet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CONFIG
# =========================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
LR = 1e-4
TOP_K = 9
TARGET_FPS = 5
GRAD_CLIP_NORM = 5.0
NUM_WORKERS = 2  # set to 0 on Windows if you hit multiprocessing issues

RUN_TAG = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

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
                except (ValueError, TypeError):
                    pass
    return sorted(keyframes)

# =========================
# DATA PREP
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
# DATASET
# =========================
class DatasetSL(Dataset):
    def __init__(self, video_paths, label_dict):
        self.video_paths = video_paths
        self.label_dict = label_dict

        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        name = os.path.basename(path).replace(".mp4", "")

        if name not in self.label_dict:
            return None

        row = self.label_dict[name]

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0  # sane fallback for corrupted metadata

        interval = max(1, int(round(fps / TARGET_FPS)))

        # Sequential read + subsample instead of repeated CAP_PROP_POS_FRAMES
        # seeks, which are unreliable/slow on many codecs (esp. variable
        # frame rate mp4s).
        frames = []
        frame_idx = 0
        while True:
            ret, f = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (224, 224))
                frames.append(f)
            frame_idx += 1
        cap.release()

        if len(frames) == 0:
            return None

        indices = list(range(0, frame_idx, interval))[:len(frames)]

        vid = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float() / 255.0
        vid = self.norm(vid)

        gt = parse_keyframes_fixed3(row)
        labels = torch.zeros(len(indices))

        for g in gt:
            closest = int(np.argmin([abs(i - g) for i in indices]))
            labels[closest] = 1.0

        return vid, labels


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    max_t = max(v.shape[0] for v, _ in batch)

    vids, lbls, masks = [], [], []

    for v, l in batch:
        pad = max_t - v.shape[0]

        vids.append(F.pad(v, (0, 0, 0, 0, 0, 0, 0, pad)))
        lbls.append(F.pad(l, (0, pad)))
        masks.append(torch.cat([torch.ones(v.shape[0]), torch.zeros(pad)]))

    return torch.stack(vids), torch.stack(lbls), torch.stack(masks)

# =========================
# MODEL (ABLATION: CNN + CLIP fusion only, NO Bi-LSTM)
# =========================
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Fusion layer: CNN (1280 for mobilenet_v2) + CLIP (768) -> 512
        self.fc = nn.Linear(2048, 512)

        # No temporal modeling (no LSTM). Per-frame classification head
        # directly on top of the fused CNN+CLIP embedding.
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        b,t,c,h,w = x.shape
        x = x.view(b*t, c, h, w)

        with torch.no_grad():
            cnn = self.backbone(x)
            cnn = self.pool(cnn).flatten(1)

        x = torch.relu(self.fc(cnn))
        x = x.view(b, t, -1)

        # No LSTM: classify each frame's fused embedding independently.
        out = self.head[1](x)  # Linear applied per-timestep (broadcasts over t)
        return out.squeeze(-1)

# =========================
# Precision@k / Recall@k
# =========================
def precision_recall_at_k(pred_idx, gt_idx, k):
    k = min(k, len(pred_idx))  # can't ask for more than we predicted (short clips)
    pred = set(pred_idx[:k])
    gt = set(gt_idx)

    correct = len(pred & gt)

    precision = correct / k if k else 0.0
    recall = correct / len(gt) if len(gt) else 0.0

    return precision, recall

# =========================
# Mean Absolute Temporal Error
# =========================
def compute_mate(pred_idx, gt_idx):
    if len(pred_idx) == 0 or len(gt_idx) == 0:
        return np.nan

    errors = [np.min(np.abs(np.array(gt_idx) - p)) for p in pred_idx]
    return float(np.mean(errors))

# =========================
# Dynamic Time Warping
# =========================
def compute_dtw(pred_idx, gt_idx):
    if len(pred_idx) == 0 or len(gt_idx) == 0:
        return np.nan

    pred = np.sort(pred_idx)
    gt = np.sort(gt_idx)

    distance, _ = fastdtw(
        pred.reshape(-1, 1),
        gt.reshape(-1, 1),
        dist=euclidean,
    )

    return distance

# =========================
# METRIC NAMES (single source of truth for print/log/return order)
# =========================
METRIC_NAMES = [
    "loss", "accuracy", "precision", "recall", "f1", "weighted_f1",
    "mae", "pooled_ap", "precision@9", "recall@9", "mate", "dtw",
]


def log_metrics(f, header, values):
    print(f"\n{header}")
    f.write(f"{header}\n")
    for name, val in zip(METRIC_NAMES, values):
        line = f"{name}: {val:.4f}"
        print(line)
        f.write(line + "\n")
    f.write("-" * 40 + "\n")

# =========================
# TRAIN
# =========================
def train(model, train_loader, val_loader, output_dir):
    opt = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    crit = nn.BCEWithLogitsLoss(reduction='none')
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = os.path.join(output_dir, f"train_log_{RUN_TAG}.txt")
    best_ckpt_path = os.path.join(output_dir, "model_best.pth")
    best_score = -float("inf")

    with open(log_path, "w") as f:
        for epoch in range(EPOCHS):
            model.train()
            running_loss, n_batches = 0.0, 0

            for batch in train_loader:
                if batch is None:
                    continue

                v, l, m = [x.to(DEVICE) for x in batch]

                opt.zero_grad()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(v)
                    loss = (crit(out, l) * m).sum() / (m.sum() + 1e-8)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    GRAD_CLIP_NORM,
                )
                scaler.step(opt)
                scaler.update()

                running_loss += loss.item()
                n_batches += 1

            train_loss = running_loss / max(n_batches, 1)
            print(f"\nEpoch {epoch + 1} | train_loss: {train_loss:.4f}")

            # ===== evaluation =====
            values = evaluate(model, val_loader)
            log_metrics(f, f"Epoch {epoch + 1} (val)", values)

            # values[7] == pooled_ap: a more informative selection metric
            # than raw val loss for a sparse-positive detection task.
            pooled_ap = values[7]
            if not np.isnan(pooled_ap) and pooled_ap > best_score:
                best_score = pooled_ap
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  -> new best model saved (pooled_ap={pooled_ap:.4f})")

    return best_ckpt_path

# =========================
# EVALUATION
# =========================
def evaluate(model, loader):
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    precision9, recall9, mate_scores, dtw_scores = [], [], [], []

    total_loss = 0.0
    batches = 0

    crit = nn.BCEWithLogitsLoss(reduction='none')

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            v, l, m = [x.to(DEVICE) for x in batch]

            out = model(v)
            loss = (crit(out, l) * m).sum() / (m.sum() + 1e-8)

            total_loss += loss.item()
            batches += 1

            probs = torch.sigmoid(out).cpu()

            for b in range(v.size(0)):
                valid_len = int(m[b].sum().item())
                if valid_len == 0:
                    continue

                # Trim away padding before computing per-video metrics so
                # padded timesteps never leak into precision/recall/AP.
                pb = probs[b, :valid_len].numpy()
                lb = l[b, :valid_len].cpu().numpy()

                k = min(TOP_K, valid_len)
                pred_idx = np.argsort(pb)[-k:]
                pred = np.zeros_like(pb)
                pred[pred_idx] = 1

                gt_idx = np.where(lb == 1)[0]

                p9, r9 = precision_recall_at_k(pred_idx, gt_idx, TOP_K)
                precision9.append(p9)
                recall9.append(r9)
                mate_scores.append(compute_mate(pred_idx, gt_idx))
                dtw_scores.append(compute_dtw(pred_idx, gt_idx))

                all_preds.extend(pred.tolist())
                all_labels.extend(lb.tolist())
                all_probs.extend(pb.tolist())

    if len(all_labels) == 0:
        # Empty loader / nothing evaluable — return NaNs rather than
        # letting sklearn raise on empty arrays.
        return tuple([np.nan] * len(METRIC_NAMES))

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mae = mean_absolute_error(all_labels, all_probs)

    # average_precision_score needs both classes present
    pooled_ap = (
        average_precision_score(all_labels, all_probs)
        if len(set(all_labels)) > 1 else np.nan
    )

    precision9_m = np.nanmean(precision9) if precision9 else np.nan
    recall9_m = np.nanmean(recall9) if recall9 else np.nan
    mate_m = np.nanmean(mate_scores) if mate_scores else np.nan
    dtw_m = np.nanmean(dtw_scores) if dtw_scores else np.nan

    return (
        total_loss / max(batches, 1),
        acc, prec, rec, f1, weighted_f1, mae, pooled_ap,
        precision9_m, recall9_m, mate_m, dtw_m,
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    train_list, val_list, test_list = prepare_data(VIDEO_ROOT)
    label_dict = build_label_dict(EXCEL_PATH)

    train_loader = DataLoader(
        DatasetSL(train_list, label_dict),
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        DatasetSL(val_list, label_dict),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        DatasetSL(test_list, label_dict),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
    )

    model = Model().to(DEVICE)

    best_ckpt_path = train(model, train_loader, val_loader, OUTPUT_DIR)

    # Reload best checkpoint (by val pooled AP) before final reporting,
    # instead of reporting metrics for whatever epoch happened to finish last.
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
        print(f"\nLoaded best checkpoint from {best_ckpt_path}")

    val_values = evaluate(model, val_loader)
    test_values = evaluate(model, test_loader)

    final_results_path = os.path.join(OUTPUT_DIR, f"final_results_{RUN_TAG}_2.txt")
    with open(final_results_path, "w") as f:
        log_metrics(f, "Final Validation Results", val_values)
        log_metrics(f, "Final Test Results", test_values)

    torch.save(
        model.state_dict(),
        os.path.join(OUTPUT_DIR, "resnet.pth"),
    )

    print("Done")