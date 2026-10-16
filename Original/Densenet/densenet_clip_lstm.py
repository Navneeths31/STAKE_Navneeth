import os
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, average_precision_score


# =========================
# PATHS
# =========================
VIDEO_ROOT = "/content/drive/MyDrive/Videos_Cropped"
EXCEL_PATH = "/content/drive/MyDrive/Manually_Extracted_keyframes.xlsx"
OUTPUT_DIR = "/content/drive/MyDrive/Original_STAKE_Code/Densenet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
LR = 1e-4
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
                    keyframes.extend(np.linspace(s+1, e-1, 3, dtype=int))
                except:
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
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
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
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        interval = max(1, int(fps / TARGET_FPS))
        indices = list(range(0, total, interval))

        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, f = cap.read()
            if ret:
                f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (224,224))
            else:
                f = np.zeros((224,224,3), dtype=np.uint8)
            frames.append(f)

        cap.release()

        vid = torch.from_numpy(np.array(frames)).permute(0,3,1,2).float()/255.0
        vid = self.norm(vid)

        gt = parse_keyframes_fixed3(row)
        labels = torch.zeros(len(indices))

        for g in gt:
            if len(indices) > 0:
                closest = np.argmin([abs(i-g) for i in indices])
                labels[closest] = 1.0

        return vid, labels

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    max_t = max(v.shape[0] for v,_ in batch)

    vids, lbls, masks = [], [], []

    for v,l in batch:
        pad = max_t - v.shape[0]

        vids.append(F.pad(v, (0,0,0,0,0,0,0,pad)))
        lbls.append(F.pad(l, (0,pad)))
        masks.append(torch.cat([torch.ones(v.shape[0]), torch.zeros(pad)]))

    return torch.stack(vids), torch.stack(lbls), torch.stack(masks)

# =========================
# MODEL
# =========================
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.clip.parameters():
            p.requires_grad = False

        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.backbone = densenet.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(1024 + 768, 512)
        self.lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=True, num_layers=2)
        self.head = nn.Linear(1024, 1)

    def forward(self, x):
        b,t,c,h,w = x.shape
        x = x.view(b*t, c, h, w)

        with torch.no_grad():
            cnn = self.backbone(x)
            cnn = self.pool(cnn).flatten(1)

            clip = self.clip(pixel_values=x).pooler_output

        fused = torch.cat([cnn, clip], dim=1)

        x = torch.relu(self.fc(fused))
        x = x.view(b, t, -1)

        x, _ = self.lstm(x)
        return self.head(x).squeeze(-1)

# =========================
# Precision@9 / Recall@9
# =========================
def precision_recall_at_k(pred_idx, gt_idx, k):

    pred = set(pred_idx[:k])
    gt = set(gt_idx)

    correct = len(pred & gt)

    precision = correct / k if k else 0
    recall = correct / len(gt) if len(gt) else 0

    return precision, recall


# =========================
# Mean Absolute Temporal Error
# =========================
def compute_mate(pred_idx, gt_idx):

    if len(pred_idx) == 0 or len(gt_idx) == 0:
        return np.nan

    errors = []

    for p in pred_idx:
        nearest = np.min(np.abs(np.array(gt_idx) - p))
        errors.append(nearest)

    return np.mean(errors)


# =========================
# Dynamic Time Warping
# =========================
def compute_dtw(pred_idx, gt_idx):

    if len(pred_idx) == 0 or len(gt_idx) == 0:
        return np.nan

    pred = np.sort(pred_idx)
    gt = np.sort(gt_idx)

    distance, _ = fastdtw(
        pred.reshape(-1,1),
        gt.reshape(-1,1),
        dist=euclidean
    )

    return distance


# =========================
# TRAIN
# =========================
def train(model, train_loader, val_loader, output_dir):

    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(reduction='none')
    scaler = torch.cuda.amp.GradScaler()

    log_path = os.path.join(output_dir, "results.txt")

    with open(log_path, "w") as f:

        for epoch in range(EPOCHS):
            model.train()

            for batch in train_loader:
                if batch is None:
                    continue

                v, l, m = [x.to(DEVICE) for x in batch]

                opt.zero_grad()

                with torch.cuda.amp.autocast():
                    out = model(v)
                    loss = (crit(out, l) * m).sum() / (m.sum() + 1e-8)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            # ===== evaluation =====
            model.eval()
            (loss, acc, prec, rec, f1, weighted_f1, mae, pooled_ap, precision9, recall9, mate, dtw) = evaluate(model, val_loader)

            print(f"\nEpoch {epoch+1}")
            print(f"Loss        : {loss:.4f}")
            print(f"Accuracy    : {acc:.4f}")
            print(f"Precision   : {prec:.4f}")
            print(f"Recall      : {rec:.4f}")
            print(f"F1-score    : {f1:.4f}")
            print(f"Weighted F1   : {weighted_f1:.4f}")
            print(f"MAE         : {mae:.4f}")
            print(f"Pooled AP     : {pooled_ap:.4f}")
            print(f"Precision@9 : {precision9:.4f}")
            print(f"Recall@9    : {recall9:.4f}")
            print(f"MATE        : {mate:.4f}")
            print(f"DTW         : {dtw:.4f}")

            f.write(f"Epoch {epoch+1}\n")
            f.write(f"Loss: {loss:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall: {rec:.4f}\n")
            f.write(f"F1-score: {f1:.4f}\n")
            f.write(f"Weighted F1: {weighted_f1:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"Pooled AP: {pooled_ap:.4f}\n")
            f.write(f"Precision@9: {precision9:.4f}\n")
            f.write(f"Recall@9: {recall9:.4f}\n")
            f.write(f"MATE: {mate:.4f}\n")
            f.write(f"DTW: {dtw:.4f}\n")
            f.write("-"*40 + "\n")

# =========================
# EVALUATION
# =========================
def evaluate(model, loader):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    precision9 = []
    recall9 = []
    mate_scores = []
    dtw_scores = []

    total_loss = 0
    batches = 0

    crit = nn.BCEWithLogitsLoss(reduction='none')

    with torch.no_grad():

        for batch in loader:

            if batch is None:
                continue

            v,l,m = [x.to(DEVICE) for x in batch]

            out = model(v)

            loss = (crit(out,l)*m).sum()/(m.sum()+1e-8)

            total_loss += loss.item()
            batches += 1

            probs = torch.sigmoid(out).cpu()

            for b in range(v.size(0)):

                pb = probs[b].numpy()
                lb = l[b].cpu().numpy()

                pred_idx = np.argsort(pb)[-TOP_K:]
                pred = np.zeros_like(pb)
                pred[pred_idx] = 1

                gt_idx = np.where(lb==1)[0]

                p9,r9 = precision_recall_at_k(pred_idx, gt_idx, TOP_K)

                precision9.append(p9)
                recall9.append(r9)

                mate_scores.append(
                    compute_mate(pred_idx, gt_idx)
                )

                dtw_scores.append(
                    compute_dtw(pred_idx, gt_idx)
                )

                all_preds.extend(pred.tolist())
                all_labels.extend(lb.tolist())
                all_probs.extend(pb.tolist())

    acc = accuracy_score(all_labels, all_preds)

    prec = precision_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    rec = recall_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    mae = mean_absolute_error(
        all_labels,
        all_probs
    )

    precision9 = np.nanmean(precision9)
    recall9 = np.nanmean(recall9)
    mate = np.nanmean(mate_scores)
    dtw = np.nanmean(dtw_scores)

    weighted_f1 = f1_score(
    all_labels,
    all_preds,
    average='weighted',
    zero_division=0
    )

    pooled_ap = average_precision_score(
        all_labels,
        all_probs
    )

    return (
    total_loss/max(batches,1),
    acc,
    prec,
    rec,
    f1,
    weighted_f1,
    mae,
    pooled_ap,
    precision9,
    recall9,
    mate,
    dtw
)

# =========================
# RUN
# =========================
train_list, val_list, test_list = prepare_data(VIDEO_ROOT)
label_dict = build_label_dict(EXCEL_PATH)

train_loader = DataLoader(
    DatasetSL(train_list, label_dict),
    batch_size=1,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    DatasetSL(val_list, label_dict),
    batch_size=1,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    DatasetSL(test_list, label_dict),
    batch_size=1,
    collate_fn=collate_fn
)

model = Model().to(DEVICE)

train(model, train_loader, val_loader, OUTPUT_DIR)

# =========================
# FINAL VALIDATION
# =========================
(
    val_loss,
    val_acc,
    val_prec,
    val_rec,
    val_f1,
    val_weighted_f1,
    val_mae,
    val_pooled_ap,
    val_precision9,
    val_recall9,
    val_mate,
    val_dtw
) = evaluate(model, val_loader)

# =========================
# FINAL TEST
# =========================
(
    test_loss,
    test_acc,
    test_prec,
    test_rec,
    test_f1,
    test_weighted_f1,
    test_mae,
    test_pooled_ap,
    test_precision9,
    test_recall9,
    test_mate,
    test_dtw
) = evaluate(model, test_loader)

print("\n========== Final Validation Results ==========")
print(f"Loss          : {val_loss:.4f}")
print(f"Accuracy      : {val_acc:.4f}")
print(f"Precision     : {val_prec:.4f}")
print(f"Recall        : {val_rec:.4f}")
print(f"F1-score      : {val_f1:.4f}")
print(f"Weighted F1   : {val_weighted_f1:.4f}")
print(f"MAE           : {val_mae:.4f}")
print(f"Pooled AP     : {val_pooled_ap:.4f}")
print(f"Precision@9   : {val_precision9:.4f}")
print(f"Recall@9      : {val_recall9:.4f}")
print(f"MATE          : {val_mate:.4f}")
print(f"DTW Distance  : {val_dtw:.4f}")

print("\n========== Final Test Results ==========")
print(f"Loss          : {test_loss:.4f}")
print(f"Accuracy      : {test_acc:.4f}")
print(f"Precision     : {test_prec:.4f}")
print(f"Recall        : {test_rec:.4f}")
print(f"F1-score      : {test_f1:.4f}")
print(f"Weighted F1   : {test_weighted_f1:.4f}")
print(f"MAE           : {test_mae:.4f}")
print(f"Pooled AP     : {test_pooled_ap:.4f}")
print(f"Precision@9   : {test_precision9:.4f}")
print(f"Recall@9      : {test_recall9:.4f}")
print(f"MATE          : {test_mate:.4f}")
print(f"DTW Distance  : {test_dtw:.4f}")

with open(os.path.join(OUTPUT_DIR, "results.txt"), "a") as f:

    f.write("\n========== Final Validation Results ==========\n")
    f.write(f"Loss: {val_loss:.4f}\n")
    f.write(f"Accuracy: {val_acc:.4f}\n")
    f.write(f"Precision: {val_prec:.4f}\n")
    f.write(f"Recall: {val_rec:.4f}\n")
    f.write(f"F1-score: {val_f1:.4f}\n")
    f.write(f"Weighted F1: {val_weighted_f1:.4f}\n")
    f.write(f"MAE: {val_mae:.4f}\n")
    f.write(f"Pooled AP: {val_pooled_ap:.4f}\n")
    f.write(f"Precision@9: {val_precision9:.4f}\n")
    f.write(f"Recall@9: {val_recall9:.4f}\n")
    f.write(f"MATE: {val_mate:.4f}\n")
    f.write(f"DTW Distance: {val_dtw:.4f}\n")

    f.write("\n========== Final Test Results ==========\n")
    f.write(f"Loss: {test_loss:.4f}\n")
    f.write(f"Accuracy: {test_acc:.4f}\n")
    f.write(f"Precision: {test_prec:.4f}\n")
    f.write(f"Recall: {test_rec:.4f}\n")
    f.write(f"F1-score: {test_f1:.4f}\n")
    f.write(f"Weighted F1: {test_weighted_f1:.4f}\n")
    f.write(f"MAE: {test_mae:.4f}\n")
    f.write(f"Pooled AP: {test_pooled_ap:.4f}\n")
    f.write(f"Precision@9: {test_precision9:.4f}\n")
    f.write(f"Recall@9: {test_recall9:.4f}\n")
    f.write(f"MATE: {test_mate:.4f}\n")
    f.write(f"DTW Distance: {test_dtw:.4f}\n")

torch.save(
    model.state_dict(),
    os.path.join(OUTPUT_DIR, "densenet_clip_lstm.pth")
)

print("Done")