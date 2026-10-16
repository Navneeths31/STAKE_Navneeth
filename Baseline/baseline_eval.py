"""
baseline_eval.py — Baseline keyframe-extraction evaluation (STAKE v10 protocol)
==============================================================================

Evaluates a suite of keyframe-extraction baselines against the *exact* ground
truth / test-split / sampling protocol used in ``model_trainer_v10.py`` so the
numbers are directly comparable to STAKE v10.

Ground truth  (identical to model_trainer_v10.parse_keyframes)
--------------------------------------------------------------
Each row of ``Manually_Extracted_keyframes.csv`` has three frame RANGES
(Keyframes_1/2/3, e.g. ``1-24``).  For every range ``start-end`` we take
``np.linspace(start+1, end-1, num=3)`` → 3 evenly-spaced interior frames.
3 ranges × 3 = **9 ground-truth keyframes** (``n_gt``).  Those original-frame
indices are mapped onto the 10-fps sampled timeline; each nearest sampled index
±2 (soft window) becomes a positive label.  Everything else is a "transition"
(negative) frame.

Evaluation protocol  (identical to model_trainer_v10.evaluate)
-------------------------------------------------------------
* Split: dropna(Video) → train_test_split(0.3, seed 42) → temp,
          then train_test_split(temp, 0.5, seed 42) → val / test.
* Sampling: every ``int(fps/10)`` frames, resize 224², ImageNet-normalize.
* Dynamic top-k: each method selects exactly ``k = n_gt`` frames.

Metrics
-------
* F1                — pooled over all frames (matches v10).
* PR-AUC (AP / mAP) — average precision from each method's continuous scores
                       (per-clip mean = mAP, plus pooled AP).
* Weighted-F1       — pooled F1 with per-frame sample weights
                       (keyframe = 10.0, transition = 0.1).

Baselines
---------
  1. Katna (pip, isolated env)          6. Lightweight SAM (MobileSAM)
  2. CLIP (ViT-B/32 + k-medoids)        7. Cosine + SSIM
  3. RL — DR-DSN (unsupervised)         8. Uniform+threshold+vote+SVM/CNN
  4. K-medoids (ResNet features)        9. CNN (per-frame conv classifier)
  5. Nearest-neighbour clustering      10. STAKE v6 (keyframe-extractor-v6-best.pth)

Trainable baselines (3, 8, 9) and the supervised STAKE v6 checkpoint are trained
on / were trained on the TRAIN split; the "test" scope (disjoint from train) is
the unbiased comparison.  The "all" scope re-uses every existing video and
therefore leaks train videos into supervised methods — see the printed warning.

Run
---
    uv run python baseline_eval.py                 # both scopes, all methods
    uv run python baseline_eval.py --scope test    # only the v10 test split
    uv run python baseline_eval.py --methods clip kmedoids
    uv run python baseline_eval.py --rebuild-cache
"""

import os
import re
import sys
import json
import time
import pickle
import argparse
import warnings
import subprocess

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import CLIPVisionModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, average_precision_score, precision_score, recall_score, accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from skimage.metrics import structural_similarity as ssim_fn
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252 → force UTF-8 so log glyphs don't crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Quiet the very verbose transformers weight-load reports.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
    _hf_logging.disable_progress_bar()
except Exception:
    pass

# ============================================================
# 1. CONFIGURATION
# ============================================================
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GT_PATH    = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Automatic keyframe extraction\Manually_Extracted_keyframes.xlsx"   # ground-truth table: .xlsx or .csv
CSV_PATH   = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Automatic keyframe extraction\Manually_Extracted_keyframes.xlsx"
VIDEO_DIR  = r"C:\Users\navne\Pictures\Kathakali\Videos_Cropped"      # v10 config said Videos_Cropped/train (absent here)

TARGET_FPS   = 5               # v10 subsampling rate
LABEL_RADIUS = 2                   # v10 soft-window radius (in sampled-index units)
SEED         = 42

W_KEYFRAME    = 10.0               # weighted-F1 / weighted-loss: keyframe weight
W_TRANSITION  = 0.1               # weighted-F1 / weighted-loss: transition weight
FOCAL_ALPHA   = 0.75              # focal loss (v10): weight on positive class
FOCAL_GAMMA   = 2.0               # focal loss (v10): focusing exponent

CACHE_DIR   = ".cache_baseline"    # overridden per --video-dir in main()
RESULTS_DIR = "baseline_results"   # overridden by --out-dir
MODELS_DIR  = "baseline_best_models"  # where each trainable baseline's best model is saved

CLIP_NAME   = "openai/clip-vit-base-patch32"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

ALL_METHODS = [
    "katna", "clip", "rl", "kmedoids", "nn_cluster",
    "sam", "cosine_ssim", "uniform_svmcnn", "cnn",
]

METHOD_LABELS = {
    "katna":          "Katna (pip)",
    "clip":           "CLIP (ViT-B/32 + k-medoids)",
    "rl":             "RL / DR-DSN [3,10]",
    "kmedoids":       "K-medoids [11]",
    "nn_cluster":     "Nearest-neighbour clustering [17]",
    "sam":            "Lightweight SAM (MobileSAM) [18]",
    "cosine_ssim":    "Cosine + SSIM [20]",
    "uniform_svmcnn": "Uniform+threshold+vote+SVM/CNN [21]",
    "cnn":            "CNN [15]",
}


def log(msg):
    print(msg, flush=True)


# ============================================================
# 2. DATA PREPARATION  (exact copy of model_trainer_v10 logic)
# ============================================================
def parse_keyframes(row):
    """Return the sorted list of the (≤9) GT keyframe indices — v10 logic."""
    keyframes = []
    for col in ["Keyframes_1", "Keyframes_2", "Keyframes_3"]:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if "-" in val:
                try:
                    start, end = map(int, val.split("-"))
                    selected = np.linspace(start + 1, end - 1, num=3, dtype=int)
                    keyframes.extend(selected.tolist())
                except Exception:
                    continue
    return sorted(set(keyframes))


def _read_csv_robust(csv_path):
    """
    Read the CSV trying utf-8 first, then falling back through common
    Excel/Windows export encodings. Avoids UnicodeDecodeError crashes on
    files that were saved as cp1252/latin-1 (typical for Excel-exported CSVs
    that contain curly quotes, em-dashes, or other non-ASCII punctuation).
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            if enc != "utf-8":
                log(f"  (note: {csv_path} was not valid UTF-8 — read using '{enc}')")
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # last resort: replace undecodable bytes instead of crashing
    log(f"  ! WARNING: could not decode {csv_path} with {encodings} — "
        f"reading utf-8 with errors='replace' ({last_err})")
    return pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")


def _read_table(data_path):
    """
    Load the ground-truth table regardless of whether it's an Excel workbook
    (.xlsx/.xls/.xlsm) or a CSV/TSV file. Excel files have no text-encoding
    concept (they're stored as XML/binary), so they sidestep the
    UnicodeDecodeError issue entirely — only the CSV path needs the encoding
    fallback logic.
    """
    ext = os.path.splitext(data_path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        engine = "openpyxl" if ext in (".xlsx", ".xlsm") else None
        return pd.read_excel(data_path, engine=engine)
    return _read_csv_robust(data_path)


def prepare_data(csv_path, val_frac=0.15, test_frac=0.15,
                 split_existing=False, video_dir=None):
    """
    Shuffle + split the ground-truth table (CSV or Excel) into train/val/test
    with the given fractions (defaults 0.15/0.15 → the v10 70/15/15 split;
    use 0.10/0.20 for 70/10/20).

    split_existing=True restricts the split population to videos that actually
    exist under video_dir (so train/val/test are all drawn from the videos on
    disk) — used for the Videos_Cropped run.
    """
    df = _read_table(csv_path)
    df = df.dropna(subset=["Video"])
    df = df[df["Video"].astype(str).str.lower() != "nan"]
    if split_existing and video_dir:
        keep = df["Video"].astype(str).str.strip().apply(
            lambda n: os.path.exists(video_path_for(n, video_dir)))
        df = df[keep]

    temp_frac = val_frac + test_frac
    train_df, temp_df = train_test_split(
        df, test_size=temp_frac, random_state=SEED, shuffle=True)
    rel_test = test_frac / temp_frac if temp_frac > 0 else 0.5
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, random_state=SEED, shuffle=True)
    return train_df, val_df, test_df


def video_path_for(video_name, video_dir=VIDEO_DIR):
    """
    Resolve <video_dir>/<word>/<name>.mp4.  The "word" (folder) is everything
    before the trailing _<number>, so multi-underscore words resolve correctly
    (e.g. Nee_Shringaaram_2 → folder 'Nee_Shringaaram', not 'Nee' as v10 assumed).
    """
    name = str(video_name).strip()
    folders = []
    if "_" in name:
        folders.append(name.rsplit("_", 1)[0])   # word may contain underscores
        folders.append(name.split("_")[0])        # v10 behaviour (fallback)
    folders.append(name)                          # folder == name (fallback)
    for folder in folders:
        for ext in (".mp4", ".MP4"):
            p = os.path.join(video_dir, folder, f"{name}{ext}")
            if os.path.exists(p):
                return p
    return os.path.join(video_dir, folders[0], f"{name}.mp4")   # non-existent


# ============================================================
# 3. FRAME SAMPLING + SOFT-WINDOW LABELS  (v10 protocol)
# ============================================================
def decode_video(video_path):
    """
    Sample frames exactly like model_trainer_v10.SignLanguageDataset.
    Returns (frames_rgb_uint8 [T,224,224,3], indices [list], total, fps) or None.
    """
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0 or fps <= 0:
        cap.release()
        return None

    interval = max(1, int(fps / TARGET_FPS))
    indices  = list(range(0, total, interval))
    wanted   = set(indices)

    # Sequential decode (no per-frame seeking) → same frames, far faster.
    grabbed = {}
    pos = 0
    while pos <= indices[-1]:
        ret, frame = cap.read()
        if not ret:
            break
        if pos in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            grabbed[pos] = cv2.resize(frame, (224, 224))
        pos += 1
    cap.release()

    frames = [grabbed.get(i, np.zeros((224, 224, 3), dtype=np.uint8)) for i in indices]
    if len(frames) == 0:
        return None
    return np.asarray(frames, dtype=np.uint8), indices, total, fps


def build_labels(indices, gt_frames, radius=LABEL_RADIUS):
    """
    Soft-window labels (v10): nearest sampled index of each GT ±radius = 1.
    Also returns gt_sampled = the GT keyframe positions on the sampled timeline
    (the nearest sampled index of each GT frame) — used for the DTW temporal metric.
    """
    labels = np.zeros(len(indices), dtype=np.float32)
    n_gt = 0
    gt_sampled = []
    idx_arr = np.asarray(indices)
    for gt in gt_frames:
        closest = int(np.argmin(np.abs(idx_arr - gt)))
        lo = max(0, closest - radius)
        hi = min(len(indices) - 1, closest + radius)
        labels[lo:hi + 1] = 1.0
        gt_sampled.append(closest)
        n_gt += 1
    return labels, n_gt, sorted(set(gt_sampled))


# ============================================================
# 4. FEATURE EXTRACTION + CACHE
# ============================================================
_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _to_224_tensor(frames_uint8):
    """(T,224,224,3) uint8 -> (T,3,224,224) ImageNet-normalized float tensor."""
    t = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float() / 255.0
    return _norm(t)


class SharedExtractors:
    """ResNet50 (2048-d pooled) + CLIP ViT-B/32 (768-d pooler)."""
    def __init__(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE).eval()
        self.clip = CLIPVisionModel.from_pretrained(CLIP_NAME).to(DEVICE).eval()

    @torch.no_grad()
    def extract(self, frames_uint8, batch=32):
        x = _to_224_tensor(frames_uint8)
        res, clip = [], []
        for i in range(0, x.shape[0], batch):
            chunk = x[i:i + batch].to(DEVICE)
            res.append(self.resnet(chunk).flatten(1).cpu())
            clip.append(self.clip(pixel_values=chunk).pooler_output.cpu())
        return torch.cat(res).numpy(), torch.cat(clip).numpy()


def cache_path(video_id):
    return os.path.join(CACHE_DIR, f"{video_id}.npz")


def build_cache_for(video_id, video_name, gt_frames, extractors, video_dir):
    """Decode + featurize one video and persist an npz. Returns True on success."""
    dec = decode_video(video_path_for(video_name, video_dir))
    if dec is None:
        return False
    frames224, indices, total, fps = dec
    labels, n_gt, gt_sampled = build_labels(indices, gt_frames)

    resnet_feat, clip_feat = extractors.extract(frames224)

    # small representations for CNN training / SSIM / frame-diff / Katna matching
    rgb96  = np.stack([cv2.resize(f, (96, 96)) for f in frames224]).astype(np.uint8)
    gray64 = np.stack([
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (64, 64)) for f in frames224
    ]).astype(np.uint8)

    np.savez_compressed(
        cache_path(video_id),
        indices=np.asarray(indices),
        labels=labels,
        n_gt=n_gt,
        gt_sampled=np.asarray(gt_sampled, dtype=np.int64),
        resnet=resnet_feat.astype(np.float32),
        clip=clip_feat.astype(np.float32),
        rgb96=rgb96,
        gray64=gray64,
        total=total,
        fps=fps,
        video_name=video_name,
    )
    return True


def load_cache(video_id):
    d = np.load(cache_path(video_id), allow_pickle=True)
    return {
        "video_id":  video_id,
        "indices":   d["indices"],
        "labels":    d["labels"].astype(np.float32),
        "n_gt":      int(d["n_gt"]),
        "gt_sampled": d["gt_sampled"].astype(np.int64) if "gt_sampled" in d.files
                      else np.array([], dtype=np.int64),
        "resnet":    d["resnet"].astype(np.float32),
        "clip":      d["clip"].astype(np.float32),
        "rgb96":     d["rgb96"],
        "gray64":    d["gray64"],
        "total":     int(d["total"]),
        "fps":       float(d["fps"]),
        "video_name": str(d["video_name"]),
    }


def ensure_cache(records_meta, video_dir, rebuild=False):
    """
    records_meta: list of (video_id, video_name, gt_frames).
    Builds cache for all; returns the list of video_ids successfully cached.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    todo = [m for m in records_meta if rebuild or not os.path.exists(cache_path(m[0]))]
    ok_ids = [m[0] for m in records_meta if os.path.exists(cache_path(m[0])) and not rebuild]

    if todo:
        log(f"  Building cache for {len(todo)} video(s) "
            f"({len(records_meta) - len(todo)} already cached)…")
        extractors = SharedExtractors()
        for vid, name, gt in tqdm(todo, desc="  cache", dynamic_ncols=True):
            try:
                if build_cache_for(vid, name, gt, extractors, video_dir):
                    ok_ids.append(vid)
            except Exception as e:
                log(f"    ! cache failed for {name}: {e}")
        del extractors
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return sorted(set(ok_ids))


# ============================================================
# 5. METRICS
# ============================================================
def topk_binary(scores, k):
    """Binary vector selecting the top-k highest scores (dynamic top-k)."""
    k = max(1, min(int(k), len(scores)))
    pred = np.zeros(len(scores), dtype=np.int64)
    pred[np.argsort(scores)[-k:]] = 1
    return pred


def dtw_distance(a, b):
    """
    Dynamic Time Warping distance between two 1-D sequences of frame positions
    (cost = |a_i - b_j|). Used as a temporal-alignment score between a method's
    selected keyframe positions and the GT keyframe positions.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def weighted_bce_loss(scores, labels, w_pos=W_KEYFRAME, w_neg=W_TRANSITION):
    """Class-weighted BCE (keyframe frames weighted high, transitions low)."""
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=float)
    bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    w = np.where(y == 1, w_pos, w_neg)
    return float(np.sum(w * bce) / np.sum(w))


def focal_loss_value(scores, labels, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    """Binary focal loss (v10 formulation), mean over frames."""
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=float)
    pos = -alpha * (1 - p) ** gamma * y * np.log(p)
    neg = -(1 - alpha) * p ** gamma * (1 - y) * np.log(1 - p)
    return float(np.mean(pos + neg))


def compute_metrics(scores_by_video, records):
    """
    scores_by_video: {video_id: np.ndarray[T]}  (higher = more keyframe-like)
    records:         {video_id: cache dict}
    Returns (metric_dict, per_clip_dict) where per_clip_dict maps
    video_id -> {"f1": ..., "ap": ..., "p9": ..., "r9": ..., "dtw": ..., "mate": ...}
    (a metric is only present for a clip if it was computable for that clip,
    e.g. AP/F1 require at least one positive label). per_clip_dict is what you
    need for paired significance tests (Wilcoxon / paired t-test / Friedman) —
    the aggregate metric_dict alone gives you n=1 per method, which you can't
    test anything with.
    """
    pooled_pred, pooled_lab, pooled_score, pooled_w = [], [], [], []
    per_clip_ap, per_clip_f1, per_clip_dtw = [], [], []
    per_clip_p9, per_clip_r9, per_clip_mate = [], [], []
    per_clip = {}

    for vid, scores in scores_by_video.items():
        rec = records[vid]
        lab = rec["labels"].astype(np.int64)
        if len(scores) != len(lab):
            # guard: align lengths (should not happen)
            m = min(len(scores), len(lab))
            scores, lab = scores[:m], lab[:m]
        k = max(1, rec["n_gt"])
        pred = topk_binary(scores, k)
        clip_metrics = {}

        # Precision@9 / Recall@9: top-9 selected frames vs soft-window positives
        k9 = min(9, len(scores))
        top9 = np.argsort(scores)[-k9:]
        tp9 = float(lab[top9].sum())
        p9 = tp9 / k9
        r9 = tp9 / max(1, int(lab.sum()))
        per_clip_p9.append(p9); per_clip_r9.append(r9)
        clip_metrics["p9"] = p9; clip_metrics["r9"] = r9

        pooled_pred.extend(pred.tolist())
        pooled_lab.extend(lab.tolist())
        pooled_score.extend(np.asarray(scores, dtype=float).tolist())
        pooled_w.extend([W_KEYFRAME if y == 1 else W_TRANSITION for y in lab])

        if lab.sum() > 0:
            ap = average_precision_score(lab, scores)
            f1c = f1_score(lab, pred, zero_division=0)
            per_clip_ap.append(ap); per_clip_f1.append(f1c)
            clip_metrics["ap"] = ap; clip_metrics["f1"] = f1c

        # Temporal metrics: selected keyframe positions vs the GT keyframe positions
        gt_pos = np.asarray(rec.get("gt_sampled", []))
        if len(gt_pos) > 0:
            sel_pos = np.sort(np.argsort(scores)[-k:])
            dtw_v = dtw_distance(sel_pos, gt_pos) / max(len(sel_pos), len(gt_pos))
            mate_v = float(np.mean([np.min(np.abs(gt_pos - p)) for p in sel_pos]))
            per_clip_dtw.append(dtw_v); per_clip_mate.append(mate_v)
            clip_metrics["dtw"] = dtw_v; clip_metrics["mate"] = mate_v

        per_clip[vid] = clip_metrics

    pooled_lab   = np.asarray(pooled_lab)
    pooled_pred  = np.asarray(pooled_pred)
    pooled_score = np.asarray(pooled_score)
    pooled_w     = np.asarray(pooled_w)

    metrics = {
        "f1":          float(f1_score(pooled_lab, pooled_pred, zero_division=0)),
        "accuracy":    float(accuracy_score(pooled_lab, pooled_pred)),
        "precision":   float(precision_score(pooled_lab, pooled_pred, zero_division=0)),
        "recall":      float(recall_score(pooled_lab, pooled_pred, zero_division=0)),
        "weighted_f1": float(f1_score(pooled_lab, pooled_pred,
                                      sample_weight=pooled_w, zero_division=0)),
        "mAP":         float(np.mean(per_clip_ap)) if per_clip_ap else 0.0,
        "pooled_AP":   float(average_precision_score(pooled_lab, pooled_score))
                       if pooled_lab.sum() > 0 else 0.0,
        "precision@9": float(np.mean(per_clip_p9)) if per_clip_p9 else 0.0,
        "recall@9":    float(np.mean(per_clip_r9)) if per_clip_r9 else 0.0,
        "dtw":         float(np.mean(per_clip_dtw)) if per_clip_dtw else float("nan"),
        "mate":        float(np.mean(per_clip_mate)) if per_clip_mate else float("nan"),
        "mae":         float(np.mean(np.abs(pooled_lab - pooled_score))),
        "weighted_loss": weighted_bce_loss(pooled_score, pooled_lab),
        "focal_loss":  focal_loss_value(pooled_score, pooled_lab),
        "mean_clip_f1": float(np.mean(per_clip_f1)) if per_clip_f1 else 0.0,
        "n_clips":     len(scores_by_video),
        "n_frames":    int(len(pooled_lab)),
    }
    return metrics, per_clip


# ============================================================
# 6. SHARED HELPERS FOR BASELINES
# ============================================================
def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def kmedoids(D, k, n_iter=60, seed=SEED):
    """PAM-style k-medoids on a precomputed distance matrix D (n×n)."""
    n = D.shape[0]
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    medoids = list(rng.choice(n, size=k, replace=False))
    for _ in range(n_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new_medoids = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                new_medoids.append(medoids[c])
                continue
            costs = D[np.ix_(members, members)].sum(axis=1)
            new_medoids.append(int(members[np.argmin(costs)]))
        if set(new_medoids) == set(medoids):
            break
        medoids = new_medoids
    return medoids


def medoid_representativeness_score(feats, k):
    """
    k-medoids on features → per-frame score = closeness to nearest medoid.
    Medoids (the selected keyframes) score highest.
    """
    n = feats.shape[0]
    if n == 1:
        return np.array([1.0], dtype=float)
    # euclidean distance matrix
    sq = np.sum(feats ** 2, axis=1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * feats @ feats.T, 0.0))
    med = kmedoids(D, k)
    nearest = D[:, med].min(axis=1)
    return _minmax(-nearest)


# ============================================================
# 7. BASELINE METHODS
# ============================================================
# ---- 7.2 CLIP (ViT-B/32 embeddings + k-medoids) ----
def method_clip(records, train_records=None):
    out = {}
    for vid, rec in records.items():
        f = rec["clip"]
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        out[vid] = medoid_representativeness_score(f, max(1, rec["n_gt"]))
    return out


# ---- 7.4 K-medoids on ResNet features ----
def method_kmedoids(records, train_records=None):
    out = {}
    for vid, rec in records.items():
        out[vid] = medoid_representativeness_score(rec["resnet"], max(1, rec["n_gt"]))
    return out


# ---- 7.5 Nearest-neighbour (sequential leader) clustering ----
def method_nn_cluster(records, train_records=None):
    out = {}
    for vid, rec in records.items():
        f = rec["resnet"]
        n = f.shape[0]
        # adaptive threshold = mean consecutive distance
        if n > 1:
            cons = np.linalg.norm(f[1:] - f[:-1], axis=1)
            thr = float(np.mean(cons)) + 1e-6
        else:
            thr = 1.0
        clusters = [[0]]
        centroids = [f[0].copy()]
        for i in range(1, n):
            d = np.linalg.norm(f[i] - np.stack(centroids), axis=1)
            j = int(np.argmin(d))
            if d[j] <= thr:
                clusters[j].append(i)
                centroids[j] = f[clusters[j]].mean(axis=0)
            else:
                clusters.append([i])
                centroids.append(f[i].copy())
        # score: representatives (near own centroid) in large clusters rank high
        score = np.zeros(n, dtype=float)
        for c, members in enumerate(clusters):
            m = np.asarray(members)
            d = np.linalg.norm(f[m] - centroids[c], axis=1)
            score[m] = (-d) + np.log(len(members) + 1.0)
        out[vid] = _minmax(score)
    return out


# ---- 7.7 Cosine + SSIM (consecutive-frame stability) ----
def method_cosine_ssim(records, train_records=None, alpha=0.5, beta=0.5):
    out = {}
    for vid, rec in records.items():
        f = rec["resnet"]
        g = rec["gray64"]
        n = f.shape[0]
        if n == 1:
            out[vid] = np.array([1.0]); continue
        fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        dissim = np.zeros(n, dtype=float)
        for i in range(1, n):
            cos = float(np.dot(fn[i], fn[i - 1]))
            s   = ssim_fn(g[i - 1], g[i], data_range=255)
            dissim[i] = alpha * (1 - cos) + beta * (1 - s)
        dissim[0] = dissim[1]
        # stable "hold" frames (low local dissimilarity) = keyframes
        stability = -0.5 * (dissim + np.r_[dissim[1:], dissim[-1]])
        out[vid] = _minmax(stability)
    return out


# ---- 7.6 Lightweight SAM (MobileSAM image-encoder embeddings) ----
_SAM_STATE = {"model": None, "ok": None, "reason": ""}


def _init_sam():
    if _SAM_STATE["ok"] is not None:
        return _SAM_STATE["ok"]
    try:
        from ultralytics import SAM
        model = SAM("mobile_sam.pt")            # auto-downloads ~40 MB weight
        model.model.to(DEVICE).eval()
        _SAM_STATE["model"] = model
        _SAM_STATE["ok"] = True
    except Exception as e:
        _SAM_STATE["ok"] = False
        _SAM_STATE["reason"] = f"{type(e).__name__}: {e}"
    return _SAM_STATE["ok"]


@torch.no_grad()
def _sam_embeddings(rgb96, batch=4):
    """Per-frame pooled MobileSAM image-encoder embedding (256-d), batched."""
    enc = _SAM_STATE["model"].model.image_encoder
    mean = torch.tensor([123.675, 116.28, 103.53], device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor([58.395, 57.12, 57.375], device=DEVICE).view(1, 3, 1, 1)
    imgs = [cv2.resize(f, (1024, 1024)) for f in rgb96]
    feats = []
    for i in range(0, len(imgs), batch):
        chunk = np.stack(imgs[i:i + batch]).astype(np.float32)   # (b,1024,1024,3)
        t = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(DEVICE)
        t = (t - mean) / std
        emb = enc(t)                                             # (b,256,64,64)
        feats.append(emb.mean(dim=(2, 3)).cpu().numpy())
        del t, emb
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(feats)


def method_sam(records, train_records=None):
    if not _init_sam():
        log(f"    ! SAM unavailable ({_SAM_STATE['reason']}) — skipping.")
        return None
    out = {}
    for vid, rec in tqdm(records.items(), desc="    sam", dynamic_ncols=True, leave=False):
        try:
            emb = _sam_embeddings(rec["rgb96"])
            out[vid] = medoid_representativeness_score(emb, max(1, rec["n_gt"]))
        except Exception as e:
            log(f"    ! SAM failed on {rec['video_name']}: {e}")
            return None
    return out


# ---- 7.1 Katna (isolated env via `uv run --no-project --with katna`) ----
# Katna decodes/clusters every full-video frame → very slow (~80 s/video), so we
# fan it out across N isolated worker subprocesses and cache each video's result
# on disk (.cache_baseline/katna_scores/<vid>.json), reused across scopes/runs.
KATNA_WORKERS = 3          # concurrent Katna subprocesses (each also uses internal MP)
KATNA_VIDEO_TIMEOUT = 300  # s per video; killed & left uncached if exceeded (hang guard)


def _katna_score_dir():
    d = os.path.join(CACHE_DIR, "katna_scores")
    os.makedirs(d, exist_ok=True)
    return d


def _katna_launch(records, vid, slot):
    """Launch one isolated Katna subprocess for a single video on a given slot.
    Each slot has its own CWD/req/res so nothing collides across concurrent slots."""
    cache_abs  = os.path.abspath(CACHE_DIR)
    script_abs = os.path.abspath("katna_extract.py")
    req = {vid: {"n_gt": int(records[vid]["n_gt"]),
                 "path": os.path.abspath(video_path_for(records[vid]["video_name"], VIDEO_DIR))}}
    req_path = os.path.join(cache_abs, f"_katna_req_{slot}.json")
    res_path = os.path.join(cache_abs, f"_katna_res_{slot}.json")
    with open(req_path, "w") as fh:
        json.dump(req, fh)
    workdir = os.path.join(cache_abs, f"katna_work_{slot}")
    os.makedirs(workdir, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENCV_NUM_THREADS="2")
    cmd = ["uv", "run", "--no-project", "--with", "katna",
           "python", script_abs, req_path, res_path, cache_abs]
    return subprocess.Popen(cmd, cwd=workdir, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _run_katna_pool(records, todo, n_workers, timeout=KATNA_VIDEO_TIMEOUT):
    """One subprocess per video, <= n_workers concurrent, each bounded by `timeout`.
    Isolating videos means a single deadlock-prone clip is killed on timeout instead
    of stalling a whole shard (Katna's internal MP can deadlock on Windows)."""
    pending = list(todo)
    free = list(range(n_workers))
    active = {}                       # slot -> (proc, vid, start_time)
    while pending or active:
        while pending and free:
            slot = free.pop()
            vid = pending.pop(0)
            active[slot] = (_katna_launch(records, vid, slot), vid, time.monotonic())
        for slot, (proc, vid, start) in list(active.items()):
            if proc.poll() is not None:
                del active[slot]; free.append(slot)
            elif time.monotonic() - start > timeout:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                del active[slot]; free.append(slot)
        time.sleep(1)


def method_katna(records, train_records=None):
    kf_dir = _katna_score_dir()
    ids = list(records.keys())

    def uncached():
        return [v for v in ids if not os.path.exists(os.path.join(kf_dir, v + ".json"))]

    todo = uncached()
    if todo:
        log(f"    Katna: {len(todo)} uncached / {len(ids)} videos "
            f"({min(KATNA_WORKERS, len(todo))} workers, {KATNA_VIDEO_TIMEOUT}s/video)…")
        _run_katna_pool(records, todo, min(KATNA_WORKERS, len(todo)))
        still = uncached()
        if still:
            log(f"    Katna: retrying {len(still)} straggler(s)…")
            _run_katna_pool(records, still, min(2, len(still)))
        missing = uncached()
        if missing:
            log(f"    ! Katna could not process {len(missing)} video(s): "
                f"{missing[:6]}{'…' if len(missing) > 6 else ''} (scored 0)")

    out, n_have = {}, 0
    for vid in ids:
        rec = records[vid]
        T = len(rec["labels"])
        score = np.zeros(T, dtype=float)
        p = os.path.join(kf_dir, vid + ".json")
        if os.path.exists(p):
            with open(p) as fh:
                e = json.load(fh)
            prox = e.get("frame_score")
            if prox is not None and len(prox) == T:
                score = _minmax(np.asarray(prox, dtype=float))
            for s in e.get("selected_idx", []):
                if 0 <= s < T:
                    score[s] = 1.0
            n_have += 1
        out[vid] = score
    if n_have == 0:
        log("    ! Katna produced no results.")
        return None
    return out


# ---- 7.3 RL: DR-DSN (unsupervised diversity-representativeness) ----
class DSN(nn.Module):
    def __init__(self, in_dim=2048, hidden=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, 512)
        self.lstm = nn.LSTM(512, hidden, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(hidden * 2, 1)

    def forward(self, x):                       # x: (1, T, in_dim)
        h = torch.relu(self.proj(x))
        h, _ = self.lstm(h)
        return torch.sigmoid(self.fc(h)).squeeze(-1)   # (1, T)


def _drdsn_reward(feats_t, probs, ignore=20):
    """Diversity + representativeness reward (Zhou et al. 2018)."""
    T = feats_t.shape[0]
    m = torch.bernoulli(probs).squeeze(0)
    idx = m.nonzero(as_tuple=True)[0]
    if len(idx) < 2:
        return torch.tensor(0.0, device=feats_t.device), m
    fn = feats_t / (feats_t.norm(dim=1, keepdim=True) + 1e-8)
    # diversity
    sel = fn[idx]
    S = sel @ sel.T
    d = 1 - S
    ii, jj = torch.meshgrid(idx, idx, indexing="ij")
    far = (ii - jj).abs() > ignore
    eye = torch.eye(len(idx), device=feats_t.device).bool()
    mask = (~eye)
    dvals = d.clone()
    dvals[far] = 1.0
    R_div = dvals[mask].mean()
    # representativeness
    dist = torch.cdist(feats_t, feats_t[idx])       # (T, |sel|)
    R_rep = torch.exp(-dist.min(dim=1).values.mean())
    return R_div + R_rep, m


def method_rl(records, train_records, epochs=6, episodes=5, sigma=0.15, lr=1e-4):
    torch.manual_seed(SEED)
    model = DSN().to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    train_ids = list(train_records.keys())
    baselines = {vid: 0.0 for vid in train_ids}

    best_reward = -float("inf")
    best_state = None

    log(f"    training DR-DSN ({len(train_ids)} clips, {epochs} epochs)…")
    for ep in range(epochs):
        np.random.shuffle(train_ids)
        ep_reward = 0.0
        for vid in train_ids:
            feats = torch.from_numpy(train_records[vid]["resnet"]).to(DEVICE)
            x = feats.unsqueeze(0)
            probs = model(x)                        # (1, T)
            p = probs.clamp(1e-6, 1 - 1e-6)
            cost = 0.0
            rewards = []
            for _ in range(episodes):
                R, m = _drdsn_reward(feats, probs)
                rewards.append(R.item())
                logp = (m * torch.log(p.squeeze(0)) +
                        (1 - m) * torch.log(1 - p.squeeze(0))).sum()
                cost = cost - (R - baselines[vid]) * logp
            cost = cost / episodes
            # sparsity regularizer toward target keyframe ratio
            cost = cost + 1.0 * ((p.mean() - sigma) ** 2)
            opt.zero_grad(); cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            r = float(np.mean(rewards))
            baselines[vid] = 0.9 * baselines[vid] + 0.1 * r
            ep_reward += r
        mean_reward = ep_reward / max(len(train_ids), 1)
        log(f"      epoch {ep+1}/{epochs}  mean reward={mean_reward:.4f}")
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        save_path = os.path.join(MODELS_DIR, "rl_dsn_best.pth")
        torch.save({"model_state_dict": best_state, "mean_reward": best_reward}, save_path)
        log(f"    saved best DR-DSN model (mean_reward={best_reward:.4f}) → {save_path}")
        model.load_state_dict(best_state)

    model.eval()
    out = {}
    with torch.no_grad():
        for vid, rec in records.items():
            feats = torch.from_numpy(rec["resnet"]).to(DEVICE).unsqueeze(0)
            out[vid] = model(feats).squeeze(0).cpu().numpy()
    return out


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


def _frames_tensor(rgb96):
    t = torch.from_numpy(rgb96).permute(0, 3, 1, 2).float() / 255.0
    return _norm(F.interpolate(t, size=96, mode="bilinear", align_corners=False)
                 if t.shape[-1] != 96 else t)


def _train_frame_cnn(train_records, epochs=8, bs=128, lr=1e-3):
    torch.manual_seed(SEED)
    X, Y = [], []
    for rec in train_records.values():
        X.append(rec["rgb96"]); Y.append(rec["labels"])
    X = np.concatenate(X); Y = np.concatenate(Y).astype(np.float32)
    pos = max(Y.sum(), 1.0); neg = max(len(Y) - Y.sum(), 1.0)
    pos_weight = torch.tensor([neg / pos], device=DEVICE)

    model = FrameCNN().to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xt = _frames_tensor(X); Yt = torch.from_numpy(Y)
    n = len(Yt)
    best_loss = float("inf")
    best_state = None
    log(f"    training FrameCNN ({n} frames, {epochs} epochs)…")
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb = Xt[b].to(DEVICE); yb = Yt[b].to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        ep_loss = tot / n
        log(f"      epoch {ep+1}/{epochs}  loss={ep_loss:.4f}")
        if ep_loss < best_loss:
            best_loss = ep_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        save_path = os.path.join(MODELS_DIR, "frame_cnn_best.pth")
        torch.save({"model_state_dict": best_state, "loss": best_loss}, save_path)
        log(f"    saved best FrameCNN model (loss={best_loss:.4f}) → {save_path}")
        model.load_state_dict(best_state)

    model.eval()
    return model


@torch.no_grad()
def _cnn_probs(model, rgb96, bs=128):
    Xt = _frames_tensor(rgb96)
    probs = []
    for i in range(0, len(Xt), bs):
        probs.append(torch.sigmoid(model(Xt[i:i + bs].to(DEVICE))).cpu().numpy())
    return np.concatenate(probs)


_CNN_CACHE = {"model": None}


def method_cnn(records, train_records):
    if _CNN_CACHE["model"] is None:
        _CNN_CACHE["model"] = _train_frame_cnn(train_records)
    model = _CNN_CACHE["model"]
    return {vid: _cnn_probs(model, rec["rgb96"]) for vid, rec in records.items()}


# ---- 7.8 Uniform + threshold + majority voting + SVM/CNN ----
def _frame_diff_cue(gray64):
    """Low motion (stable hold) → high keyframe cue."""
    n = len(gray64)
    m = np.zeros(n, dtype=float)
    for i in range(1, n):
        m[i] = np.mean(np.abs(gray64[i].astype(float) - gray64[i - 1].astype(float)))
    m[0] = m[1] if n > 1 else 0.0
    return 1.0 - _minmax(m)


_SVM_CACHE = {"scaler": None, "pca": None, "svm": None}


def _train_svm(train_records):
    X, Y = [], []
    for rec in train_records.values():
        X.append(rec["resnet"]); Y.append(rec["labels"])
    X = np.concatenate(X); Y = np.concatenate(Y).astype(int)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=min(64, Xs.shape[1], Xs.shape[0])).fit(Xs)
    Xp = pca.transform(Xs)
    log(f"    training SVM ({len(Y)} frames)…")
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced",
              random_state=SEED).fit(Xp, Y)
    _SVM_CACHE.update(scaler=scaler, pca=pca, svm=svm)

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "svm_uniform_best.pkl")
    with open(save_path, "wb") as fh:
        pickle.dump({"scaler": scaler, "pca": pca, "svm": svm}, fh)
    log(f"    saved best SVM model (scaler+pca+svm) → {save_path}")


def _svm_probs(rec):
    Xp = _SVM_CACHE["pca"].transform(_SVM_CACHE["scaler"].transform(rec["resnet"]))
    return _SVM_CACHE["svm"].predict_proba(Xp)[:, 1]


def method_uniform_svmcnn(records, train_records):
    if _SVM_CACHE["svm"] is None:
        _train_svm(train_records)
    if _CNN_CACHE["model"] is None:
        _CNN_CACHE["model"] = _train_frame_cnn(train_records)
    cnn = _CNN_CACHE["model"]
    out = {}
    for vid, rec in records.items():
        cue = _frame_diff_cue(rec["gray64"])          # threshold/motion cue
        p_svm = _svm_probs(rec)                         # SVM vote
        p_cnn = _cnn_probs(cnn, rec["rgb96"])           # CNN vote
        # soft majority vote (mean of the three normalized cues)
        out[vid] = (_minmax(cue) + _minmax(p_svm) + _minmax(p_cnn)) / 3.0
    return out


METHOD_FUNCS = {
    "katna":          method_katna,
    "clip":           method_clip,
    "rl":             method_rl,
    "kmedoids":       method_kmedoids,
    "nn_cluster":     method_nn_cluster,
    "sam":            method_sam,
    "cosine_ssim":    method_cosine_ssim,
    "uniform_svmcnn": method_uniform_svmcnn,
    "cnn":            method_cnn,
}

TRAINABLE = {"rl", "cnn", "uniform_svmcnn"}


# ============================================================
# 8. RUNNER
# ============================================================
def run_scope(scope_name, eval_ids, all_records, train_records, methods):
    log(f"\n{'='*70}\n  SCOPE: {scope_name}  ({len(eval_ids)} videos)\n{'='*70}")
    eval_records = {vid: all_records[vid] for vid in eval_ids}
    results = {}
    per_clip_by_method = {}
    for name in methods:
        fn = METHOD_FUNCS[name]
        log(f"\n  ▶ {METHOD_LABELS[name]}")
        t0 = time.time()
        try:
            if name in TRAINABLE:
                scores = fn(eval_records, train_records)
            else:
                scores = fn(eval_records, train_records)
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"    ! {name} crashed: {e}")
            scores = None
        if not scores:
            results[name] = {"available": False}
            continue
        m, per_clip = compute_metrics(scores, eval_records)
        m["available"] = True
        m["seconds"] = round(time.time() - t0, 1)
        results[name] = m
        per_clip_by_method[name] = per_clip
        log(f"    F1={m['f1']:.3f} mAP={m['mAP']:.3f} P@9={m['precision@9']:.3f} "
            f"R@9={m['recall@9']:.3f} MAE={m['mae']:.3f} DTW={m['dtw']:.2f} "
            f"MATE={m['mate']:.2f} wLoss={m['weighted_loss']:.3f} "
            f"Focal={m['focal_loss']:.3f} ({m['seconds']}s)")
    return results, per_clip_by_method


def print_table(scope_name, results):
    log(f"\n{'='*94}")
    log(f"  RESULTS — {scope_name}")
    log(f"{'='*94}")
    header = (f"  {'Method':<44}{'F1':>6}{'Acc':>6}{'wF1':>6}{'mAP':>6}"
              f"{'P@9':>6}{'R@9':>6}{'MAE↓':>7}{'DTW↓':>7}{'MATE↓':>7}"
              f"{'wLoss↓':>8}{'Focal↓':>8}")
    log(header); log("  " + "-" * 123)
    ranked = sorted(results.items(),
                    key=lambda kv: (kv[1].get("mAP", -1) if kv[1].get("available") else -1),
                    reverse=True)
    for name, m in ranked:
        label = METHOD_LABELS[name][:43]
        if not m.get("available"):
            log(f"  {label:<44}{'—  (unavailable)':>49}")
            continue
        log(f"  {label:<44}{m['f1']:>6.3f}{m['accuracy']:>6.3f}{m['weighted_f1']:>6.3f}"
            f"{m['mAP']:>6.3f}{m['precision@9']:>6.3f}{m['recall@9']:>6.3f}"
            f"{m['mae']:>7.3f}{m['dtw']:>7.2f}{m['mate']:>7.2f}"
            f"{m['weighted_loss']:>8.3f}{m['focal_loss']:>8.3f}")
    log("  " + "-" * 123)
    log("  (P@9/R@9 = precision/recall of top-9 vs positives; ↓ lower better: "
        "MAE=|label-score|, DTW=warp dist, MATE=temporal err, wLoss=weighted BCE, Focal=focal)")


def save_per_clip_metrics(all_per_clip, meta):
    """
    all_per_clip: {scope: {method_key: {video_id: {metric: value}}}}
    Saved separately from the aggregate baseline_results.csv because this is
    what pairwise/omnibus significance tests need: matched per-clip values
    for each method, evaluated on the same clips.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "per_clip_metrics.json")
    with open(path, "w") as fh:
        json.dump({"meta": meta, "per_clip": all_per_clip}, fh, indent=2)
    log(f"  Saved per-clip metrics (for significance testing) → {path}")


def save_results(all_results, meta):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "baseline_results.json"), "w") as fh:
        json.dump({"meta": meta, "results": all_results}, fh, indent=2)
    rows = []
    for scope, res in all_results.items():
        for name, m in res.items():
            row = {"scope": scope, "method_key": name, "method": METHOD_LABELS[name]}
            row.update(m)
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    df.to_csv(csv_path, index=False)
    log(f"\n  Saved → {csv_path}")
    log(f"  Saved → {os.path.join(RESULTS_DIR, 'baseline_results.json')}")


def main():
    global VIDEO_DIR, CACHE_DIR, RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["test", "all", "both"], default="both")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS,
                    help=f"subset of {ALL_METHODS}")
    ap.add_argument("--video-dir", default=VIDEO_DIR)
    ap.add_argument("--gt-path", default=GT_PATH,
                    help="ground-truth table path (.xlsx/.xls/.csv)")
    ap.add_argument("--val-frac", type=float, default=0.15, help="val fraction")
    ap.add_argument("--test-frac", type=float, default=0.15, help="test fraction")
    ap.add_argument("--split-existing", action="store_true",
                    help="split only videos present under --video-dir")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--cache-dir", default=None, help="override cache dir")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #videos per scope")
    args = ap.parse_args()

    VIDEO_DIR   = args.video_dir
    RESULTS_DIR = args.out_dir
    CACHE_DIR   = args.cache_dir or f".cache_{re.sub(r'[^A-Za-z0-9]+', '_', VIDEO_DIR).strip('_')}"
    methods = [m for m in args.methods if m in METHOD_FUNCS]

    train_frac = max(0.0, 1.0 - args.val_frac - args.test_frac)
    split_tag  = f"{round(train_frac*100)}-{round(args.val_frac*100)}-{round(args.test_frac*100)}"
    log(f"Device: {DEVICE} | Video dir: {VIDEO_DIR} | Cache: {CACHE_DIR}")

    train_df, val_df, test_df = prepare_data(
        args.gt_path, val_frac=args.val_frac, test_frac=args.test_frac,
        split_existing=args.split_existing, video_dir=VIDEO_DIR)
    tot = len(train_df) + len(val_df) + len(test_df)
    log(f"Split {split_tag} (seed {SEED}, shuffle, "
        f"{'existing-only' if args.split_existing else 'full ground-truth table'}): "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)} (of {tot})")

    def meta_of(df):
        out = []
        for _, r in df.iterrows():
            name = str(r["Video"]).strip()
            out.append((name, name, parse_keyframes(r)))   # (video_id, name, gt)
        return out

    train_meta = meta_of(train_df)
    test_meta  = meta_of(test_df)
    need_all = args.scope in ("all", "both")
    all_meta = meta_of(pd.concat([train_df, val_df, test_df])) if need_all else []

    if args.limit:   # debug: cap cache-building + eval for a fast smoke test
        train_meta = train_meta[:args.limit]
        test_meta  = test_meta[:args.limit]
        all_meta   = all_meta[:args.limit]

    log("\nBuilding / loading feature cache…")
    train_ids = ensure_cache(train_meta, VIDEO_DIR, args.rebuild_cache)
    test_ids  = ensure_cache(test_meta,  VIDEO_DIR, args.rebuild_cache)
    all_ids   = ensure_cache(all_meta,   VIDEO_DIR, args.rebuild_cache) if need_all else []

    log(f"  Existing on disk → train={len(train_ids)} test={len(test_ids)}"
        + (f" all={len(all_ids)}" if need_all else ""))
    if not (train_ids or test_ids):
        log("No videos found on disk — check --video-dir."); sys.exit(1)

    needed = sorted(set(train_ids) | set(test_ids) | set(all_ids))
    all_records = {vid: load_cache(vid) for vid in needed}
    train_records = {vid: all_records[vid] for vid in train_ids}

    scopes = []
    if args.scope in ("test", "both"):
        scopes.append((f"test-split-{split_tag}", test_ids))
    if args.scope in ("all", "both"):
        scopes.append(("all-existing-videos", all_ids))

    if need_all and (TRAINABLE & set(methods) or "stake_v6" in methods):
        log("\n  ⚠ NOTE: the 'all-existing-videos' scope includes TRAIN videos, so "
            "supervised\n     methods (rl/cnn/uniform_svmcnn/stake_v6) are optimistic "
            "there.\n     Use the test-split scope for the unbiased comparison.")

    meta = {
        "device": str(DEVICE), "video_dir": VIDEO_DIR, "cache_dir": CACHE_DIR,
        "seed": SEED, "split": split_tag, "split_existing": args.split_existing,
        "target_fps": TARGET_FPS, "label_radius": LABEL_RADIUS,
        "w_keyframe": W_KEYFRAME, "w_transition": W_TRANSITION,
        "focal_alpha": FOCAL_ALPHA, "focal_gamma": FOCAL_GAMMA,
        "methods": methods,
        "counts": {"train": len(train_ids), "test": len(test_ids), "all": len(all_ids)},
    }

    all_results = {}
    all_per_clip = {}
    for scope_name, ids in scopes:
        if not ids:
            log(f"\n  (scope {scope_name} has 0 videos — skipped)"); continue
        res, per_clip = run_scope(scope_name, ids, all_records, train_records, methods)
        all_results[scope_name] = res
        all_per_clip[scope_name] = per_clip
        print_table(scope_name, res)
        save_results(all_results, meta)   # incremental: persist after each scope
        save_per_clip_metrics(all_per_clip, meta)

    log("\nDone.")

if __name__ == "__main__":
    main()