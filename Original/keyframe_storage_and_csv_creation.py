# ==========================================
# 1. INSTALL
# ==========================================
#!pip install transformers -q

# ==========================================
# 2. IMPORTS
# ==========================================
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from transformers import CLIPModel
from tqdm import tqdm
import pandas as pd

# ==========================================
# 3. PATHS (UPDATE IF NEEDED)
# ==========================================

VIDEO_ROOT = r"C:\Users\navne\Pictures\Kathakali\Mudrapedia"
WEIGHTS_PATH = r"C:\Users\navne\Pictures\Kathakali\Best_Models\best_MobileNetV2_v9.pth"
OUTPUT_DIR = r"C:\Users\navne\Pictures\Kathakali\Mudrapedia_keyframes"

excel_rows = []

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 4. CONFIG
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 9
TARGET_FPS = 5

# ==========================================
# 5. MODEL (MobileNetV2)
# ==========================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # CLIP
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
        for p in self.clip.parameters():
            p.requires_grad = False

        # MobileNetV2 backbone (1280-dim)
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Fusion
        self.fc = nn.Linear(1280 + 768, 512)

        # Temporal modeling
        self.lstm = nn.LSTM(
            512, 512,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )

        # Head (must match training)
        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        with torch.no_grad():
            cnn = self.pool(self.backbone(x)).view(x.size(0), -1)   # (B*T, 1280)
            clip = self.clip(pixel_values=x).pooler_output          # (B*T, 768)

        fused = torch.cat([cnn, clip], dim=1)

        x = torch.relu(self.fc(fused))
        x = x.view(b, t, -1)

        x, _ = self.lstm(x)
        return self.head(x).squeeze(-1)


# ==========================================
# 6. LOAD MODEL
# ==========================================

model = Model().to(DEVICE)

state_dict = torch.load(
    WEIGHTS_PATH,
    map_location=DEVICE
)

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)

print("Missing keys:")
print(missing)

print("\nUnexpected keys:")
print(unexpected)

model.eval()

print("MobileNetV2 model loaded")

# ==========================================
# 7. PREPROCESS
# ==========================================
normalize = transforms.Normalize(
    mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225]
)

def load_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {path}")
        return None, None, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total <= 0 or fps <= 0:
        cap.release()
        return None, None, None

    interval = max(1, int(fps / TARGET_FPS))
    indices = list(range(0, total, interval))

    frames_resized = []
    frames_original = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            continue

        frames_original.append(frame.copy())

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224,224))
        frames_resized.append(frame_resized)

    cap.release()

    if len(frames_resized) == 0:
        return None, None, None

    vid = torch.from_numpy(np.array(frames_resized)).permute(0,3,1,2).float()/255.0
    vid = normalize(vid)

    return vid.unsqueeze(0).to(DEVICE), frames_original, indices

# ==========================================
# 8. KEYFRAME EXTRACTION
# ==========================================
THRESHOLD = 0.5

def extract_keyframes(video_path):

    vid_tensor, raw_frames, indices = load_video(video_path)

    if vid_tensor is None:
        return

    with torch.no_grad():
        out = model(vid_tensor)
        probs = torch.sigmoid(out).cpu().numpy()[0]

    # class name from folder
    word = os.path.basename(os.path.dirname(video_path))

    # video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # save directory
    save_dir = os.path.join(OUTPUT_DIR, word, video_name)
    os.makedirs(save_dir, exist_ok=True)

    saved_count = 0

    for idx, prob in enumerate(probs):

        # keep only frames >= threshold
        if prob < THRESHOLD:
            continue

        if idx >= len(raw_frames):
            continue

        frame = raw_frames[idx]
        frame_id = int(indices[idx])

        save_path = os.path.join(
            save_dir,
            f"frame_{frame_id}_prob_{prob:.3f}.jpg"
        )

        cv2.imwrite(save_path, frame)

        # add row for excel
        excel_rows.append({
            "video_name": video_name,
            "frame_index": frame_id,
            "saved_path": save_path,
            "probability": float(prob),
            "class": word,
        })

        saved_count += 1

    print(f"{video_name}: saved {saved_count} keyframes")

# ==========================================
# 9. COLLECT VIDEOS
# ==========================================
video_list = []

for word in os.listdir(VIDEO_ROOT):
    word_path = os.path.join(VIDEO_ROOT, word)

    if not os.path.isdir(word_path):
        continue

    for v in os.listdir(word_path):
        if v.endswith(".mp4"):
            video_list.append(os.path.join(word_path, v))

print(f"Total videos: {len(video_list)}")

# ==========================================
# 10. RUN
# ==========================================
for vid in tqdm(video_list):
    extract_keyframes(vid)

print("Keyframe extraction completed")

# ==========================================
# 11. SAVE EXCEL
# ==========================================
df = pd.DataFrame(excel_rows)

excel_path = os.path.join(
    OUTPUT_DIR,
    "keyframe_summary.xlsx"
)

df.to_excel(excel_path, index=False)

print(f"Excel saved to: {excel_path}")