import os
import cv2
import pandas as pd

# ==========================
# CONFIG
# ==========================
VIDEO_ROOT = r"C:\Users\navne\Pictures\Kathakali\Videos_Cropped"
OUTPUT_CSV = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Automatic keyframe extraction\Video_all_frame_counts.csv"

# ==========================
# COUNT FRAMES
# ==========================
results = []

video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg")

for root, dirs, files in os.walk(VIDEO_ROOT):
    for file in files:
        if file.lower().endswith(video_extensions):
            video_path = os.path.join(root, file)

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Could not open: {video_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            class_name = os.path.basename(os.path.dirname(video_path))

            results.append({
                "class": class_name,
                "video": file,
                "frames": frame_count,
                "fps": round(fps, 2),
                "duration_seconds": round(duration, 2),
                "path": video_path
            })

            print(f"{file}: {frame_count} frames")

# ==========================
# SAVE RESULTS
# ==========================
df = pd.DataFrame(results)
df = df.sort_values(["class", "video"])

df.to_csv(OUTPUT_CSV, index=False)

print("\n====================================")
print(f"Processed {len(df)} videos")
print(f"Saved frame counts to {OUTPUT_CSV}")

print("\nFrame Count Statistics")
print(df["frames"].describe())