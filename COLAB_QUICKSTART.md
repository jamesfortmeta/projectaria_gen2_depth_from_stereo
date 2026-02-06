# Colab Quick Start (Do After Upload Completes)

## Pre-Flight Checklist

- [ ] Data uploaded to Google Drive at `MyDrive/aria_gen2_data/` (~1.5 hrs)
- [ ] Renamed folder from `aria_gen2_data_for_drive` to `aria_gen2_data`
- [ ] Ready to open Colab

---

## Step 1: Open Notebook in Colab (2 minutes)

1. Go to https://colab.research.google.com
2. **File → Upload notebook**
3. Select: `/Users/jamesfort/Documents/Projects/2026_Feb5/projectaria_gen2_depth_from_stereo/depth_from_stereo.ipynb`
4. **Runtime → Change runtime type → T4 GPU → Save**

---

## Step 2: Add Setup Cells (5 minutes)

Insert **3 new cells** at the very top of the notebook:

### Cell 0: Clone & Setup
```python
# Clone repository
!git clone https://github.com/jamesfortmeta/projectaria_gen2_depth_from_stereo
%cd projectaria_gen2_depth_from_stereo

# Install dependencies
!pip install -q projectaria-tools[all]==2.0.0
!pip install -q omegaconf flash-attn

print("✅ Setup complete")
```

### Cell 1: Mount Drive & Copy Data
```python
from google.colab import drive
import os
import shutil

# Mount Google Drive
drive.mount('/content/drive')

# Copy model weights
print("Copying model weights...")
os.makedirs('./FoundationStereo/pretrained_models', exist_ok=True)
shutil.copytree(
    '/content/drive/MyDrive/aria_gen2_data/23-51-11',
    './FoundationStereo/pretrained_models/23-51-11',
    dirs_exist_ok=True
)

# Copy dataset
print("Copying dataset...")
os.makedirs('./dataset', exist_ok=True)
shutil.copytree(
    '/content/drive/MyDrive/aria_gen2_data/cook_0',
    './dataset/cook_0',
    dirs_exist_ok=True
)

print("✅ Data ready")
print(f"Model: {os.path.exists('./FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth')}")
print(f"VRS: {os.path.exists('./dataset/cook_0/video.vrs')}")
```

### Cell 2: System Check
```python
import torch

print("="*60)
print("SYSTEM INFO")
print("="*60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("="*60)
```

---

## Step 3: Update VRS Path (1 minute)

Find the cell that starts with `# Configuration` and change:

```python
# FROM:
VRS_FILE_PATH = vrs_file_path if 'vrs_file_path' in dir() else "path/to/your/recording.vrs"

# TO:
VRS_FILE_PATH = "./dataset/cook_0/video.vrs"
```

---

## Step 4: Run All (10-15 minutes)

1. **Runtime → Run all**
2. Watch for errors
3. If error occurs:
   - Copy the full error traceback
   - Run Cell 2 (System Info)
   - Paste both to Claude locally

---

## Troubleshooting

### Error: "No module named 'rerun'"
**Fix:** Runtime → Restart session and run all

### Error: "CUDA out of memory"
**Fix:** In Configuration cell, change:
```python
OUTPUT_WIDTH = 256   # was 512
OUTPUT_HEIGHT = 256  # was 512
```

### Error: "File not found"
**Fix:** Check paths in Cell 1 match your Drive folder name exactly

### Need to Pull Code Updates
After Claude fixes something locally, add and run this cell:
```python
!git pull origin main
print("✅ Code updated")
```

---

## When You Hit an Error

**Copy this to Claude:**
```
Running on Colab, got this error:

[Run Cell 2, paste system info here]

Error in cell: [which cell/step]

[Paste full error traceback]
```

---

## Expected Runtime

| Step | Time |
|------|------|
| Setup cells | ~3 min |
| Data copy from Drive | ~2 min |
| FoundationStereo model load | ~30 sec |
| Inference (per frame) | ~5 sec |
| **Total first run** | **~10 min** |

---

## Success Checklist

- [ ] All cells run without errors
- [ ] Disparity map visualized
- [ ] Depth map visualized
- [ ] 3D point cloud displayed

## 🎉 If everything works, you're done!

## 😅 If you hit errors, copy them to Claude and I'll fix the code
