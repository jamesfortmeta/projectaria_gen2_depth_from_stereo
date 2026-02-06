# Automated Colab Workflow - Zero Manual Copying

## The Automated Bridge

```
Colab runs → Error occurs → Auto-downloads → You say "read colab error" → Claude reads & fixes
```

**No manual copying of errors!**

---

## One-Time Setup (After Upload Completes)

### 1. Open Colab
- Go to: https://colab.research.google.com
- File → Upload notebook → `depth_from_stereo.ipynb`
- Runtime → T4 GPU

### 2. Add ONE Setup Cell at Top

```python
# === AUTOMATED SETUP CELL ===
# Clone repo
!git clone https://github.com/jamesfortmeta/projectaria_gen2_depth_from_stereo
%cd projectaria_gen2_depth_from_stereo

# Install deps
!pip install -q projectaria-tools[all]==2.0.0 omegaconf flash-attn

# Mount Drive
from google.colab import drive
import shutil
from pathlib import Path
drive.mount('/content/drive')

# Copy data
Path('./FoundationStereo/pretrained_models').mkdir(parents=True, exist_ok=True)
Path('./dataset').mkdir(exist_ok=True)

shutil.copytree('/content/drive/MyDrive/aria_gen2_data/23-51-11',
                './FoundationStereo/pretrained_models/23-51-11',
                dirs_exist_ok=True)
shutil.copytree('/content/drive/MyDrive/aria_gen2_data/cook_0',
                './dataset/cook_0',
                dirs_exist_ok=True)

print("✅ Setup complete")
```

### 3. Run The Pipeline

Add this cell:

```python
# === RUN PIPELINE ===
!python run_colab.py
```

---

## The Workflow Loop

### When It Works
🎉 Celebrate! Download your outputs from `/content/projectaria_gen2_depth_from_stereo/outputs/`

### When It Errors

**In Colab:**
1. Error auto-downloads to your Downloads folder as `colab_error.txt`
2. Nothing else to do in Colab

**Locally (with me):**
1. You say: **"read colab error"**
2. I run: `./read_colab_error.sh`
3. I analyze the error and fix the code
4. I commit & push the fix

**Back in Colab:**
Add and run this cell:
```python
!git pull origin main
!python run_colab.py  # Retry
```

**Repeat until success!**

---

## Example Session

### First Run (hits error)
```
You in Colab: [Run cell] !python run_colab.py
Colab: ❌ Error! Downloading colab_error.txt...

You locally: "Claude, read colab error"
Me: [Runs read_colab_error.sh, sees error]
Me: "I see the issue - missing import. Fixing..."
Me: [Fixes code, commits, pushes]

You in Colab: !git pull origin main
You in Colab: !python run_colab.py
Colab: ✅ SUCCESS!
```

---

## Commands Reference

| Situation | Command |
|-----------|---------|
| **First run** | `!python run_colab.py` |
| **After Claude fixes code** | `!git pull origin main` then retry |
| **Error occurred** | Say "read colab error" to Claude |
| **Check if data copied** | `!ls -lh dataset/cook_0/video.vrs` |

---

## File Locations

| File | Location |
|------|----------|
| Error log (auto-downloaded) | `~/Downloads/colab_error.txt` |
| Results (if success) | `/content/.../outputs/` in Colab |
| Runner script | `run_colab.py` (in repo) |

---

## Troubleshooting

### "Error file not found"
Error didn't auto-download. Manually download:
```python
from google.colab import files
files.download('colab_error.txt')
```

### "Can't pull from GitHub"
Reset:
```python
!rm -rf /content/projectaria_gen2_depth_from_stereo
# Re-run setup cell
```

### Out of memory
Reduce resolution - I'll tell you how when you report the error

---

## Why This Works

1. **Zero manual error copying** - Colab auto-downloads error file
2. **Claude reads locally** - Simple command I run
3. **Fast iteration** - Just pull & retry after each fix
4. **No authentication setup** - Uses public GitHub repo

---

## Ready to Run?

1. ✅ Wait for Google Drive upload (1.5 hrs)
2. ✅ Run setup cell in Colab
3. ✅ Run `!python run_colab.py`
4. ✅ If error → say "read colab error"
5. ✅ I fix → you pull → retry
