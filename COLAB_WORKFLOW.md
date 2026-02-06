# Colab + Claude Workflow Guide

## Overview

This workflow enables you to develop locally with Claude, run on Colab's GPU, and debug errors with Claude's help.

```
┌─────────────┐      git push      ┌──────────┐      git clone/pull      ┌─────────────┐
│  Local Mac  │ ──────────────────►│  GitHub  │◄───────────────────────│   Colab     │
│  + Claude   │◄────────────────── │          │                        │   (GPU)     │
│  (develop)  │   error logs       └──────────┘                        │   (execute) │
└─────────────┘                                                        └─────────────┘
```

---

## Initial Setup (One Time)

### 1. Upload Data to Google Drive

Create folder structure in Google Drive:
```
MyDrive/aria_gen2_data/
├── 23-51-11/
│   ├── cfg.yaml
│   └── model_best_bp2.pth
└── cook_0/
    ├── video.vrs
    └── mps/
        └── slam/
            ├── semidense_points.csv.gz
            └── semidense_observations.csv.gz
```

Upload from your Mac:
1. Go to drive.google.com
2. Create `aria_gen2_data` folder
3. Upload model weights + dataset (~8.8 GB total)

### 2. Open Notebook in Colab

1. Go to https://colab.research.google.com
2. File → Upload notebook → select `depth_from_stereo.ipynb` from Downloads
3. Runtime → Change runtime type → **T4 GPU**

### 3. Insert Workflow Cells

Copy the 5 utility cells from `colab_workflow_cells.py` to the TOP of your notebook:
- Cell 1: Clone repo
- Cell 2: Pull latest
- Cell 3: System info
- Cell 4: Error reporter
- Cell 5: Upload from Drive

### 4. Initial Run

Run cells in order:
1. Clone repo (Cell 1)
2. Upload data from Drive (Cell 5)
3. Run notebook cells sequentially

---

## Development Loop

### When Everything Works:
Celebrate! 🎉

### When You Hit an Error:

#### Step 1: Capture Error Info
Run the "System Info" cell (Cell 3), copy output

#### Step 2: Get Full Error
The error should show full traceback. Copy:
- Error type
- Error message
- Full traceback
- Which cell/step failed

#### Step 3: Report to Claude (Local)
Open Claude locally and paste:

```
I'm running FoundationStereo on Colab and got this error:

[PASTE SYSTEM INFO]

[PASTE ERROR + TRACEBACK]

The error occurred in: [CELL/STEP NAME]
```

#### Step 4: Claude Fixes Code
I'll:
- Analyze the error
- Identify the root cause
- Fix the code locally
- Commit and push to GitHub

#### Step 5: Pull Updates in Colab
Run Cell 2 (Pull latest), then retry the failed step

#### Step 6: Repeat Until Success
Iterate: error → report → fix → pull → retry

---

## Pro Tips

### Use Error Reporter Wrapper
For error-prone cells, wrap with error reporter:

```python
with error_reporter("Loading FoundationStereo model"):
    cfg = OmegaConf.load(cfg_path)
    model = FoundationStereo(cfg)
    model.cuda()
```

This gives formatted error output that's easy to copy to Claude.

### Save Intermediate Results
Colab sessions timeout after 90min idle. Save checkpoints:

```python
# After model loads successfully
torch.save({
    'model': model.state_dict(),
    'cfg': cfg
}, 'checkpoint.pth')

# Save outputs
np.savez('results.npz',
    disparity=disparity_map,
    depth=depth_map
)
```

### Monitor GPU Memory
Check GPU usage:
```python
!nvidia-smi
```

---

## Error Reporting Template

When reporting errors to Claude, use this format:

```
## Error Report

**Context:**
Running on Colab with T4 GPU

**System Info:**
[paste from Cell 3]

**What I was doing:**
[e.g., "Loading FoundationStereo model"]

**Error:**
```
[paste full traceback]
```

**Cell code that failed:**
```python
[paste the cell code]
```
```

---

## File Transfer Cheat Sheet

**From Mac to Google Drive:** Drag & drop via browser

**From Drive to Colab:** Run Cell 5 (mount_drive_and_copy)

**From Colab to Mac:**
```python
from google.colab import files
files.download('output.png')
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Session disconnected | Re-run all cells from top |
| GPU out of memory | Reduce resolution: `OUTPUT_WIDTH = 256` |
| `rerun` not found | Runtime → Restart session and run all |
| Files not found | Check paths in Cell 5 match your Drive |
| Git pull fails | Delete `/content/projectaria_gen2_depth_from_stereo`, re-clone |

---

## Quick Reference

**Start new session:**
1. Cell 1 (clone)
2. Cell 5 (upload data)
3. Run all cells

**After code fix:**
1. Cell 2 (pull latest)
2. Re-run failed cell

**Report error:**
1. Cell 3 (system info) → copy
2. Copy error traceback
3. Paste both to Claude locally
