# Setup Instructions for Isolated GPU Server

## What's Ready

| Item | Status | Location |
|------|--------|----------|
| Repo code | Committed (not pushed) | `projectaria_gen2_depth_from_stereo/` |
| FoundationStereo code | Committed (not pushed) | `FoundationStereo/` |
| cook_0 VRS | Downloaded | `dataset/cook_0/video.vrs` (2.26 GB) |
| Semi-dense points | Downloaded | `dataset/cook_0/mps/slam/` (~3.2 GB) |
| Model weights | **NEEDS DOWNLOAD** | See below |

## Step 1: Push Code to GitHub

```bash
cd /Users/jamesfort/Documents/Projects/2026_Feb5/projectaria_gen2_depth_from_stereo
git push origin main
```

## Step 2: Download Model Weights

Go to: https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf

Download the **23-51-11** folder containing:
- `cfg.yaml` (~1 KB)
- `model_best_bp2.pth` (~1.4 GB)

Place in:
```
FoundationStereo/pretrained_models/23-51-11/
├── cfg.yaml
└── model_best_bp2.pth
```

## Step 3: Transfer to Isolated Server

### Transfer Code (via Git)
```bash
# On the isolated server
git clone https://github.com/jamesfortmeta/projectaria_gen2_depth_from_stereo
```

### Transfer Large Files (via SCP/rsync)
From your Mac:
```bash
# Model weights (~1.4 GB)
scp -r FoundationStereo/pretrained_models/23-51-11/ user@server:/path/to/repo/FoundationStereo/pretrained_models/

# Dataset (~5.5 GB)
rsync -avz --progress dataset/cook_0/ user@server:/path/to/repo/dataset/cook_0/
```

## Step 4: Setup on GPU Server

```bash
cd projectaria_gen2_depth_from_stereo
conda env create -f FoundationStereo/environment.yml
conda activate foundation_stereo
pip install flash-attn
pip install projectaria-tools[all]
```

## Step 5: Run the Notebook

Update paths in the notebook:
- `VRS_FILE_PATH = "./dataset/cook_0/video.vrs"`

Then run `depth_from_stereo.ipynb`

## File Sizes Summary

| File | Size |
|------|------|
| model_best_bp2.pth | ~1.4 GB |
| video.vrs | 2.26 GB |
| semidense_points.csv.gz | 252 MB |
| semidense_observations.csv.gz | 2.9 GB |
| **Total to transfer** | **~6.8 GB** |
