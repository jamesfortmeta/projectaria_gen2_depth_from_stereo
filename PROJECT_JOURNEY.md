# Project Journey: Aria Gen 2 Depth Estimation with FoundationStereo

## Goal

Run FoundationStereo (CVPR 2025 Best Paper Nominee) stereo depth estimation on sequences from Meta's Aria Gen 2 Pilot Dataset, with a fully automated debugging workflow.

---

## Phase 1: Hardware Assessment (Decided Against Local Execution)

### What I Did

1. **Checked my local machine**: MacBook Pro M1 Pro (16GB RAM, 14-core GPU)
2. **Researched FoundationStereo requirements**:
   - Requires NVIDIA CUDA (not available on Apple Silicon)
   - Memory: 2.3-18.5GB depending on model
   - Dependencies: xformers, flash-attn (CUDA-only)
3. **Researched Aria Gen 2 Pilot Dataset**:
   - Egocentric multimodal dataset with VRS sensor recordings
   - Includes MPS outputs (SLAM, semi-dense point clouds)
   - Multiple scenarios: cooking, cleaning, playing

### Decision

**Cannot run locally** - M1 Pro lacks NVIDIA GPU. Initially considered isolated GPU server, but ultimately chose **Google Colab** (free T4 GPU with 16GB VRAM).

---

## Phase 2: Repository Setup

### What I Did

1. **Forked the source repository**:
   - Original: https://github.com/facebookresearch/projectaria_gen2_depth_from_stereo
   - My fork: https://github.com/jamesfortmeta/projectaria_gen2_depth_from_stereo

2. **Cloned locally**:
   ```bash
   git clone https://github.com/jamesfortmeta/projectaria_gen2_depth_from_stereo.git
   cd projectaria_gen2_depth_from_stereo
   ```

3. **Integrated FoundationStereo code**:
   - Cloned FoundationStereo as submodule
   - Removed nested .git to integrate directly
   - Pushed to my fork

4. **Created .gitignore** to exclude large files:
   - Model weights (*.pth)
   - VRS recordings (*.vrs)
   - Dataset folders
   - MPS outputs (*.csv.gz)

---

## Phase 3: Data Acquisition

### What I Downloaded

1. **Sample VRS Recording** (255 MB):
   - Used for initial testing
   - Via `aria_dataset_downloader`

2. **Cook_0 Sequence** (5.5 GB total):
   - video.vrs (2.26 GB)
   - mps/slam/semidense_points.csv.gz (252 MB)
   - mps/slam/semidense_observations.csv.gz (2.9 GB)
   - Location: `./dataset/cook_0/`

3. **FoundationStereo Model Weights** (3.3 GB):
   - From Google Drive (rate-limited initially)
   - Manually downloaded via browser
   - Model: 23-51-11 (ViT-Large)
   - Files:
     - model_best_bp2.pth (3.3 GB)
     - cfg.yaml
   - Location: `./FoundationStereo/pretrained_models/23-51-11/`

### Git Authentication Fix

**Problem**: Push failed with "fatal: could not read Username"

**Solution**: Installed GitHub CLI and authenticated
```bash
brew install gh
gh auth login
```

---

## Phase 4: Google Colab Preparation

### What I Did

1. **Created upload folder** for Google Drive:
   - `aria_gen2_data_for_drive/` (8.3 GB)
   - Structure:
     ```
     aria_gen2_data_for_drive/
     ├── 23-51-11/           # Model weights
     │   ├── cfg.yaml
     │   └── model_best_bp2.pth
     └── cook_0/             # Dataset
         ├── video.vrs
         └── mps/slam/...
     ```

2. **Started Google Drive upload**:
   - Size: 8.3 GB
   - Estimated time: 1.5 hours
   - Status: In progress

3. **Modified Jupyter notebook**:
   - Added automated setup cells
   - Integrated with self-healing runner

---

## Phase 5: Workflow Evolution (Multiple Iterations)

### Iteration 1: Manual Notebook Cells
- **Approach**: Run cells manually in Colab
- **Problem**: Too much manual interaction
- **My feedback**: "Can't you just add the cell yourself? Why ask me?"

### Iteration 2: Automated Error Output
- **Approach**: Script with formatted error for copy-paste
- **Problem**: Still requires copying error from Colab to Claude
- **My feedback**: "Don't make me copy it"

### Iteration 3: Streamlined Copy-Paste
- **Approach**: Single copy command for errors
- **Problem**: Still one manual action per error
- **My feedback**: "One copy paste per error is too much. I want you to be able to iterate yourself on the error stream."

### Final Solution: Zero-Manual-Action Self-Healing Workflow ✅

**Key Innovation**: Use GitHub as communication bridge between Colab and Claude

**How it works**:
1. Colab runs pipeline in a loop
2. On error → auto-commits to GitHub branch `colab-errors`
3. Claude monitors GitHub locally via `watch_for_errors.sh`
4. Claude sees error → analyzes → fixes → pushes to `main`
5. Colab detects update → auto-pulls → retries
6. Repeats until success (max 10 attempts)

**Manual actions required**: **ZERO** (after one-time setup)

---

## Final Architecture

### Repository Structure

```
projectaria_gen2_depth_from_stereo/
├── FoundationStereo/                    # Stereo matching model
│   ├── core/
│   │   ├── foundation_stereo.py
│   │   └── utils/utils.py
│   └── pretrained_models/
│       └── 23-51-11/
│           ├── cfg.yaml
│           └── model_best_bp2.pth       # 3.3 GB (gitignored)
├── dataset/
│   └── cook_0/                          # Aria recording (gitignored)
│       ├── video.vrs                    # 2.26 GB
│       └── mps/slam/
├── self_healing_runner.py               # ⭐ Colab auto-fixer
├── watch_for_errors.sh                  # ⭐ Local error monitor
├── depth_from_stereo.ipynb              # Modified notebook
├── ZERO_ACTION_WORKFLOW.md              # ⭐ Complete guide
└── [various workflow docs]
```

### Key Files I Created

1. **self_healing_runner.py**:
   - Runs on Colab
   - Loops up to 10 attempts
   - Auto-commits errors to GitHub
   - Polls for fixes every 30s
   - Auto-retries when fix detected

2. **watch_for_errors.sh**:
   - Runs locally on my Mac
   - Polls GitHub every 15s
   - Shows Claude errors when detected
   - Waits for Claude to push fix

3. **ZERO_ACTION_WORKFLOW.md**:
   - Complete setup guide
   - GitHub token instructions
   - Colab secrets configuration
   - Full flow diagram with timeline

---

## Technical Challenges Solved

### Challenge 1: Apple Silicon Incompatibility
- **Problem**: FoundationStereo requires CUDA
- **Solution**: Use Google Colab (free T4 GPU)

### Challenge 2: Large File Management
- **Problem**: Model weights (3.3 GB), datasets (5.5 GB) can't go in Git
- **Solution**: Created .gitignore, separate upload folder for Google Drive

### Challenge 3: Colab ↔ Local Communication
- **Problem**: Can't access Colab filesystem from local machine
- **Solution**: Use GitHub as communication bridge with automated polling

### Challenge 4: Manual Iteration Overhead
- **Problem**: Copying errors manually is tedious
- **Solution**: Self-healing loop with GitHub-based automation

---

## Current Status

### ✅ Completed
- [x] Repository forked and setup
- [x] FoundationStereo code integrated
- [x] Aria Gen 2 dataset downloaded (cook_0)
- [x] Model weights downloaded (23-51-11)
- [x] Upload folder prepared (8.3 GB)
- [x] Self-healing workflow implemented
- [x] All code committed and pushed to GitHub

### 🔄 In Progress
- [ ] Google Drive upload (1.5 hours remaining)

### 📋 Next Steps (One-Time Setup - 3 minutes)

1. **Create GitHub Token** (2 min):
   - Visit: https://github.com/settings/tokens
   - Generate new token (classic)
   - Name: `Colab Auto-Runner`
   - Expiration: 90 days
   - Scopes: Check `repo` (all)
   - Copy token (starts with `ghp_...`)

2. **Add to Colab Secrets** (1 min):
   - Open: https://colab.research.google.com
   - Upload `depth_from_stereo.ipynb`
   - Click 🔑 icon → Secrets
   - Add: Name=`GITHUB_TOKEN`, Value=your token
   - Enable "Notebook access"

3. **Run Once and Walk Away**:
   - In Colab: Run setup cells + `!python self_healing_runner.py`
   - On local Mac: Run `./watch_for_errors.sh`
   - Claude monitors and auto-fixes errors
   - Come back to results

---

## Workflow Comparison

| Metric | Old Manual Workflow | New Self-Healing Workflow |
|--------|---------------------|---------------------------|
| Manual actions per error | 3 (copy + paste + git pull) | **0** |
| Time per iteration | ~2-5 min | ~1-3 min (automated) |
| With 5 errors | 15 manual actions | **0 manual actions** |
| Requires attention | Constant | **None** |
| Setup time | None | 3 min (one-time) |

---

## Key Decisions Made

1. **Colab over isolated server**: Easier integration with Claude, no server access needed
2. **GitHub as bridge**: Enables automated communication between Colab and local
3. **Self-healing loop**: Eliminates all manual actions during debugging
4. **Single upload folder**: Simplified Google Drive organization
5. **Cook_0 sequence**: Representative cooking scenario from dataset

---

## Technologies Used

- **FoundationStereo**: Zero-shot stereo matching (CVPR 2025)
- **Aria Gen 2 Pilot Dataset**: Egocentric multimodal recordings
- **Project Aria Tools**: VRS data provider, MPS utilities
- **Google Colab**: Free T4 GPU (16GB VRAM)
- **GitHub**: Version control + error communication bridge
- **Python**: Pipeline execution
- **Bash**: Automation scripts

---

## Lessons Learned

1. **Automation is worth the upfront investment**: Self-healing workflow saves hours over multiple iterations
2. **Use version control creatively**: GitHub can be more than just code storage
3. **Cloud GPU is viable for research**: Colab provides sufficient resources for free
4. **Minimize manual actions relentlessly**: Every copy-paste adds up over iterations
5. **Document as you go**: Clear workflows prevent confusion later

---

## Timeline

| Date | Milestone |
|------|-----------|
| Today | Initial research and planning |
| Today | Repository setup and FoundationStereo integration |
| Today | Dataset download (cook_0 sequence) |
| Today | Model weights acquisition (23-51-11) |
| Today | Upload folder creation |
| Today | Self-healing workflow implementation |
| **Now** | **Google Drive upload in progress** |
| Next | One-time Colab setup (3 min) |
| Next | First automated run |

---

## Success Criteria

The project will be considered successful when:

1. ✅ FoundationStereo runs on Aria Gen 2 data in Colab
2. ✅ Self-healing loop handles errors automatically
3. ✅ Zero manual actions required during iteration
4. ✅ Outputs saved: disparity.npy, depth.npy, visualizations

---

## Repository

**GitHub**: https://github.com/jamesfortmeta/projectaria_gen2_depth_from_stereo

**Key branches**:
- `main`: Production code
- `colab-errors`: Automated error reporting (ephemeral)

---

*Last updated: 2026-02-06*
*Total project time: ~3 hours (includes research, setup, downloads)*
*Automation time saved (estimated): 30-60 min per debugging session*
